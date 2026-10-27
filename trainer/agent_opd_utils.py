"""Student-owned tool trajectories and token-level on-policy distillation."""

import json
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from trainer.train_agent import execute_tool, parse_tool_calls


@dataclass
class AgentTrajectory:
    input_ids: list
    assistant_mask: list
    turns: int
    tool_calls: int
    truncated: bool


def tool_observation(tokenizer, calls, tools, open_thinking=False):
    """Render only the environment response and the next assistant prefix."""
    marker = "<|agent_opd_observation_boundary|>"
    messages = [{"role": "assistant", "content": marker}]
    allowed = {tool["function"]["name"] for tool in (tools or [])}
    for call in calls:
        name, arguments = call.get("name", ""), call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        result = None
        if isinstance(name, str) and name in allowed and isinstance(arguments, dict):
            result = execute_tool(name, arguments)
        if result is None:
            result = {"error": "unknown tool or invalid arguments"}
        messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)[:2048]})
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        tools=tools, open_thinking=open_thinking,
    )
    _, found, observation = rendered.partition(marker)
    if not found:
        raise ValueError("The chat template must preserve assistant content")
    return tokenizer(observation, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def collect_trajectory(student, tokenizer, messages, tools=None, *, max_turns=3,
                       max_new_tokens=256, max_total_len=2048, temperature=1.0,
                       open_thinking=False, autocast_ctx=None):
    """Sample each assistant turn, execute local tools, and preserve sampled IDs.

    Only student-generated tokens (including each turn's first EOS) are targets.
    Environment observations remain visible context with no direct loss.
    """
    if min(max_turns, max_new_tokens, max_total_len) < 1 or temperature <= 0:
        raise ValueError("Generation limits and temperature must be positive")
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        tools=tools, open_thinking=open_thinking,
    )
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not ids or len(ids) >= max_total_len:
        raise ValueError("Prompt leaves no generation space; increase --max_total_len")
    mask = [False] * len(ids)
    model = getattr(student, "module", student)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    turns, call_count, truncated = 0, 0, False
    try:
        for turn in range(max_turns):
            inputs = torch.tensor([ids], device=device)
            with autocast_ctx if autocast_ctx is not None else nullcontext():
                output = model.generate(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    max_new_tokens=min(max_new_tokens, max_total_len - len(ids)),
                    do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                    repetition_penalty=1.0, num_beams=1, use_cache=True,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                )
            new_ids = output[0, len(ids):].tolist()
            if tokenizer.eos_token_id in new_ids:
                new_ids = new_ids[:new_ids.index(tokenizer.eos_token_id) + 1]
            if not new_ids:
                truncated = True
                break
            ids.extend(new_ids)
            mask.extend([True] * len(new_ids))
            turns += 1
            if new_ids[-1] != tokenizer.eos_token_id:
                truncated = True
                break
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            calls = [call for call in parse_tool_calls(text) if isinstance(call, dict)]
            call_count += len(calls)
            if not calls:
                break
            if turn + 1 == max_turns:
                truncated = True
                break
            observation = tool_observation(tokenizer, calls, tools, open_thinking)
            if observation[:1] == [tokenizer.eos_token_id]:
                observation = observation[1:]  # EOS is already in the sampled IDs.
            if len(ids) + len(observation) >= max_total_len:
                truncated = True
                break
            ids.extend(observation)
            mask.extend([False] * len(observation))
    finally:
        model.train(was_training)
    return AgentTrajectory(ids, mask, turns, call_count, truncated)


def pack_trajectories(trajectories, pad_token_id, device):
    if not trajectories:
        raise ValueError("At least one trajectory is required")
    if any(len(t.input_ids) != len(t.assistant_mask) for t in trajectories):
        raise ValueError("Trajectory tokens and masks must have the same length")
    length = max(len(t.input_ids) for t in trajectories)
    ids, attention, targets = [], [], []
    for trajectory in trajectories:
        size = len(trajectory.input_ids)
        ids.append(trajectory.input_ids + [pad_token_id] * (length - size))
        attention.append([1] * size + [0] * (length - size))
        targets.append(trajectory.assistant_mask + [False] * (length - size))
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(attention, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.bool, device=device)[:, 1:],
    )


def distillation_loss(student_logits, teacher_logits, target_mask,
                      temperature=1.0, direction="forward_kl"):
    """Full-vocabulary KL averaged over assistant tokens on sampled prefixes.

    Sampling is detached (the GKD semi-gradient); this is not a REINFORCE
    estimator of trajectory-level reverse KL. Teacher gradients are blocked.
    """
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3:
        raise ValueError("Student and teacher logits must share [batch, time, vocab]")
    if target_mask.shape != student_logits.shape[:2]:
        raise ValueError("Target mask must align with the shifted logits")
    if temperature <= 0 or direction not in {"forward_kl", "reverse_kl"}:
        raise ValueError("Invalid distillation temperature or KL direction")
    if not target_mask.any():
        return student_logits[:, :0, :].sum()
    student_logps = F.log_softmax(student_logits[target_mask].float() / temperature, dim=-1)
    teacher_logps = F.log_softmax(teacher_logits.detach()[target_mask].float() / temperature, dim=-1)
    if direction == "forward_kl":
        loss = F.kl_div(student_logps, teacher_logps, log_target=True, reduction="batchmean")
    else:
        loss = F.kl_div(teacher_logps, student_logps, log_target=True, reduction="batchmean")
    return loss * temperature ** 2
