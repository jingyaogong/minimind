import argparse
import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentic.data_analysis_env import (
    AgenticToolEnv,
    average_agentic_metrics,
    extract_final_answer,
    format_agentic_user_prompt,
    get_agentic_tools,
    parse_tool_calls,
    score_agentic_trajectory,
)
from dataset.agentic_dataset import load_agentic_jsonl


def apply_eval_model_profile(args):
    if not args.model_profile:
        return
    path = args.model_profiles_path
    with open(path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    if args.model_profile not in profiles:
        raise ValueError(f"Unknown model_profile={args.model_profile}. Available: {', '.join(profiles)}")
    profile = profiles[args.model_profile]
    for key, value in profile.items():
        if hasattr(args, key):
            setattr(args, key, value)


def load_model(args):
    import torch
    from transformers import AutoTokenizer

    from model.model_minimind import MiniMindForCausalLM
    from model.model_minimind_mla import MiniMindMLAForCausalLM
    from trainer.trainer_utils import build_lm_config, get_model_suffix, resolve_attention_type

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    config = build_lm_config(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_total_len,
        use_moe=bool(args.use_moe),
        attention_type=resolve_attention_type(args),
        kv_lora_rank=args.kv_lora_rank,
        q_lora_rank=args.q_lora_rank,
        rope_dim=args.rope_dim,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
    )
    model = MiniMindMLAForCausalLM(config) if args.attention_type == "mla" or args.use_mla else MiniMindForCausalLM(config)
    suffix = get_model_suffix(config)
    weight_path = os.path.join(args.save_dir, f"{args.weight}_{config.hidden_size}{suffix}.pth")
    weights = torch.load(weight_path, map_location=args.device)
    model.load_state_dict(weights, strict=False)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    model = model.to(args.device)
    if args.device != "cpu" and args.dtype != "float32":
        model = model.to(dtype=dtype)
    return model.eval(), tokenizer


def generate_once(model, tokenizer, messages, tools, args):
    import torch

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
        open_thinking=bool(args.open_thinking),
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
    if inputs["input_ids"].size(1) > args.max_total_len:
        inputs["input_ids"] = inputs["input_ids"][:, -args.max_total_len:]
        inputs["attention_mask"] = inputs["attention_mask"][:, -args.max_total_len:]
    st = time.time()
    with torch.no_grad():
        generated = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=bool(args.do_sample),
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(generated[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
    return text, time.time() - st


def run_episode(model, tokenizer, sample, args):
    tools = get_agentic_tools(sample.get("tools"))
    messages = [
        {"role": "system", "content": "你是一个面向运营数据分析的 Agent。"},
        {"role": "user", "content": format_agentic_user_prompt(sample)},
    ]
    env = AgenticToolEnv(sample, repo_root=args.repo_root, timeout=args.tool_timeout)
    turn_outputs = []
    elapsed = 0.0
    unfinished = False
    for turn in range(args.max_turns):
        text, dt = generate_once(model, tokenizer, messages, tools, args)
        elapsed += dt
        turn_outputs.append(text)
        calls = parse_tool_calls(text)
        if not calls:
            break
        unfinished = turn == args.max_turns - 1
        messages.append({"role": "assistant", "content": text})
        for call in calls:
            result = env.execute(call.get("name", ""), call.get("arguments", {}))
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
        if unfinished:
            break
    reward, parts = score_agentic_trajectory(
        turn_outputs,
        sample,
        repo_root=args.repo_root,
        unfinished=unfinished,
        execute_tools=True,
    )
    return {
        "id": sample.get("id", ""),
        "question": sample.get("question", ""),
        "prediction": extract_final_answer(turn_outputs),
        "turn_outputs": turn_outputs,
        "reward": reward,
        "metrics": parts,
        "latency_sec": elapsed,
    }


def eval_predictions(args, rows):
    pred_map = {}
    with open(args.pred, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                pred_map[item.get("id")] = item
    outputs = []
    for sample in rows:
        item = pred_map.get(sample.get("id"), {})
        turns = item.get("turn_outputs") or [item.get("prediction") or item.get("response") or item.get("output") or ""]
        reward, parts = score_agentic_trajectory(turns, sample, repo_root=args.repo_root)
        outputs.append(
            {
                "id": sample.get("id", ""),
                "question": sample.get("question", ""),
                "prediction": extract_final_answer(turns),
                "turn_outputs": turns,
                "reward": reward,
                "metrics": parts,
            }
        )
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Agentic DataAnalysis model or predictions")
    parser.add_argument("--data", required=True)
    parser.add_argument("--pred", default="")
    parser.add_argument("--output", default="reports/agentic_eval_predictions.jsonl")
    parser.add_argument("--json_output", default="reports/agentic_eval_metrics.json")
    parser.add_argument("--bad_cases_output", default="reports/agentic_eval_bad_cases.jsonl")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--weight", default="agent_grpo")
    parser.add_argument("--tokenizer_path", default="model")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--hidden_size", default=1024, type=int)
    parser.add_argument("--num_hidden_layers", default=24, type=int)
    parser.add_argument("--num_attention_heads", default=8, type=int)
    parser.add_argument("--num_key_value_heads", default=4, type=int)
    parser.add_argument("--intermediate_size", default=3264, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--attention_type", default="mla", choices=["gqa", "mha", "mqa", "mla"])
    parser.add_argument("--use_mla", default=0, type=int, choices=[0, 1])
    parser.add_argument("--kv_lora_rank", default=192, type=int)
    parser.add_argument("--q_lora_rank", default=384, type=int)
    parser.add_argument("--rope_dim", default=None, type=int)
    parser.add_argument("--max_total_len", default=2500, type=int)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--max_turns", default=4, type=int)
    parser.add_argument("--tool_timeout", default=3, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--do_sample", default=1, type=int)
    parser.add_argument("--open_thinking", default=0, type=int)
    parser.add_argument("--model_profile", type=str, default="")
    parser.add_argument("--model_profiles_path", type=str, default="configs/searchlm_profiles.json")
    return parser.parse_args()


def main():
    args = parse_args()
    apply_eval_model_profile(args)
    rows = load_agentic_jsonl(args.data)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    if args.pred:
        outputs = eval_predictions(args, rows)
    else:
        model, tokenizer = load_model(args)
        outputs = [run_episode(model, tokenizer, sample, args) for sample in rows]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.bad_cases_output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    metrics = average_agentic_metrics([item["metrics"] for item in outputs])
    metrics["samples"] = len(outputs)
    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(args.bad_cases_output, "w", encoding="utf-8") as f:
        for item in outputs:
            if item["metrics"].get("task_success", 0) < 1:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
