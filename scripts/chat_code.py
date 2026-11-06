"""Chat with a local MiniMind model and optionally run Python tool calls.

This is a convenience runner, not a security boundary. Always review code before
approving it, and only use it with a model you trust.
"""

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from execution import ExecutionResult, execute_code


MAX_TOOL_ROUNDS = 4
TOOL = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "Run a short Python program to calculate or verify something. "
            "The user must approve every execution. Do not use this for file, "
            "network, or system operations."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source code"}},
            "required": ["code"],
        },
    },
}


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for match in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            payload = json.loads(match.strip())
        except json.JSONDecodeError:
            # Let the model see a parse error and retry instead of presenting
            # malformed tool markup as a normal assistant answer.
            calls.append({"name": "__invalid_tool_call__", "arguments": {"raw": match.strip()}})
            continue
        if isinstance(payload, dict):
            arguments = payload.get("arguments")
            if isinstance(arguments, str):
                try:
                    payload["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            calls.append(payload)
    return calls


def confirm_and_execute(code: str, timeout: float) -> dict:
    print("\n\033[33mModel requested Python execution:\033[0m")
    print("\n".join("  " + line for line in (code.rstrip() or "(empty)").splitlines()))
    print("\nThis runs on your computer. The lightweight restrictions are not a secure sandbox.")
    try:
        approved = input("Run this code? [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        approved = False
        print()
    if not approved:
        return {"success": False, "error": "Execution declined by the user."}
    result = execute_code(code, timeout=timeout)
    return result.__dict__ if isinstance(result, ExecutionResult) else result


def generate(model, tokenizer, messages, args) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=[TOOL],
        open_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
    inputs = {key: value.to(args.device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Chat with MiniMind and approve Python tool calls.")
    parser.add_argument("--model", default="./minimind-3", help="Transformers-compatible local model directory or Hub ID")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=5.0, help="Python execution timeout in seconds")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    model.eval().to(args.device)

    print("\nMiniMind Code CLI")
    print("Type 'clear' to reset the conversation, or 'exit' to quit.")
    print("Python execution requires confirmation; this is not a secure sandbox.\n")
    system_prompt = (
        "You are MiniMind, a helpful assistant. Use execute_python only when useful. "
        "Keep code short; avoid file, network, and system operations. Explain results after execution. "
        'For tool calls, emit valid JSON exactly like this: <tool_call>{"name":"execute_python",'
        '"arguments":{"code":"print(2 + 3)"}}</tool_call>. Escape quotes and newlines in code strings.'
    )
    messages = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if user_text.lower() == "clear":
            messages = messages[:1]
            print("Conversation cleared.")
            continue
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})
        for _ in range(MAX_TOOL_ROUNDS + 1):
            try:
                answer = generate(model, tokenizer, messages, args)
            except Exception as exc:
                print(f"\nGeneration failed: {exc}")
                break
            calls = parse_tool_calls(answer)
            messages.append({"role": "assistant", "content": answer})
            if not calls:
                print(f"MiniMind: {answer.strip()}\n")
                break

            if len(calls) > 1:
                print("Only one tool call is handled per model turn.")
            for call in calls[:1]:
                name = call.get("name")
                arguments = call.get("arguments", {})
                if name == "__invalid_tool_call__":
                    result = {
                        "success": False,
                        "error": "Tool call was not valid JSON. Retry with a valid execute_python call, e.g. code='print(2 + 3)'.",
                    }
                elif name != "execute_python" or not isinstance(arguments, dict):
                    result = {"success": False, "error": f"Unsupported or malformed tool call: {name!r}."}
                else:
                    code = arguments.get("code", "")
                    result = confirm_and_execute(code if isinstance(code, str) else "", args.timeout)
                print(f"Tool result: {json.dumps(result, ensure_ascii=False)}\n")
                messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
        else:
            print("Stopped after the maximum number of tool rounds.\n")


if __name__ == "__main__":
    main()
