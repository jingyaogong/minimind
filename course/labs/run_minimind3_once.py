import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Run one MiniMind-3 inference.")
    parser.add_argument("--model_path", default=str(ROOT / "minimind-3"))
    parser.add_argument("--prompt", default="MiniMind 是什么？")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--open_thinking", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model path not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True)
    model = model.eval().to(args.device)

    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=args.open_thinking,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(args.device)

    generate_kwargs = {
        "inputs": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        generated_ids = model.generate(**generate_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    response_ids = generated_ids[0, prompt_len:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    print("[Prompt]")
    print(args.prompt)
    print()
    print("[Shapes]")
    print(f"input_ids.shape={tuple(inputs['input_ids'].shape)}")
    print(f"generated_ids.shape={tuple(generated_ids.shape)}")
    print(f"new_tokens={response_ids.numel()}")
    print()
    print("[Response]")
    print(response.strip())


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
