import argparse
import os
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]


def show_encoding(tokenizer, text, title):
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded.input_ids
    raw_tokens = tokenizer.convert_ids_to_tokens(ids)

    print(f"[{title}]")
    print(f"text={text!r}")
    print(f"input_ids={ids}")
    print(f"raw_tokens={raw_tokens}")
    print(f"decoded_keep_special={tokenizer.decode(ids, skip_special_tokens=False)!r}")
    print(f"decoded_skip_special={tokenizer.decode(ids, skip_special_tokens=True)!r}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Inspect MiniMind tokenizer behavior.")
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"))
    parser.add_argument("--text", default="MiniMind 是什么？")
    parser.add_argument("--open_thinking", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print("[Tokenizer]")
    print(f"path={args.tokenizer_path}")
    print(f"len(tokenizer)={len(tokenizer)}")
    print(f"vocab_size={tokenizer.vocab_size}")
    print(f"bos_token={tokenizer.bos_token!r}, bos_token_id={tokenizer.bos_token_id}")
    print(f"eos_token={tokenizer.eos_token!r}, eos_token_id={tokenizer.eos_token_id}")
    print(f"pad_token={tokenizer.pad_token!r}, pad_token_id={tokenizer.pad_token_id}")
    print(f"unk_token={tokenizer.unk_token!r}, unk_token_id={tokenizer.unk_token_id}")
    print()

    show_encoding(tokenizer, "<|im_start|>", "special token: im_start")
    show_encoding(tokenizer, "<|im_end|>", "special token: im_end")
    show_encoding(tokenizer, "<think>", "special token: think")
    show_encoding(tokenizer, "</think>", "special token: /think")
    show_encoding(tokenizer, args.text, "plain text")

    messages = [{"role": "user", "content": args.text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=args.open_thinking,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids

    print("[Chat Template]")
    print(f"open_thinking={args.open_thinking}")
    print(prompt)
    print()
    print("[Prompt Encoding]")
    print(f"prompt_token_count={len(prompt_ids)}")
    print(f"first_40_prompt_ids={prompt_ids[:40]}")
    print(f"first_40_raw_tokens={tokenizer.convert_ids_to_tokens(prompt_ids[:40])}")
    print()
    print("[Prompt Decode]")
    print(tokenizer.decode(prompt_ids, skip_special_tokens=False))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
