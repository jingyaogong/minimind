import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "model"))

    conversation = [
        {"role": "user", "content": "MiniMind 是什么？"}
    ]

    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    config = MiniMindConfig(
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=512,
    )
    model = MiniMindForCausalLM(config).eval()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        generated_ids = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=8,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            top_p=1.0,
            top_k=0,
            temperature=1.0,
        )

    prompt_len = input_ids.shape[1]
    new_ids = generated_ids[0, prompt_len:]

    print("[Prompt]")
    print(prompt)
    print()

    print("[Encoded]")
    print(f"input_ids.shape={tuple(input_ids.shape)}")
    print(f"attention_mask.shape={tuple(attention_mask.shape)}")
    print(f"first_20_input_ids={input_ids[0, :20].tolist()}")
    print()

    print("[Forward]")
    print(f"logits.shape={tuple(outputs.logits.shape)}")
    print(f"last_position_logits.shape={tuple(outputs.logits[:, -1, :].shape)}")
    print()

    print("[Generate]")
    print(f"generated_ids.shape={tuple(generated_ids.shape)}")
    print(f"new_token_ids={new_ids.tolist()}")
    print(f"decoded_new_text={tokenizer.decode(new_ids, skip_special_tokens=True)!r}")
    print()
    print("Note: the model is randomly initialized, so decoded text is not meaningful.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
