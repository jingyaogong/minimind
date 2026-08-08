import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.serve_openai_api import parse_response  # noqa: E402


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        },
    }
]


def main():
    model_dir = ROOT / "minimind-3"
    print(f"project_root={ROOT}")
    print(f"model_dir={model_dir}")
    print()

    print("[Transformers model files]")
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]:
        path = model_dir / name
        print(f"{name}: exists={path.exists()} size={path.stat().st_size if path.exists() else 0}")

    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    print()
    print("[Config summary]")
    for key in ["model_type", "architectures", "hidden_size", "num_hidden_layers", "vocab_size"]:
        print(f"{key}={config.get(key)}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    messages = [{"role": "user", "content": "请用工具计算 123 + 456。"}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=TOOLS,
        open_thinking=True,
    )
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    has_open_think_prompt = prompt.endswith("<think>\n") or "<|im_start|>assistant\n<think>\n" in prompt

    print()
    print("[Chat template surface]")
    print(f"contains_tools_tag={'<tools>' in prompt}")
    print(f"contains_tool_call_instruction={'<tool_call>' in prompt}")
    print(f"has_open_think_prompt={has_open_think_prompt}")
    print(f"prompt_chars={len(prompt)}")
    print(f"prompt_tokens={len(ids)}")

    sample_model_text = (
        "<think>\n需要调用计算工具。\n</think>\n\n"
        '<tool_call>\n{"name": "calculate_math", "arguments": {"expression": "123 + 456"}}\n</tool_call>'
    )
    content, reasoning_content, tool_calls = parse_response(sample_model_text)

    print()
    print("[OpenAI response surface]")
    print(f"content={content!r}")
    print(f"reasoning_content={reasoning_content!r}")
    print(f"tool_calls={json.dumps(tool_calls, ensure_ascii=False)}")

    request_shape = {
        "model": "minimind",
        "messages": messages,
        "tools": TOOLS,
        "stream": True,
        "chat_template_kwargs": {"open_thinking": True},
    }
    print()
    print("[OpenAI request surface]")
    print(json.dumps(request_shape, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
