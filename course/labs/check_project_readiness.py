import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def status(label, state, detail=""):
    suffix = f" - {detail}" if detail else ""
    print(f"{state:16} {label}{suffix}")


def has_module(name):
    return importlib.util.find_spec(name) is not None


def check_import(name, optional=False):
    if has_module(name):
        module = __import__(name)
        version = getattr(module, "__version__", "unknown")
        status(name, "OK", f"version={version}")
    else:
        state = "OPTIONAL-MISSING" if optional else "MISSING"
        status(name, state)


def check_path(label, path, optional=False):
    path = ROOT / path
    if path.exists():
        if path.is_file():
            detail = f"{path.relative_to(ROOT)} ({path.stat().st_size} bytes)"
        else:
            detail = str(path.relative_to(ROOT))
        status(label, "OK", detail)
    else:
        state = "OPTIONAL-MISSING" if optional else "MISSING"
        status(label, state, str(path.relative_to(ROOT)))


def main():
    print(f"project_root={ROOT}")
    print(f"python={sys.version.split()[0]}")
    print()

    print("[Core packages]")
    for name in ["torch", "transformers", "datasets"]:
        check_import(name)

    print()
    print("[Optional packages]")
    for name in ["fastapi", "uvicorn", "openai", "streamlit", "swanlab"]:
        check_import(name, optional=True)

    print()
    print("[Torch backend]")
    if has_module("torch"):
        import torch

        status("cuda", "OK" if torch.cuda.is_available() else "MISSING", f"available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            status("cuda_device_count", "OK", str(torch.cuda.device_count()))
    else:
        status("cuda", "MISSING", "torch is not importable")

    print()
    print("[Built-in tokenizer/source files]")
    check_path("tokenizer_json", "model/tokenizer.json")
    check_path("tokenizer_config", "model/tokenizer_config.json")
    check_path("model_source", "model/model_minimind.py")
    check_path("eval_script", "eval_llm.py")

    print()
    print("[Default training data]")
    check_path("pretrain_t2t_mini", "dataset/pretrain_t2t_mini.jsonl")
    check_path("sft_t2t_mini", "dataset/sft_t2t_mini.jsonl")
    check_path("rlaif", "dataset/rlaif.jsonl", optional=True)
    check_path("agent_rl", "dataset/agent_rl.jsonl", optional=True)

    print()
    print("[Default torch weights]")
    check_path("pretrain_768", "out/pretrain_768.pth")
    check_path("full_sft_768", "out/full_sft_768.pth")

    print()
    print("[Transformers-format model]")
    check_path("minimind_3_dir", "minimind-3", optional=True)
    check_path("minimind_3_config", "minimind-3/config.json", optional=True)
    check_path("minimind_3_tokenizer", "minimind-3/tokenizer.json", optional=True)
    check_path("minimind_3_safetensors", "minimind-3/model.safetensors", optional=True)
    check_path("minimind_3_bin", "minimind-3/pytorch_model.bin", optional=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
