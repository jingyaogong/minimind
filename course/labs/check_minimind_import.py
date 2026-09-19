import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch
import transformers
import datasets

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def main():
    config = MiniMindConfig(
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=6400,
    )
    model = MiniMindForCausalLM(config)
    params = sum(p.numel() for p in model.parameters())

    print(f"torch={torch.__version__}")
    print(f"transformers={transformers.__version__}")
    print(f"datasets={datasets.__version__}")
    print(f"tiny_minimind_params={params}")


if __name__ == "__main__":
    main()
