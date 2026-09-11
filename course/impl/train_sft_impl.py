"""Stage script skeleton: hand-written SFT implementation.

Lesson 17 connects the pretrain stage to the SFT stage. The full training
logic is assembled after the model, dataset, train-loop, and checkpoint
helpers are implemented.
"""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Course MiniMind SFT implementation")
    parser.add_argument("--data_path", default="course/labs/tiny_sft.jsonl")
    parser.add_argument("--tokenizer_path", default="model")
    parser.add_argument("--from_weight", default="course_pretrain")
    parser.add_argument("--save_weight", default="course_sft")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--from_resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args(argv)


def main():
    raise NotImplementedError("Assemble after the SFT stage lessons.")


if __name__ == "__main__":
    main()
