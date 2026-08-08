"""Stage script skeleton: hand-written Pretrain implementation.

This file will be assembled after the model, dataset, loss, and train-loop
lessons are complete.
"""

from __future__ import annotations

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Course MiniMind pretrain implementation")
    parser.add_argument("--data_path", default="dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--tokenizer_path", default="model")
    parser.add_argument("--save_path", default="out/course_pretrain.pth")
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--max_steps", type=int, default=20)
    return parser.parse_args()


def main():
    raise NotImplementedError("Assemble after the Pretrain stage lessons.")


if __name__ == "__main__":
    main()
