import os
import json
import argparse
import random


def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def reservoir_sample(path, k, seed=42):
    random.seed(seed)
    reservoir = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < k:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = line
    return reservoir


def main():
    parser = argparse.ArgumentParser(description="生成固定验证集（JSONL）")
    parser.add_argument("--data_path", required=True, type=str, help="原始 jsonl 数据路径")
    parser.add_argument("--out_path", default="eval/val_pretrain.jsonl", type=str, help="输出验证集路径")
    parser.add_argument("--val_size", default=2000, type=int, help="验证集样本数（优先）")
    parser.add_argument("--val_ratio", default=0.0, type=float, help="验证集比例（若 val_size=0 则使用）")
    parser.add_argument("--seed", default=42, type=int, help="随机种子")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"data_path not found: {args.data_path}")

    if args.val_size <= 0:
        if args.val_ratio <= 0:
            raise ValueError("val_size <=0 时必须提供 val_ratio > 0")
        total = count_lines(args.data_path)
        args.val_size = max(1, int(total * args.val_ratio))

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    samples = reservoir_sample(args.data_path, args.val_size, seed=args.seed)

    with open(args.out_path, "w", encoding="utf-8") as f:
        for line in samples:
            line = line.strip()
            if not line:
                continue
            # 简单校验 JSONL 格式
            try:
                _ = json.loads(line)
            except Exception:
                continue
            f.write(line + "\n")

    print(f"[OK] Saved {len(samples)} lines to {args.out_path}")


if __name__ == "__main__":
    main()
