import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.search_shortqa_dataset import build_sft_conversation


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Build SFT data for SearchShortQA")
    parser.add_argument("--input", required=True, help="Raw JSONL with question/answer/contexts fields")
    parser.add_argument("--output", required=True, help="Output SFT JSONL path")
    parser.add_argument("--max_contexts", type=int, default=6)
    parser.add_argument("--max_chars_per_context", type=int, default=700)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    count = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for sample in iter_jsonl(args.input):
            if args.limit and count >= args.limit:
                break
            conversations = build_sft_conversation(
                sample,
                max_contexts=args.max_contexts,
                max_chars_per_context=args.max_chars_per_context,
            )
            out.write(json.dumps({"conversations": conversations}, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} SFT samples to {args.output}")


if __name__ == "__main__":
    main()
