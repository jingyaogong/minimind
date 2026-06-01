import argparse
import json
from collections import Counter


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                yield i, json.loads(line)


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def validate_row(row):
    errors = []
    if not str(row.get("id", "")).strip():
        errors.append("missing_id")
    if not str(row.get("question", "")).strip():
        errors.append("missing_question")
    answers = row.get("answers") or row.get("answer")
    if not as_list(answers):
        errors.append("missing_answer")
    contexts = row.get("contexts") or []
    if not contexts:
        errors.append("missing_contexts")
    doc_ids = {str(doc.get("id", "")) for doc in contexts if isinstance(doc, dict)}
    gold_ids = {str(x) for x in as_list(row.get("gold_doc_ids"))}
    if not gold_ids:
        errors.append("missing_gold_doc_ids")
    elif not gold_ids.issubset(doc_ids):
        errors.append("gold_doc_not_in_contexts")
    empty_contexts = sum(1 for doc in contexts if not str(doc.get("text", "")).strip())
    if empty_contexts:
        errors.append("empty_context_text")
    return errors, len(contexts), len(doc_ids), len(gold_ids)


def main():
    parser = argparse.ArgumentParser(description="Validate SearchShortQA JSONL")
    parser.add_argument("--data", required=True)
    parser.add_argument("--max_errors", type=int, default=20)
    args = parser.parse_args()

    error_counts = Counter()
    context_counts = Counter()
    n = 0
    examples = []
    for lineno, row in iter_jsonl(args.data):
        n += 1
        errors, n_ctx, _, _ = validate_row(row)
        context_counts[n_ctx] += 1
        for err in errors:
            error_counts[err] += 1
        if errors and len(examples) < args.max_errors:
            examples.append({"line": lineno, "id": row.get("id"), "errors": errors})

    print(f"rows: {n}")
    print("context_count_distribution:", dict(sorted(context_counts.items())))
    print("error_counts:", dict(error_counts))
    if examples:
        print("error_examples:")
        for ex in examples:
            print(json.dumps(ex, ensure_ascii=False))
    if error_counts:
        raise SystemExit(1)
    print("SearchShortQA validation passed.")


if __name__ == "__main__":
    main()
