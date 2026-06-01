import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.search_shortqa_dataset import normalize_answers, normalize_contexts, normalize_gold_doc_ids
from trainer.search_shortqa_reward import average_metrics, score_search_shortqa_response


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prediction_text(row):
    for key in ("prediction", "response", "output", "answer"):
        if key in row:
            return str(row[key])
    return ""


def main():
    parser = argparse.ArgumentParser(description="Evaluate SearchShortQA predictions")
    parser.add_argument("--data", required=True, help="Gold JSONL with question/answer/contexts")
    parser.add_argument("--pred", required=True, help="Prediction JSONL aligned with data or keyed by id/question")
    parser.add_argument("--by", choices=["order", "id", "question"], default="order")
    parser.add_argument("--show_bad", type=int, default=5)
    parser.add_argument("--experiment", default="", help="Experiment name for JSON output")
    parser.add_argument("--json_output", default="", help="Optional path to save metric summary JSON")
    parser.add_argument("--bad_cases_output", default="", help="Optional path to save bad cases JSONL")
    args = parser.parse_args()

    data = load_jsonl(args.data)
    pred = load_jsonl(args.pred)
    if args.by == "order":
        pairs = list(zip(data, pred))
    else:
        pred_map = {str(row.get(args.by, "")): row for row in pred}
        pairs = [(row, pred_map.get(str(row.get(args.by, "")), {})) for row in data]

    metrics = []
    bad_cases = []
    for gold, pred_row in pairs:
        response = prediction_text(pred_row)
        score, parts = score_search_shortqa_response(
            response=response,
            answers=normalize_answers(gold),
            gold_doc_ids=normalize_gold_doc_ids(gold),
            contexts=normalize_contexts(gold),
        )
        parts["reward_total"] = score
        metrics.append(parts)
        if parts["answer_f1"] < 0.5 or parts["citation_valid"] < 1.0 or parts["format_score"] < 1.0:
            bad_cases.append((gold, response, parts))

    summary = average_metrics(metrics)
    print("SearchShortQA metrics")
    print("=" * 80)
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

    if args.json_output:
        result = {
            "kind": "search_shortqa_eval",
            "experiment_name": args.experiment or os.path.splitext(os.path.basename(args.pred))[0],
            "data": args.data,
            "pred": args.pred,
            "match_by": args.by,
            "num_samples": len(pairs),
            "num_bad_cases": len(bad_cases),
            "metrics": summary,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_output)), exist_ok=True)
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    if args.bad_cases_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.bad_cases_output)), exist_ok=True)
        with open(args.bad_cases_output, "w", encoding="utf-8") as f:
            for gold, response, parts in bad_cases:
                row = {
                    "id": gold.get("id"),
                    "question": gold.get("question") or gold.get("query") or "",
                    "gold_answers": normalize_answers(gold),
                    "gold_doc_ids": normalize_gold_doc_ids(gold),
                    "response": response,
                    "metrics": parts,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.show_bad > 0 and bad_cases:
        print("\nBad cases")
        print("=" * 80)
        for gold, response, parts in bad_cases[: args.show_bad]:
            question = gold.get("question") or gold.get("query") or ""
            print(f"Q: {question}")
            print(f"Gold: {normalize_answers(gold)} cites={normalize_gold_doc_ids(gold)}")
            print(f"Pred: {response}")
            print("Metrics:", json.dumps(parts, ensure_ascii=False))
            print("-" * 80)


if __name__ == "__main__":
    main()
