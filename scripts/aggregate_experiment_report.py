import argparse
import csv
import json
import os
from collections import OrderedDict


TRAIN_FINAL_KEYS = [
    "loss",
    "logits_loss",
    "aux_loss",
    "reward",
    "f1",
    "em",
    "cite",
    "format",
    "kl_ref",
    "avg_len",
    "lr",
]

EVAL_KEYS = [
    "answer_f1",
    "exact_match",
    "citation_precision",
    "citation_recall",
    "citation_valid",
    "format_score",
    "reward_total",
]


def fmt(value, digits=4):
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_run(runs, name):
    if name not in runs:
        runs[name] = OrderedDict(
            [
                ("experiment_name", name),
                ("stage", ""),
                ("backend", ""),
                ("model_profile", ""),
                ("attention_type", ""),
                ("hidden_size", ""),
                ("num_layers", ""),
                ("world_size", ""),
                ("per_gpu_batch", ""),
                ("accumulation_steps", ""),
                ("effective_batch", ""),
                ("max_seq_len", ""),
                ("tokens_per_optimizer_step", ""),
                ("peak_memory_gb", ""),
                ("last_step", ""),
            ]
        )
    return runs[name]


def merge_train_summary(runs, obj):
    name = obj.get("experiment_name") or os.path.splitext(os.path.basename(obj.get("source_log", "run")))[0]
    row = ensure_run(runs, name)
    setup = obj.get("setup", {})
    summary = obj.get("summary", {})
    final = summary.get("final", {})
    best = summary.get("best", {})

    row["stage"] = setup.get("stage", row.get("stage", ""))
    row["backend"] = setup.get("distributed", row.get("backend", ""))
    row["model_profile"] = setup.get("model_profile", row.get("model_profile", ""))
    row["attention_type"] = setup.get("attention_type", row.get("attention_type", ""))
    row["hidden_size"] = setup.get("hidden_size", row.get("hidden_size", ""))
    row["num_layers"] = setup.get("num_layers", row.get("num_layers", ""))
    row["world_size"] = setup.get("world_size", row.get("world_size", ""))
    row["per_gpu_batch"] = setup.get("per_gpu", row.get("per_gpu_batch", ""))
    row["accumulation_steps"] = setup.get("accumulation_steps", row.get("accumulation_steps", ""))
    row["effective_batch"] = setup.get("effective_batch", row.get("effective_batch", ""))
    row["max_seq_len"] = setup.get("max_seq_len", row.get("max_seq_len", ""))
    row["tokens_per_optimizer_step"] = setup.get("tokens_per_optimizer_step", row.get("tokens_per_optimizer_step", ""))
    row["peak_memory_gb"] = obj.get("peak_memory_gb") or row.get("peak_memory_gb", "")
    row["last_step"] = summary.get("last_step", row.get("last_step", ""))

    for key in TRAIN_FINAL_KEYS:
        if key in final:
            row[f"train_final_{key}"] = final[key]
        if key in best:
            row[f"train_best_{key}"] = best[key]


def merge_eval_summary(runs, obj):
    name = obj.get("experiment_name") or os.path.splitext(os.path.basename(obj.get("pred", "eval")))[0]
    row = ensure_run(runs, name)
    metrics = obj.get("metrics", obj)
    row["eval_num_samples"] = obj.get("num_samples", "")
    row["eval_num_bad_cases"] = obj.get("num_bad_cases", "")
    for key in EVAL_KEYS:
        if key in metrics:
            row[f"eval_{key}"] = metrics[key]


def merge_input(runs, path):
    obj = load_json(path)
    kind = obj.get("kind", "")
    if kind == "train_log_summary":
        merge_train_summary(runs, obj)
    elif kind == "search_shortqa_eval":
        merge_eval_summary(runs, obj)
    elif "summary" in obj and "setup" in obj:
        merge_train_summary(runs, obj)
    else:
        merge_eval_summary(runs, obj)


def metric_delta(row, baseline, key):
    value = row.get(key)
    base = baseline.get(key) if baseline else None
    if not isinstance(value, (int, float)) or not isinstance(base, (int, float)):
        return ""
    return value - base


def collect_fieldnames(rows):
    preferred = [
        "experiment_name",
        "stage",
        "backend",
        "model_profile",
        "attention_type",
        "hidden_size",
        "num_layers",
        "world_size",
        "per_gpu_batch",
        "accumulation_steps",
        "effective_batch",
        "max_seq_len",
        "tokens_per_optimizer_step",
        "peak_memory_gb",
        "last_step",
        "train_final_loss",
        "train_best_loss",
        "train_final_reward",
        "train_best_reward",
        "train_final_f1",
        "train_best_f1",
        "train_final_cite",
        "train_best_cite",
        "train_final_format",
        "train_final_kl_ref",
        "eval_answer_f1",
        "eval_exact_match",
        "eval_citation_valid",
        "eval_format_score",
        "eval_reward_total",
    ]
    all_keys = {key for row in rows for key in row.keys()}
    return [key for key in preferred if key in all_keys] + sorted(all_keys - set(preferred))


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows, fieldnames):
    header = "| " + " | ".join(fieldnames) + " |"
    sep = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for key in fieldnames) + " |")
    return "\n".join(lines)


def write_markdown(path, title, rows, fieldnames, baseline_name):
    baseline = next((row for row in rows if row.get("experiment_name") == baseline_name), None)
    report_fields = [key for key in fieldnames if key in {
        "experiment_name",
        "stage",
        "backend",
        "model_profile",
        "attention_type",
        "hidden_size",
        "world_size",
        "effective_batch",
        "peak_memory_gb",
        "train_final_loss",
        "train_final_reward",
        "train_final_f1",
        "train_final_cite",
        "eval_answer_f1",
        "eval_exact_match",
        "eval_citation_valid",
        "eval_format_score",
        "eval_reward_total",
    }]

    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        markdown_table(rows, report_fields),
        "",
    ]

    if baseline:
        lines.extend(["## Deltas Vs Baseline", ""])
        delta_fields = ["experiment_name"]
        for key in ("eval_answer_f1", "eval_citation_valid", "eval_format_score", "eval_reward_total", "train_final_reward", "peak_memory_gb"):
            if any(key in row for row in rows):
                delta_fields.append(f"delta_{key}")
        delta_rows = []
        for row in rows:
            delta_row = {"experiment_name": row.get("experiment_name", "")}
            for field in delta_fields[1:]:
                key = field.replace("delta_", "", 1)
                delta_row[field] = metric_delta(row, baseline, key)
            delta_rows.append(delta_row)
        lines.extend([markdown_table(delta_rows, delta_fields), ""])

    lines.extend(
        [
            "## Resume Metric Candidates",
            "",
            "- DeepSpeed/显存：填写 `peak_memory_gb`，对比 DDP baseline 的 `delta_peak_memory_gb`，可写峰值显存下降比例。",
            "- GRPO/效果：填写 `eval_answer_f1`、`eval_citation_valid`、`eval_format_score`，对比 SFT baseline 的提升。",
            "- MLA/效率：结合 `scripts/estimate_searchlm_scale.py` 和推理 benchmark，写 KV Cache 压缩比例、吞吐或延迟变化。",
            "",
        ]
    )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Aggregate training/evaluation JSON summaries into a resume-ready report")
    parser.add_argument("--inputs", nargs="+", required=True, help="JSON files from parse_train_log.py or eval_search_shortqa.py")
    parser.add_argument("--output_md", default="reports/search_shortqa_experiment_report.md")
    parser.add_argument("--output_csv", default="reports/search_shortqa_experiment_report.csv")
    parser.add_argument("--title", default="SearchShortQA Experiment Report")
    parser.add_argument("--baseline", default="", help="Experiment name used for delta calculation")
    args = parser.parse_args()

    runs = OrderedDict()
    for path in args.inputs:
        merge_input(runs, path)
    rows = list(runs.values())
    fieldnames = collect_fieldnames(rows)

    if args.output_csv:
        write_csv(args.output_csv, rows, fieldnames)
    if args.output_md:
        write_markdown(args.output_md, args.title, rows, fieldnames, args.baseline)

    print(f"Aggregated {len(rows)} experiments")
    if args.output_md:
        print(f"Markdown report: {args.output_md}")
    if args.output_csv:
        print(f"CSV table: {args.output_csv}")


if __name__ == "__main__":
    main()
