import argparse
import csv
import json
import os
import re
from statistics import mean


EPOCH_RE = re.compile(r"Epoch:\[(?P<epoch>\d+)/(?P<epochs>\d+)\]\((?P<step>\d+)/(?P<iters>\d+)\)")
KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_ /-]*):\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.IGNORECASE)
SETUP_KEY_RE = re.compile(r"^\s{2}([^:]+):\s*(.*)$")
INLINE_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*|max_seq_len|tokens_per_optimizer_step~)=([^,\s]+)")
MEMORY_RE = re.compile(
    r"(?:peak|max).*?(?:memory|mem).*?([-+]?\d+(?:\.\d+)?)\s*(GB|GiB|MB|MiB)",
    re.IGNORECASE,
)


def to_number(value):
    value = str(value).strip().replace(",", "")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(ch in value.lower() for ch in (".", "e")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def normalize_key(key):
    return key.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def parse_setup_line(line, setup):
    if not line.startswith("  "):
        return
    stripped = line.strip()
    if not stripped or stripped.startswith("="):
        return

    match = SETUP_KEY_RE.match(line)
    value = stripped
    if match and "=" not in match.group(1):
        name = normalize_key(match.group(1))
        value = match.group(2).strip()
        setup[name] = value

    for segment in re.split(r",\s+", value):
        segment = segment.strip()
        if "=" in segment:
            key, raw_value = segment.split("=", 1)
            setup[normalize_key(key)] = to_number(raw_value)
        elif "~" in segment:
            key, raw_value = segment.split("~", 1)
            setup[normalize_key(key)] = to_number(raw_value)


def parse_metric_line(line):
    epoch_match = EPOCH_RE.search(line)
    if not epoch_match:
        return None

    record = {k: int(v) for k, v in epoch_match.groupdict().items()}
    for key, value in KEY_VALUE_RE.findall(line):
        normalized = normalize_key(key)
        if normalized.startswith("epoch"):
            continue
        record[normalized] = to_number(value)
    return record


def parse_memory_line(line):
    match = MEMORY_RE.search(line)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"mb", "mib"}:
        value /= 1024
    return value


def summarize_records(records):
    summary = {
        "num_records": len(records),
        "last_step": None,
        "final": {},
        "best": {},
        "mean": {},
    }
    if not records:
        return summary

    last = records[-1]
    summary["last_step"] = last.get("step")
    numeric_keys = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if isinstance(value, (int, float)) and key not in {"epoch", "epochs", "step", "iters"}
        }
    )
    for key in numeric_keys:
        values = [record[key] for record in records if isinstance(record.get(key), (int, float))]
        if not values:
            continue
        summary["final"][key] = values[-1]
        summary["mean"][key] = mean(values)
        if key in {"loss", "logits_loss", "aux_loss", "kl_ref", "epoch_time"}:
            summary["best"][key] = min(values)
        else:
            summary["best"][key] = max(values)
    return summary


def parse_log(path, experiment_name=None):
    setup = {}
    records = []
    peak_memory_gb = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[Training Setup]" in line:
                setup["stage"] = line.strip().split("stage=", 1)[-1] if "stage=" in line else line.strip()
                continue
            parse_setup_line(line, setup)
            record = parse_metric_line(line)
            if record:
                records.append(record)
            memory_gb = parse_memory_line(line)
            if memory_gb is not None:
                peak_memory_gb = memory_gb if peak_memory_gb is None else max(peak_memory_gb, memory_gb)

    name = experiment_name or os.path.splitext(os.path.basename(path))[0]
    summary = summarize_records(records)
    return {
        "kind": "train_log_summary",
        "experiment_name": name,
        "source_log": path,
        "setup": setup,
        "summary": summary,
        "peak_memory_gb": peak_memory_gb,
        "records": records,
    }


def write_csv(path, parsed):
    records = parsed["records"]
    if not records:
        fieldnames = ["experiment_name"]
    else:
        fieldnames = ["experiment_name"] + sorted({key for record in records for key in record})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {"experiment_name": parsed["experiment_name"]}
            row.update(record)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Parse SearchLM/SearchShortQA training logs into structured metrics")
    parser.add_argument("--log", required=True, help="Training log file")
    parser.add_argument("--experiment", default="", help="Experiment name used in JSON/CSV outputs")
    parser.add_argument("--output", default="", help="Output summary JSON path")
    parser.add_argument("--csv", default="", help="Optional per-log-record CSV path")
    parser.add_argument("--drop_records", action="store_true", help="Do not store full metric timeline in JSON")
    args = parser.parse_args()

    parsed = parse_log(args.log, experiment_name=args.experiment or None)
    output_obj = dict(parsed)
    if args.drop_records:
        output_obj.pop("records", None)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_obj, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(output_obj, ensure_ascii=False, indent=2))

    if args.csv:
        write_csv(args.csv, parsed)


if __name__ == "__main__":
    main()
