#!/usr/bin/env python
"""Held-out evaluation harness for MiniMind math RLVR checkpoints.

Reports greedy accuracy and sampled pass@k separately (they are not the same
thing), together with format / no-answer rates, completion-length statistics
and per-difficulty breakdowns.  The verifier is the same ``trainer.math_verifier``
module used by the training reward, so a checkpoint cannot be scored by a
different answer grammar than the one it was trained against.

Fixed from the repository root, e.g.

    python scripts/eval_heldout.py \
        --model_path out/mini_rule_seed42_768.pth \
        --data_path dataset/math_heldout.jsonl \
        --output logs/mini_rule_seed42_eval.json \
        --num_samples 16 --max_new_tokens 512 --limit 100 \
        --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from trainer.math_verifier import ANSWER_MARKER, verdict  # noqa: E402
from trainer.trainer_utils import setup_seed  # noqa: E402

VERIFIER_PATH = REPO_ROOT / "trainer" / "math_verifier.py"


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased pass@k estimator: ``1 - C(n-c, k) / C(n, k)``.

    ``n`` sampled completions, ``c`` of them correct.  Returns ``None`` when the
    sample budget is smaller than ``k`` rather than silently reporting a number
    computed from too few samples.
    """

    if k > n:
        return None
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _median(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def summarize_group(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    """Aggregate one set of per-row records (all rows, or one difficulty)."""

    sampled = [completion for row in rows for completion in row["sampled"]]
    greedy = [row["greedy"] for row in rows]
    n_samples = len(sampled)
    sample_budget = len(rows[0]["sampled"]) if rows else 0

    summary: dict[str, Any] = {
        "rows": len(rows),
        "sample_budget": sample_budget,
        "greedy_accuracy": _rate(sum(1 for item in greedy if item["correct"]), len(greedy)),
        "sampled_pass_at_1": _rate(sum(row["sampled_n_correct"] for row in rows), sum(len(row["sampled"]) for row in rows)),
        "format_rate": _rate(sum(1 for item in sampled if item["format_ok"]), n_samples),
        "no_answer_rate": _rate(sum(1 for item in sampled if item["status"] == "no_answer"), n_samples),
        "mean_completion_tokens": (sum(item["tokens"] for item in sampled) / n_samples) if n_samples else 0.0,
        "median_completion_tokens": _median([item["tokens"] for item in sampled]),
        "greedy_format_rate": _rate(sum(1 for item in greedy if item["format_ok"]), len(greedy)),
        "greedy_no_answer_rate": _rate(sum(1 for item in greedy if item["status"] == "no_answer"), len(greedy)),
        "greedy_mean_completion_tokens": (sum(item["tokens"] for item in greedy) / len(greedy)) if greedy else 0.0,
        "status_counts": {
            status: sum(1 for item in sampled if item["status"] == status)
            for status in ("correct", "incorrect", "no_answer", "invalid_gt")
        },
        "marker_hit_rate": _rate(sum(1 for item in sampled if item["marker_hit"]), n_samples),
    }
    for k in ks:
        estimates = [pass_at_k(len(row["sampled"]), row["sampled_n_correct"], k) for row in rows]
        values = [value for value in estimates if value is not None]
        summary[f"pass_at_{k}"] = (sum(values) / len(values)) if values else None
    return summary


def summarize(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    summary = summarize_group(rows, ks)
    by_difficulty = {}
    for difficulty in sorted({row["difficulty"] for row in rows}):
        by_difficulty[difficulty] = summarize_group(
            [row for row in rows if row["difficulty"] == difficulty], ks
        )
    summary["by_difficulty"] = by_difficulty
    return summary


# --------------------------------------------------------------------------- #
# model / data plumbing
# --------------------------------------------------------------------------- #
def load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(str(REPO_ROOT / "model"))
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model = MiniMindForCausalLM(config).to(args.device).eval()

    state = torch.load(args.model_path, map_location=args.device)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    # strict=True: a structure mismatch must fail here, not quietly score a
    # partially random model.
    model.load_state_dict(state, strict=True)
    return model, tokenizer


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            conversations = row.get("conversations") or []
            if len(conversations) != 1 or conversations[0].get("role") != "user":
                raise ValueError(f"{path}:{lineno}: expected exactly one user message")
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def build_prompt(tokenizer, row: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        row["conversations"],
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=True,
    )


def completion_length(generated: torch.Tensor, eos_token_id: int) -> int:
    """Token count up to and including the first EOS (same convention as training)."""

    ids = generated.tolist()
    if eos_token_id is not None and eos_token_id in ids:
        return ids.index(eos_token_id) + 1
    return len(ids)


def score_completion(text: str, gt: str, marker: str, tokens: int) -> dict[str, Any]:
    result = verdict(text, gt, marker=marker)
    return {
        "text": text,
        "status": result["status"],
        "extracted": result["extracted"],
        "correct": result["correct"],
        "think_closed": result["think_closed"],
        "marker_hit": result["marker_hit"],
        "format_ok": result["format_ok"],
        "tokens": tokens,
    }


@torch.no_grad()
def evaluate_row(model, tokenizer, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    prompt = build_prompt(tokenizer, row)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)

    greedy_ids = model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    greedy_part = greedy_ids[0, encoded["input_ids"].shape[1]:]
    greedy = score_completion(
        tokenizer.decode(greedy_part, skip_special_tokens=True),
        row["gt"],
        args.marker,
        completion_length(greedy_part, tokenizer.eos_token_id),
    )

    sampled_ids = model.generate(
        **encoded,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = encoded["input_ids"].shape[1]
    sampled = [
        score_completion(
            tokenizer.decode(sampled_ids[i, prompt_len:], skip_special_tokens=True),
            row["gt"],
            args.marker,
            completion_length(sampled_ids[i, prompt_len:], tokenizer.eos_token_id),
        )
        for i in range(sampled_ids.shape[0])
    ]
    return {
        "sample_id": row.get("sample_id") or "",
        "difficulty": row.get("difficulty") or "unknown",
        "source": row.get("source") or "unknown",
        "gt": str(row["gt"]),
        "prompt": prompt,
        "greedy": greedy,
        "sampled": sampled,
        "sampled_n_correct": sum(1 for item in sampled if item["correct"]),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_provenance(data_path: Path, manifest_path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "data_path": str(data_path),
        "data_file_sha256": sha256_file(data_path),
        "verifier_sha256": sha256_file(VERIFIER_PATH),
    }
    if manifest_path.is_file():
        info["data_manifest_sha256"] = sha256_file(manifest_path)
        recorded = (json.loads(manifest_path.read_text(encoding="utf-8")).get("outputs") or {}).get(
            data_path.name
        )
        info["data_manifest_recorded_sha256"] = recorded
        info["data_matches_manifest"] = recorded == info["data_file_sha256"]
    else:
        info["data_manifest_sha256"] = None
        info["data_matches_manifest"] = None
    return info


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a MiniMind checkpoint on held-out math problems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=Path, required=True, help="MiniMind .pth checkpoint")
    parser.add_argument("--data_path", type=Path, default=Path("dataset/math_heldout.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("logs/eval_heldout.json"))
    parser.add_argument("--raw_output", type=Path, default=None,
                        help="JSONL with every sampled completion (default: <output>.jsonl)")
    parser.add_argument("--manifest", type=Path, default=Path("dataset/manifests/math_manifest.json"))
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N rows (0 = all)")
    parser.add_argument("--num_samples", type=int, default=16, help="sampled completions per problem")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--marker", type=str, default=ANSWER_MARKER)
    parser.add_argument("--hidden_size", type=int, default=768, help="must match training")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="must match training")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--pass_at", type=str, default="1,4,8,16",
                        help="comma separated k values; k > num_samples is reported as null")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_samples < 1:
        print("error: --num_samples must be >= 1", file=sys.stderr)
        return 2
    ks = sorted({int(value) for value in args.pass_at.split(",") if value.strip()})

    setup_seed(args.seed)
    model, tokenizer = load_model(args)

    rows = read_rows(args.data_path)
    if args.limit:
        rows = rows[: args.limit]
    provenance = data_provenance(args.data_path, args.manifest)

    records = []
    for index, row in enumerate(rows, 1):
        record = evaluate_row(model, tokenizer, row, args)
        records.append(record)
        if index % 10 == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] rows evaluated", flush=True)

    summary = summarize(records, ks)
    payload = {
        "model": str(args.model_path),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": args.device,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "marker": args.marker,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "use_moe": args.use_moe,
        "rows_evaluated": len(records),
        "rows_available": len(read_rows(args.data_path)),
        "metric_scope": "sampled (greedy_* keys are computed on the greedy pass)",
        **provenance,
    }
    for key in ("greedy_accuracy", "sampled_pass_at_1", "format_rate", "no_answer_rate",
                "mean_completion_tokens", "median_completion_tokens"):
        payload[key] = summary[key]
    for k in ks:
        payload[f"pass_at_{k}"] = summary[f"pass_at_{k}"]
    payload["by_difficulty"] = summary["by_difficulty"]
    payload["status_counts"] = summary["status_counts"]
    payload["samples"] = records

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    raw_path = args.raw_output or args.output.with_suffix(".jsonl")
    with raw_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({
                "sample_id": record["sample_id"], "difficulty": record["difficulty"],
                "gt": record["gt"], "mode": "greedy", **record["greedy"],
            }, ensure_ascii=False) + "\n")
            for index, item in enumerate(record["sampled"]):
                handle.write(json.dumps({
                    "sample_id": record["sample_id"], "difficulty": record["difficulty"],
                    "gt": record["gt"], "mode": "sampled", "index": index, **item,
                }, ensure_ascii=False) + "\n")

    print(
        f"rows={len(records)} greedy_acc={summary['greedy_accuracy']:.4f} "
        + " ".join(
            f"pass@{k}={'n/a' if summary[f'pass_at_{k}'] is None else format(summary[f'pass_at_{k}'], '.4f')}"
            for k in ks
        )
        + f" format={summary['format_rate']:.4f} no_answer={summary['no_answer_rate']:.4f}"
    )
    if provenance.get("data_matches_manifest") is False:
        recorded = provenance.get("data_manifest_recorded_sha256")
        if recorded is None:
            print(f"NOTE: {args.data_path} 不是 math_manifest.json 登记的评估集（例如难度探针），"
                  "此时 data_matches_manifest=false 属**预期**，不是数据被改。", file=sys.stderr)
        else:
            print(f"WARNING: {args.data_path} 的 sha256 与 manifest 记录不一致 "
                  f"(got {provenance['data_file_sha256'][:12]}…, want {recorded[:12]}…)；"
                  "这次评估的结论不能归档，先查数据是不是被改过"
                  "（scripts/clean_math_data.py --check-only）。", file=sys.stderr)
    print(f"wrote {args.output} and {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
