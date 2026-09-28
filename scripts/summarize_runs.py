#!/usr/bin/env python
"""Cross-run summary and figures for the RLVR audit (SOP sections 10.3, 10.4, 11).

One run tells you what happened; six runs tell you whether a knob did anything.
This script walks ``runs/<run_tag>/`` (the section 13 archives), joins them with
the pre-registered registry rows, and writes:

* ``reports/summary.csv``   one row per run, machine readable, missing = empty
* ``reports/summary.md``    the section 10.3 evaluation matrix, the seed
                            aggregate from 10.4, and the section 11.2 decision
                            tree applied per run
* ``reports/figures/*.png`` the six figures listed in section 11.1, each with a
                            caption naming model / dataset / seed / sample count
                            / temperature / length convention
* ``reports/manifest.json`` which runs and files went in, and what was missing

Two honesty rules, same as ``archive_run.py``:

* numbers are read, never recomputed from prose: training dynamics come from
  ``metrics.csv`` and held-out numbers from ``eval.json``; a run without an
  evaluation contributes empty cells rather than zeros;
* runs are labelled by evidence level (``archived`` vs ``logs-only``), and
  ``--require-eval`` turns any gap into a non-zero exit code so a half-finished
  table cannot be mistaken for a finished one.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.archive_run import check_archive, read_json, read_metrics  # noqa: E402
from scripts.experiment_registry import DEFAULT_REGISTRY, baseline_row, load_registry  # noqa: E402

# Runs that are not experiment arms but still belong in the section 10.3 matrix.
# ``math_cold_cot_seed42`` 是六个臂真正加载的起点（CoT v2）；``math_cold_seed42`` 是
# pilot 的模板 CoT 版本，只作为「CoT 不可学」的对照留在矩阵里。
EXTRA_TAGS = ("math_cold_cot_seed42", "math_cold_seed42")

# The section 10.3 evaluation matrix, in SOP order.
MATRIX_ROWS = [
    ("冷启动 SFT（CoT v2，RL 起点）", "math_cold_cot_seed42"),
    ("rule GRPO", "mini_rule_seed42"),
    ("dynamic sampling", "mini_rule_dynamic_seed42"),
    ("length penalty", "mini_rule_len_seed42"),
    ("RM-only", "mini_rm_seed42"),
    ("calibrated mix", "mini_mix_seed42"),
    ("冷启动 SFT（CoT v1，pilot 对照）", "math_cold_seed42"),
]

# Knobs that define an arm; two runs that agree on all of them differ only by seed.
ARM_KNOBS = ["model", "reward_mode", "G", "max_gen_len", "beta", "dynamic_sampling",
             "len_penalty_coef", "alpha", "beta_mix"]

CSV_COLUMNS = [
    "run_tag", "evidence", "archive_status", "model", "reward_mode", "seed",
    "dynamic_sampling", "len_penalty_coef", "alpha", "beta_mix", "hypothesis",
    "metrics_rows", "last_step",
    "reward_first", "reward_last", "reward_gain", "reward_max",
    "correct_rate_first", "correct_rate_last", "correct_rate_gain",
    "format_rate_first", "format_rate_last", "format_rate_gain",
    "completion_tokens_first", "completion_tokens_last",
    "kl_ref_last", "entropy_last", "policy_loss_last", "adv_std_last",
    "valid_groups_last", "all_correct_groups_last", "all_wrong_groups_last",
    "acr_correct_last", "acr_wrong_last", "filtered_fraction_last",
    "eval_greedy_accuracy", "eval_pass_at_1", "eval_pass_at_4", "eval_pass_at_8",
    "eval_pass_at_16", "eval_format_rate", "eval_no_answer_rate", "eval_marker_hit_rate",
    "eval_mean_tokens", "eval_median_tokens", "eval_num_samples", "eval_temperature",
    "eval_rows_evaluated", "eval_data_matches_manifest",
    "eval_easy_greedy", "eval_medium_greedy", "eval_hard_greedy",
    "eval_easy_pass1", "eval_medium_pass1", "eval_hard_pass1",
    "heldout_gain", "reward_eval_gap", "passk_inversion", "hacking_suspect",
    "decision_tree", "seed_group", "warnings",
]

MIN_ROWS_FOR_TREND = 2
EPS = 1e-9


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_mpl() -> Any:
    """Import pyplot with a writable cache dir, or return None.

    Figures are a nice-to-have; the CSV and the markdown must still be produced
    on a machine without matplotlib.
    """

    if "MPLCONFIGDIR" not in os.environ:
        cache = Path.home() / ".matplotlib"
        try:
            cache.mkdir(parents=True, exist_ok=True)
            writable = os.access(cache, os.W_OK)
        except OSError:
            writable = False
        if not writable:
            fallback = Path(tempfile.gettempdir()) / "matplotlib-rlvr"
            fallback.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(fallback)
    try:
        import matplotlib
    except Exception:
        return None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def numeric_rows(header: list[str], rows: list[dict[str, str]]) -> list[dict[str, float | None]]:
    """Metrics CSV as floats; blank or unparseable cells stay None."""

    out: list[dict[str, float | None]] = []
    for row in rows:
        converted: dict[str, float | None] = {}
        for column in header:
            try:
                converted[column] = float(row[column])
            except (KeyError, TypeError, ValueError):
                converted[column] = None
        out.append(converted)
    return out


def series(rows: list[dict[str, float | None]], column: str) -> list[float | None]:
    values = [row.get(column) for row in rows]
    return [value if value is not None else None for value in values]


#: 首/末端点各取多少个 log 点做均值。单点太抖：``mini_rule_seed42`` 的第 5 步
#: correct_rate 是 0.75、第 1000 步是 0.50，只看这两点会得出"训练在退步"的结论，
#: 而整段均值其实稳定在 0.32 附近——决策树会被这点噪声误导。
# ``mix`` 臂的奖励是两个分量的组内 z-score 加权，均值恒为 0：首末差没有信息量，
# 渲染和诊断都必须走单独的文案，否则会把"设计如此"误读成"reward 没动"。
STANDARDIZED_REWARD_MODES = {"mix"}
RM_REWARD_MODES = {"rm", "official"}
STANDARDIZED_REWARD_CELL = "n/a（mix 臂为组内标准分量，均值恒 0）"

TREND_WINDOW_MAX = 10


def trend_window(n: int) -> int:
    """首/末窗口的点数：长序列取 10，短序列退化成 1（保持小 fixture 的语义）。"""

    return max(1, min(TREND_WINDOW_MAX, n // 5))


def trend_endpoints(values: list[float | None]) -> tuple[float | None, float | None]:
    """返回 (首窗均值, 末窗均值)；窗口大小见 :func:`trend_window`。"""

    present = [value for value in values if value is not None]
    if not present:
        return None, None
    width = trend_window(len(present))
    head = present[:width]
    tail = present[-width:]
    return sum(head) / len(head), sum(tail) / len(tail)


# 旧名字保留为别名：外部脚本/笔记里可能还引用它。
first_last = trend_endpoints


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


# --------------------------------------------------------------------------- #
# per-run collection
# --------------------------------------------------------------------------- #
def collect_run(
    tag: str,
    *,
    runs_dir: Path,
    log_dir: Path,
    registry_row: dict[str, str] | None,
) -> dict[str, Any]:
    """Gather everything known about one run, without inventing anything."""

    run_dir = runs_dir / tag
    archived = (run_dir / "manifest.json").is_file()
    metrics_path = run_dir / "metrics.csv" if archived else log_dir / f"{tag}_metrics.csv"
    if not metrics_path.is_file():
        metrics_path = log_dir / f"{tag}_metrics.csv"
    eval_path = run_dir / "eval.json" if (run_dir / "eval.json").is_file() else log_dir / f"{tag}_eval.json"

    header: list[str] = []
    rows: list[dict[str, str]] = []
    if metrics_path.is_file():
        header, rows = read_metrics(metrics_path)
    numeric = numeric_rows(header, rows)

    eval_payload: dict[str, Any] | None = None
    if eval_path.is_file():
        try:
            eval_payload = read_json(eval_path)
        except json.JSONDecodeError:
            eval_payload = None

    manifest = read_json(run_dir / "manifest.json") if archived else None
    warnings: list[str] = []
    if archived:
        warnings.extend(check_archive(run_dir))
    else:
        warnings.append(f"not archived yet: run `python scripts/archive_run.py --run-tag {tag}`")
    if eval_payload is None:
        warnings.append("no held-out evaluation")
    if len(numeric) < MIN_ROWS_FOR_TREND:
        warnings.append(f"only {len(numeric)} logged step(s); no trend can be read")

    return {
        "run_tag": tag,
        "evidence": "archived" if archived else ("logs-only" if header else "missing"),
        "archive_status": (manifest or {}).get("status", "n/a" if not archived else "unknown"),
        "registry_row": registry_row,
        "metrics_path": metrics_path if header else None,
        "eval_path": eval_path if eval_payload is not None else None,
        "header": header,
        "rows": numeric,
        "eval": eval_payload,
        "warnings": warnings,
    }


def arm_signature(row: dict[str, Any]) -> tuple[str, ...]:
    registry = row.get("registry_row") or {}
    return tuple(str(registry.get(knob, "")) for knob in ARM_KNOBS)


def seed_group_name(signature: tuple[str, ...]) -> str:
    return "|".join(signature)


def decision_tree(row: dict[str, Any]) -> str:
    """Apply the SOP section 11.2 decision tree to one run's numbers.

    The tree in the SOP reads "reward up? -> correctness up? -> check held-out
    and seeds".  Training-side correctness and held-out accuracy are therefore
    reported together: a run whose training correctness rises while held-out
    stays flat is a reward-eval gap, not a capability claim, and the wording
    says so instead of letting the first branch flatter the run.

    ``heldout_gain`` 只是**点估计的方向**：本函数不做配对检验，因此措辞一律不写"显著"，
    并要求读者去 ``paired_compare.py`` 的 McNemar p 值与本项目实测的噪声地板
    （``reports/report.md`` §4.7）复核。示例：``mini_rule_dynamic_seed42`` 的 −0.005
    恰好落在噪声地板内，不能当负结果引用。
    """

    reward_gain = row.get("reward_gain")
    mode = (row.get("reward_mode") or "").strip()
    if row.get("metrics_rows", 0) < MIN_ROWS_FOR_TREND or reward_gain is None:
        return "数据不足：需要至少两个 log_interval 的记录才能判断"
    if reward_gain <= EPS and mode not in STANDARDIZED_REWARD_MODES:
        # mix 臂的 reward 是组内标准分量，首末差恒为 0，不能当成"reward 没动"来读。
        return "否：查 reward 范围、verifier、学习率、rollout、NaN"

    train_correct_up = (row.get("correct_rate_gain") or 0) > EPS
    heldout = row.get("heldout_gain")
    if heldout is not None and heldout > EPS:
        return "是：训练与 held-out 同向上升 → 真实任务信号候选（第二 seed 复核）"
    if heldout is not None and heldout < -EPS:
        # 负结果也要按奖励源分类，否则会把 RM hack 误诊成"长度偏置"。
        if mode in RM_REWARD_MODES:
            return ("否：held-out 下降 —— RM 奖励 hack：奖励与任务正确性背离"
                    "（先拆 reward 组成 + 看 ACR；显著性见 paired_compare）")
        if mode in STANDARDIZED_REWARD_MODES:
            return ("否：held-out 下降 —— RM 分量污染了 rule 信号"
                    "（负结果：先看 component 退化组；显著性见 paired_compare）")
        return ("否：held-out 下降（负结果）：先对照 paired_compare 的 p 值与本项目实测的噪声地板"
                "（report §4.7）再定性，别只看点估计方向")
    if train_correct_up:
        if heldout is None:
            return "训练侧正确率上升，但还没有 held-out 数字：先跑 eval_heldout.py 再下结论"
        return "否：训练侧正确率上升但 held-out 未涨（reward-eval gap）：先查 held-out 难度/样本量/verifier 口径"
    if mode in STANDARDIZED_REWARD_MODES:
        return "否：mix 臂的 reward 是组内标准化分量（均值恒为 0），只能读 correct/held-out 两处"
    if (row.get("format_rate_gain") or 0) > EPS:
        return "否：格式吸收 reward"
    if (row.get("completion_tokens_last") or 0) > (row.get("completion_tokens_first") or 0):
        return "否：长度偏置 / 过长 hack"
    if (row.get("acr_wrong_last") or 0) >= 0.5:
        return "否：容量 / 难度 / 冷启动（ACR_wrong 高）"
    return "否：奖励分辨率或优化设置问题"


def summarize(eval_payload: dict[str, Any] | None, prefix: str) -> dict[str, float | None]:
    if not eval_payload:
        return {}
    keys = {
        f"{prefix}_greedy_accuracy": "greedy_accuracy",
        f"{prefix}_pass_at_1": "sampled_pass_at_1",
        f"{prefix}_pass_at_4": "pass_at_4",
        f"{prefix}_pass_at_8": "pass_at_8",
        f"{prefix}_pass_at_16": "pass_at_16",
        f"{prefix}_format_rate": "format_rate",
        f"{prefix}_no_answer_rate": "no_answer_rate",
        f"{prefix}_marker_hit_rate": "marker_hit_rate",
        f"{prefix}_mean_tokens": "mean_completion_tokens",
        f"{prefix}_median_tokens": "median_completion_tokens",
        f"{prefix}_num_samples": "num_samples",
        f"{prefix}_temperature": "temperature",
        f"{prefix}_rows_evaluated": "rows_evaluated",
        f"{prefix}_data_matches_manifest": "data_matches_manifest",
    }
    out: dict[str, float | None] = {}
    for target, source in keys.items():
        value = eval_payload.get(source)
        out[target] = value if isinstance(value, (int, float)) else (value if isinstance(value, bool) else None)
    return out


def build_row(run: dict[str, Any]) -> dict[str, Any]:
    registry = run.get("registry_row") or {}
    rows = run["rows"]
    reward_first, reward_last = trend_endpoints(series(rows, "avg_reward"))
    correct_first, correct_last = trend_endpoints(series(rows, "correct_rate"))
    format_first, format_last = trend_endpoints(series(rows, "format_rate"))
    tokens_first, tokens_last = trend_endpoints(series(rows, "avg_completion_tokens"))
    rewards = [value for value in series(rows, "avg_reward") if value is not None]

    last = rows[-1] if rows else {}
    valid = last.get("valid_groups")
    all_correct = last.get("all_correct_groups")
    all_wrong = last.get("all_wrong_groups")
    group_total = None
    if None not in (valid, all_correct, all_wrong):
        group_total = (valid or 0) + (all_correct or 0) + (all_wrong or 0)

    row: dict[str, Any] = {
        "run_tag": run["run_tag"],
        "evidence": run["evidence"],
        "archive_status": run["archive_status"],
        "model": registry.get("model", ""),
        "reward_mode": registry.get("reward_mode", ""),
        "seed": registry.get("seed", ""),
        "dynamic_sampling": registry.get("dynamic_sampling", ""),
        "len_penalty_coef": registry.get("len_penalty_coef", ""),
        "alpha": registry.get("alpha", ""),
        "beta_mix": registry.get("beta_mix", ""),
        "hypothesis": registry.get("hypothesis", ""),
        "metrics_rows": len(rows),
        "last_step": last.get("step"),
        "reward_first": reward_first,
        "reward_last": reward_last,
        "reward_gain": None if reward_first is None or reward_last is None else reward_last - reward_first,
        "reward_max": max(rewards) if rewards else None,
        "correct_rate_first": correct_first,
        "correct_rate_last": correct_last,
        "correct_rate_gain": None if correct_first is None or correct_last is None else correct_last - correct_first,
        "format_rate_first": format_first,
        "format_rate_last": format_last,
        "format_rate_gain": None if format_first is None or format_last is None else format_last - format_first,
        "completion_tokens_first": tokens_first,
        "completion_tokens_last": tokens_last,
        "kl_ref_last": last.get("kl_ref"),
        "entropy_last": last.get("entropy"),
        "policy_loss_last": last.get("policy_loss"),
        "adv_std_last": last.get("adv_std"),
        "valid_groups_last": valid,
        "all_correct_groups_last": all_correct,
        "all_wrong_groups_last": all_wrong,
        "acr_correct_last": _ratio(all_correct, group_total),
        "acr_wrong_last": _ratio(all_wrong, group_total),
        "filtered_fraction_last": last.get("filtered_fraction"),
    }
    row.update(summarize(run.get("eval"), "eval"))
    by_difficulty = (run.get("eval") or {}).get("by_difficulty") or {}
    for name in ("easy", "medium", "hard"):
        entry = by_difficulty.get(name)
        entry = entry if isinstance(entry, dict) else {}
        row[f"eval_{name}_greedy"] = entry.get("greedy_accuracy")
        row[f"eval_{name}_pass1"] = entry.get("sampled_pass_at_1")
    row["seed_group"] = seed_group_name(arm_signature(run))
    row["warnings"] = "; ".join(run["warnings"])
    return row


def add_cross_run_columns(rows: list[dict[str, Any]], baseline_tag: str | None) -> None:
    baseline = next((row for row in rows if row["run_tag"] == baseline_tag), None)
    base_greedy = (baseline or {}).get("eval_greedy_accuracy")
    base_pass16 = (baseline or {}).get("eval_pass_at_16")
    for row in rows:
        greedy = row.get("eval_greedy_accuracy")
        row["heldout_gain"] = None if greedy is None or base_greedy is None else greedy - base_greedy
        if row["reward_gain"] is not None and row["heldout_gain"] is not None:
            row["reward_eval_gap"] = row["reward_gain"] - row["heldout_gain"]
        else:
            row["reward_eval_gap"] = None
        pass16 = row.get("eval_pass_at_16")
        pass1 = row.get("eval_pass_at_1")
        base_pass1 = (baseline or {}).get("eval_pass_at_1")
        row["passk_inversion"] = bool(
            None not in (base_pass16, pass16, pass1, base_pass1)
            and (pass1 - base_pass1) > 0
            and (pass16 - base_pass16) < 0
        )
        row["hacking_suspect"] = bool(
            row["reward_gain"] is not None and row["reward_gain"] > EPS
            and (row.get("correct_rate_gain") or 0) <= EPS
            and row.get("heldout_gain") is not None and row["heldout_gain"] <= EPS
        )
        row["decision_tree"] = decision_tree(row)


def run_rows(
    *,
    runs_dir: Path,
    log_dir: Path,
    registry_path: Path,
    tags: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, dict[str, Any]]]:
    """Collect every included run once and return (table rows, notes, raw runs)."""

    _header, registry_rows = load_registry(registry_path)
    registry = {(row.get("run_tag") or "").strip(): row for row in registry_rows}
    if tags is None:
        tags = list(registry) + [tag for tag in EXTRA_TAGS if tag not in registry]
    collected = {
        tag: collect_run(tag, runs_dir=runs_dir, log_dir=log_dir, registry_row=registry.get(tag))
        for tag in tags
    }
    baseline = baseline_row(registry_rows) or (registry_rows[0] if registry_rows else None)
    baseline_tag = (baseline or {}).get("run_tag")
    rows = [build_row(collected[tag]) for tag in tags]
    add_cross_run_columns(rows, baseline_tag)
    notes = [f"reference arm: {baseline_tag}"] if baseline_tag else []
    return rows, notes, collected


# --------------------------------------------------------------------------- #
# rendering: csv
# --------------------------------------------------------------------------- #
def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_cell(row.get(column)) for column in CSV_COLUMNS})


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


# --------------------------------------------------------------------------- #
# rendering: markdown
# --------------------------------------------------------------------------- #
def fmt(value: Any, digits: int = 3, dash: str = "—") -> str:
    if value is None or value == "":
        return dash
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_delta(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{value:+.{digits}f}"
    return "—"


def reward_trend_cell(row: dict[str, Any]) -> str:
    """首/末窗口均值；标准化奖励（mix）渲染成 n/a 而不是 0.000。"""

    if (row.get("reward_mode") or "").strip() in STANDARDIZED_REWARD_MODES:
        return STANDARDIZED_REWARD_CELL
    return f"{fmt(row.get('reward_first'))} → {fmt(row.get('reward_last'))} ({fmt_delta(row.get('reward_gain'))})"


def reward_gain_cell(row: dict[str, Any]) -> str:
    if (row.get("reward_mode") or "").strip() in STANDARDIZED_REWARD_MODES:
        return "n/a（标准化）"
    return fmt_delta(row.get("reward_gain"))


def render_matrix(rows: list[dict[str, Any]]) -> list[str]:
    """The SOP section 10.3 evaluation matrix."""

    index = {row["run_tag"]: row for row in rows}
    lines = [
        "| 权重 | 题数 | greedy | pass@1 | pass@16 | format | avg len | easy | medium | hard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    # 矩阵里最常出现的题数当作基准；更小的 eval 一律打 ⚠️，避免有人把 20 题的数字
    # 和 400 题的数字并排引用（SOP §10.1 要求每次评估写清 rows_evaluated）。
    evaluated = [row.get("eval_rows_evaluated") for row in index.values()
                 if row.get("eval_rows_evaluated")]
    reference_rows = max(evaluated) if evaluated else None
    flagged = False
    for label, tag in MATRIX_ROWS:
        row = index.get(tag)
        if row is None:
            lines.append(f"| {label} (`{tag}`) | — | — | — | — | — | — | — | — | — |")
            continue
        marker = "" if row["evidence"] == "archived" else " *"
        rows_n = row.get("eval_rows_evaluated")
        if rows_n and reference_rows and rows_n < reference_rows:
            marker += " ⚠️"
            flagged = True
        lines.append(
            f"| {label} (`{tag}`){marker} | {fmt(rows_n)} | {fmt(row.get('eval_greedy_accuracy'))} | "
            f"{fmt(row.get('eval_pass_at_1'))} | {fmt(row.get('eval_pass_at_16'))} | "
            f"{fmt(row.get('eval_format_rate'))} | {fmt(row.get('eval_mean_tokens'), 1)} | "
            f"{fmt(row.get('eval_easy_greedy'))} | {fmt(row.get('eval_medium_greedy'))} | "
            f"{fmt(row.get('eval_hard_greedy'))} |"
        )
    if flagged:
        lines += ["", f"> ⚠️ 该行的题数少于本表基准（{reference_rows} 题），**不能和满量表直接比**；"
                      "重跑评估补齐题数后再引用。"]
    return lines


def render_seed_aggregate(rows: list[dict[str, Any]]) -> list[str]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["seed_group"], []).append(row)
    lines = [
        "| 臂（除 seed 外全部相同） | seeds | greedy mean [min, max] | pass@1 mean [min, max] |",
        "|---|---|---|---|",
    ]
    any_group = False
    for signature, members in sorted(groups.items()):
        evaluated = [row for row in members if row.get("eval_greedy_accuracy") is not None]
        if len(evaluated) < 2:
            continue
        any_group = True
        seeds = ", ".join(str(row.get("seed")) for row in sorted(evaluated, key=lambda r: str(r.get("seed"))))

        def stat(key: str) -> str:
            values = [row[key] for row in evaluated if row.get(key) is not None]
            if not values:
                return "—"
            mean = sum(values) / len(values)
            return f"{mean:.3f} [{min(values):.3f}, {max(values):.3f}]"

        lines.append(f"| {signature} | {seeds} | {stat('eval_greedy_accuracy')} | {stat('eval_pass_at_1')} |")
    if not any_group:
        lines.append("| （无）| — | — | 关键臂目前只有单个 seed：SOP §10.4 要求至少两个 seed，未满足时必须在 README 写明资源限制 |")
    return lines


def render_summary_md(
    rows: list[dict[str, Any]],
    *,
    notes: list[str],
    figures: list[dict[str, str]],
    data_sources: dict[str, str],
) -> str:
    generated = utcnow()
    lines = [
        "# RLVR 审计汇总（自动生成，勿手改）",
        "",
        f"- 生成时间：{generated}",
        f"- 预注册表：`{data_sources['registry']}`",
        f"- 训练日志：`{data_sources['runs']}`（归档）/ `{data_sources['logs']}`（未归档）",
        f"- 评估数据：`{data_sources['heldout']}`（400 题，与训练集无重叠）",
        "- 本文件与 `summary.csv` 只是**视图**；每个数字的原始来源是 "
        "`runs/<run_tag>/metrics.csv` 与 `runs/<run_tag>/eval.json`。",
        "- 口径：训练侧为 log_interval 采样点的组内统计；held-out 侧为 greedy + pass@k"
        "（每个 run 的采样数与温度见 summary.csv 的 `eval_num_samples` / `eval_temperature`），"
        "长度=到首个 EOS 的 token 数（不含 prompt）。",
        "",
        "## 1. 训练动态",
        "",
        f"| run_tag | evidence | steps | reward (首/末{TREND_WINDOW_MAX}点均值) | correct (首/末{TREND_WINDOW_MAX}点均值) | format last | tokens last | filtered last | ACR wrong last | 决策树提示 |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run_tag']}` | {row['evidence']} | {row.get('metrics_rows', 0)} | "
            f"{reward_trend_cell(row)} | "
            f"{fmt(row.get('correct_rate_first'))} → {fmt(row.get('correct_rate_last'))} "
            f"({fmt_delta(row.get('correct_rate_gain'))}) | "
            f"{fmt(row.get('format_rate_last'))} | {fmt(row.get('completion_tokens_last'), 1)} | "
            f"{fmt(row.get('filtered_fraction_last'))} | {fmt(row.get('acr_wrong_last'))} | "
            f"{row.get('decision_tree', '')} |"
        )

    lines += [
        "",
        "## 2. Held-out 评估矩阵（SOP §10.3）",
        "",
        "`*` = 该行尚未归档（`runs/<run_tag>/manifest.json` 不存在），数字请先归档再引用。",
        "",
    ]
    lines += render_matrix(rows)
    lines += [
        "",
        "### 2.1 reward–eval 对照（§11.2 决策树的定量版本）",
        "",
        "`reward_gain` 是训练期组内 average reward 的首末差，`heldout_gain` 是相对参考臂的 greedy 差值；"
        "两者单位不同，`reward_eval_gap` 只用于排序，不能当作物理量。",
        "",
        "| run_tag | reward_gain | train correct_gain | heldout_gain | reward_eval_gap | pass@k inversion | reward hacking 嫌疑 |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run_tag']}` | {reward_gain_cell(row)} | "
            f"{fmt_delta(row.get('correct_rate_gain'))} | {fmt_delta(row.get('heldout_gain'))} | "
            f"{fmt_delta(row.get('reward_eval_gap'))} | {fmt(row.get('passk_inversion'))} | "
            f"{fmt(row.get('hacking_suspect'))} |"
        )

    lines += [
        "",
        "## 3. 按难度的 held-out 正确率",
        "",
        "| run_tag | easy greedy | medium greedy | hard greedy | easy pass@1 | medium pass@1 | hard pass@1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run_tag']}` | {fmt(row.get('eval_easy_greedy'))} | {fmt(row.get('eval_medium_greedy'))} | "
            f"{fmt(row.get('eval_hard_greedy'))} | {fmt(row.get('eval_easy_pass1'))} | "
            f"{fmt(row.get('eval_medium_pass1'))} | {fmt(row.get('eval_hard_pass1'))} |"
        )

    lines += [
        "",
        "## 4. seed 汇总（SOP §10.4）",
        "",
    ]
    lines += render_seed_aggregate(rows)

    lines += [
        "",
        "## 5. 图表清单（caption 全文）",
        "",
    ]
    if figures:
        for figure in figures:
            lines += [f"### `{figure['path']}`", "", figure["caption"], ""]
    else:
        lines.append("（本次运行未生成图表：`--no-figures` 或 matplotlib 不可用）")
        lines.append("")

    lines += [
        "## 6. 警告与缺口",
        "",
    ]
    if notes:
        lines += [f"- {note}" for note in notes]
    flagged = [row for row in rows if row.get("warnings")]
    if not flagged:
        lines.append("- 无")
    for row in flagged:
        lines.append(f"- `{row['run_tag']}`: {row['warnings']}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _compact(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def figure_caption(rows: list[dict[str, Any]], extra: str) -> str:
    """Every figure carries its own provenance: SOP section 11.1 requires it."""

    models = sorted({row.get("model") or "unknown" for row in rows if row.get("model")})
    seeds = sorted({str(row.get("seed")) for row in rows if row.get("seed") not in (None, "")})
    samples = sorted({_compact(row.get("eval_num_samples")) for row in rows if row.get("eval_num_samples")})
    temps = sorted({_compact(row.get("eval_temperature")) for row in rows if row.get("eval_temperature")})
    parts = [
        f"models: {', '.join(models) if models else 'n/a'}",
        "train data: dataset/math_rlvr_train.jsonl",
        "eval data: dataset/math_heldout.jsonl (400 problems, no train overlap)",
        f"seeds: {', '.join(seeds) if seeds else 'n/a'}",
        f"held-out sampling: {('/'.join(samples)) if samples else 'n/a'} completions per problem"
        f" at temperature {('/'.join(temps)) if temps else 'n/a'}",
        "length convention: tokens up to the first EOS, prompt excluded",
        extra,
    ]
    return " | ".join(part for part in parts if part)


def _save(plt, fig, path: Path, caption: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.01, 0.005, caption, fontsize=6.5, va="bottom", wrap=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)  # a long test session must not accumulate open figures


def _series_by_run(rows, runs, column: str) -> list[tuple[str, list, list]]:
    out = []
    for row in rows:
        run = runs.get(row["run_tag"])
        if run is None or not run["rows"]:
            continue
        steps = [item.get("step") for item in run["rows"]]
        out.append((row["run_tag"], steps, series(run["rows"], column)))
    return out


def fig_training_dynamics(plt, rows, ctx, out_dir: Path, caption: str, dpi: int) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        ("avg_reward", "average reward (group mean)", axes[0][0]),
        ("format_rate", "format rate", axes[0][1]),
        ("correct_rate", "correctness (verifier)", axes[1][0]),
        ("avg_completion_tokens", "completion tokens", axes[1][1]),
    ]
    plotted = False
    for column, title, axis in panels:
        for tag, steps, values in _series_by_run(rows, ctx["runs"], column):
            axis.plot(steps, values, marker=".", linewidth=1.2, label=tag)
            plotted = True
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("optimizer step", fontsize=9)
        axis.grid(alpha=0.25)
    if plotted:
        axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle("Training dynamics per arm (log_interval sampling points)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    _save(plt, fig, out_dir / "fig1_training_dynamics.png", caption, dpi)
    return []


def fig_acr(plt, rows, ctx, out_dir: Path, caption: str, dpi: int) -> list[str]:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for row in rows:
        run = ctx["runs"].get(row["run_tag"])
        if run is None or not run["rows"]:
            continue
        steps = [item.get("step") for item in run["rows"]]
        correct = series(run["rows"], "all_correct_groups")
        wrong = series(run["rows"], "all_wrong_groups")
        valid = series(run["rows"], "valid_groups")
        totals = [
            (c or 0) + (w or 0) + (v or 0) if None not in (c, w, v) else None
            for c, w, v in zip(correct, wrong, valid)
        ]
        acr_correct = [_ratio(c, t) for c, t in zip(correct, totals)]
        acr_wrong = [_ratio(w, t) for w, t in zip(wrong, totals)]
        if any(value is not None for value in acr_wrong):
            axis.plot(steps, acr_wrong, marker="o", linewidth=1.2, label=f"{row['run_tag']} ACR_wrong")
            plotted = True
        if any(value is not None for value in acr_correct):
            axis.plot(steps, acr_correct, marker="x", linestyle="--", linewidth=1.0,
                      label=f"{row['run_tag']} ACR_correct")
            plotted = True
    if not plotted:
        axis.text(0.5, 0.5, "no group counters in the logs", ha="center", va="center", fontsize=10)
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("optimizer step", fontsize=9)
    axis.set_ylabel("share of groups", fontsize=9)
    axis.set_title("All-correct / all-wrong group ratio (SOP §6.7)", fontsize=11)
    axis.grid(alpha=0.25)
    if plotted:
        axis.legend(fontsize=7, loc="best")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(plt, fig, out_dir / "fig2_acr.png", caption, dpi)
    return []


def _grouped_bars(axis, rows, keys, labels, title, ylabel) -> None:
    width = 0.8 / max(len(keys), 1)
    plotted = False
    for offset, (key, label) in enumerate(zip(keys, labels)):
        positions, values = [], []
        for index, row in enumerate(rows):
            value = row.get(key)
            if value is None:
                continue
            positions.append(index - 0.4 + width / 2 + offset * width)
            values.append(value)
        if values:
            axis.bar(positions, values, width=width, label=label)
            plotted = True
    axis.set_xticks(range(len(rows)))
    axis.set_xticklabels([row["run_tag"] for row in rows], rotation=20, ha="right", fontsize=7)
    axis.set_title(title, fontsize=11)
    axis.set_ylabel(ylabel, fontsize=9)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25, axis="y")
    if plotted:
        axis.legend(fontsize=8)


def fig_heldout_arms(plt, rows, ctx, out_dir: Path, caption: str, dpi: int) -> list[str]:
    fig, axis = plt.subplots(figsize=(9, 4.8))
    _grouped_bars(
        axis, rows,
        ["eval_greedy_accuracy", "eval_pass_at_1", "eval_pass_at_16"],
        ["greedy", "pass@1", "pass@16"],
        "Held-out accuracy per arm (SOP §10.3)",
        "accuracy",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(plt, fig, out_dir / "fig3_heldout_arms.png", caption, dpi)
    return []


def fig_difficulty(plt, rows, ctx, out_dir: Path, caption: str, dpi: int) -> list[str]:
    fig, axis = plt.subplots(figsize=(9, 4.8))
    _grouped_bars(
        axis, rows,
        ["eval_easy_greedy", "eval_medium_greedy", "eval_hard_greedy"],
        ["easy", "medium", "hard"],
        "Held-out greedy accuracy by difficulty bucket (SOP §11.1)",
        "greedy accuracy",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(plt, fig, out_dir / "fig4_difficulty.png", caption, dpi)
    return []


def length_samples(run_dir: Path, eval_payload: dict[str, Any] | None) -> dict[str, list[int]]:
    """Per-completion token counts by sampling mode, from the archived run."""

    buckets: dict[str, list[int]] = {"greedy": [], "sampled": []}
    raw = run_dir / "eval.jsonl"
    if raw.is_file():
        with raw.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tokens = record.get("tokens")
                mode = record.get("mode", "sampled")
                if isinstance(tokens, int) and mode in buckets:
                    buckets[mode].append(tokens)
    if not any(buckets.values()) and eval_payload:
        for record in eval_payload.get("samples") or []:
            for key, mode in (("greedy", "greedy"), ("sampled", "sampled")):
                items = record.get(key)
                items = items if isinstance(items, list) else [items]
                for item in items:
                    if isinstance(item, dict) and isinstance(item.get("tokens"), int):
                        buckets[mode].append(item["tokens"])
    return {mode: values for mode, values in buckets.items() if values}


def fig_length_histogram(plt, rows, ctx, out_dir: Path, caption: str, dpi: int) -> list[str]:
    fig, axis = plt.subplots(figsize=(9, 4.6))
    plotted, missing = False, []
    for row in rows:
        buckets = length_samples(ctx["runs_dir"] / row["run_tag"], (ctx["runs"].get(row["run_tag"]) or {}).get("eval"))
        values = buckets.get("sampled") or buckets.get("greedy")
        if not values:
            missing.append(row["run_tag"])
            continue
        axis.hist(values, bins=30, alpha=0.5, label=f"{row['run_tag']} (n={len(values)})")
        plotted = True
    notes = [] if not missing else [
        "no per-completion token counts (need runs/<tag>/eval.jsonl, or eval.json with samples): "
        + ", ".join(missing)
    ]
    if not plotted:
        axis.text(0.5, 0.5, "no eval.jsonl found", ha="center", va="center", fontsize=10)
    axis.set_xlabel("completion tokens (up to first EOS)", fontsize=9)
    axis.set_ylabel("completions", fontsize=9)
    axis.set_title("Held-out completion length distribution", fontsize=11)
    axis.grid(alpha=0.25)
    if plotted:
        axis.legend(fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(plt, fig, out_dir / "fig5_length_histogram.png", caption, dpi)
    return notes


def fig_reward_eval_gap(plt, rows, ctx, out_dir: Path, caption: str, dpi: int) -> list[str]:
    fig, axis = plt.subplots(figsize=(7, 5))
    points = [row for row in rows if row.get("reward_gain") is not None and row.get("heldout_gain") is not None]
    notes: list[str] = []
    if points:
        axis.axhline(0, color="grey", linewidth=0.8)
        axis.axvline(0, color="grey", linewidth=0.8)
        for row in points:
            axis.scatter(row["reward_gain"], row["heldout_gain"], s=45)
            axis.annotate(row["run_tag"], (row["reward_gain"], row["heldout_gain"]),
                          textcoords="offset points", xytext=(5, 4), fontsize=7)
        axis.set_xlabel("training reward gain (last − first logged step)", fontsize=9)
        axis.set_ylabel("held-out greedy accuracy gain vs reference arm", fontsize=9)
    else:
        axis.text(0.5, 0.5, "need a train run plus an eval per arm", ha="center", va="center", fontsize=10)
        notes.append("needs runs that have both metrics.csv and eval.json")
    plotted_tags = {row["run_tag"] for row in points}
    skipped = [row["run_tag"] for row in rows if row["run_tag"] not in plotted_tags]
    if skipped:
        notes.append("not plotted (missing train gain or eval): " + ", ".join(skipped))
    axis.set_title("Reward gain vs held-out gain (reward hacking is the lower-right corner)", fontsize=11)
    axis.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(plt, fig, out_dir / "fig6_reward_eval_gap.png", caption, dpi)
    return notes


def render_figures(
    rows: list[dict[str, Any]],
    *,
    runs: dict[str, dict[str, Any]],
    runs_dir: Path,
    figures_dir: Path,
    dpi: int,
) -> tuple[list[dict[str, str]], list[str]]:
    """Produce the SOP section 11.1 figure set; a failure is a note, not a crash."""

    plt = ensure_mpl()
    if plt is None:
        return [], ["matplotlib is not installed; figures were skipped (CSV and markdown are unaffected)"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    ctx = {"runs": runs, "runs_dir": runs_dir}
    planned = [
        ("fig1_training_dynamics.png", fig_training_dynamics,
         "source: runs/<tag>/metrics.csv, one point per log_interval"),
        ("fig2_acr.png", fig_acr,
         "source: runs/<tag>/metrics.csv group counters; ACR = share of groups that step"),
        ("fig3_heldout_arms.png", fig_heldout_arms,
         "source: runs/<tag>/eval.json; a bar is absent when that run has no evaluation"),
        ("fig4_difficulty.png", fig_difficulty,
         "source: runs/<tag>/eval.json by_difficulty; the 400 held-out problems are split evenly"),
        ("fig5_length_histogram.png", fig_length_histogram,
         "source: runs/<tag>/eval.jsonl per-completion token counts"),
        ("fig6_reward_eval_gap.png", fig_reward_eval_gap,
         "reference arm is the first baseline-signature row of the registry"),
    ]
    figures: list[dict[str, str]] = []
    notes: list[str] = []
    for name, function, caption_extra in planned:
        caption = figure_caption(rows, caption_extra)
        try:
            notes.extend(function(plt, rows, ctx, figures_dir, caption, dpi) or [])
        except Exception as exc:  # a broken figure must never hide the numbers
            notes.append(f"{name}: figure failed ({type(exc).__name__}: {exc})")
            continue
        if (figures_dir / name).is_file():
            figures.append({"path": str(figures_dir / name), "caption": caption})
    return figures, notes


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reports/summary.{csv,md} and the SOP §11.1 figures from runs/<run_tag>/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--run-tag", action="append", default=None,
                        help="only these run tags (repeatable); default = registry + cold start")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--require-eval", action="store_true",
                        help="exit non-zero if any included run lacks eval.json")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows, notes, runs = run_rows(
        runs_dir=args.runs_dir,
        log_dir=args.log_dir,
        registry_path=args.registry,
        tags=args.run_tag,
    )
    if not any(row.get("metrics_rows") for row in rows):
        notes.append("no run has metrics yet: the figures written here are placeholders")
    figures: list[dict[str, str]] = []
    if not args.no_figures:
        figures, figure_notes = render_figures(
            rows,
            runs=runs,
            runs_dir=args.runs_dir,
            figures_dir=args.out_dir / "figures",
            dpi=args.dpi,
        )
        notes += figure_notes

    data_sources = {
        "registry": str(args.registry),
        "runs": str(args.runs_dir),
        "logs": str(args.log_dir),
        "heldout": "dataset/math_heldout.jsonl",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_md = render_summary_md(rows, notes=notes, figures=figures, data_sources=data_sources)
    (args.out_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    write_csv(rows, args.out_dir / "summary.csv")

    missing_eval = [row["run_tag"] for row in rows if row.get("eval_greedy_accuracy") is None]
    manifest = {
        "created_at": utcnow(),
        "runs_dir": str(args.runs_dir),
        "log_dir": str(args.log_dir),
        "registry": str(args.registry),
        "runs": [
            {
                "run_tag": row["run_tag"],
                "evidence": row["evidence"],
                "archive_status": row["archive_status"],
                "has_eval": row.get("eval_greedy_accuracy") is not None,
                "metrics_rows": row.get("metrics_rows"),
                "warnings": row.get("warnings", ""),
            }
            for row in rows
        ],
        "figures": figures,
        "missing_eval": missing_eval,
        "notes": notes,
        "files": {
            name: sha256_file(args.out_dir / name)
            for name in ("summary.md", "summary.csv")
            if (args.out_dir / name).is_file()
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if not args.quiet:
        print(f"wrote {args.out_dir}/summary.md, {args.out_dir}/summary.csv"
              + (f", {len(figures)} figures" if figures else ""))
        for note in notes:
            print(f"  - {note}")
        if missing_eval:
            print(f"  - runs without eval: {', '.join(missing_eval)}")
    if args.require_eval and missing_eval:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
