#!/usr/bin/env python
"""逐题配对比较：把若干 RL 臂和冷启动基线放在**同一批 held-out 题**上做配对检验。

为什么不能用两个百分比直接相减
--------------------------------
`greedy 0.3725 → 0.3950` 看着像 "涨了 2.25 个点"，但那只是净差。400 道题里究竟有多少
题从错变对、多少题从对变错、这个净变化在零假设下有多罕见，只有**逐题配对**才答得出来。

具体做法（McNemar 精确检验）：

* 固定同一批题（按 ``sample_id`` 取交集，缺题会显式报警而不是静默丢）；
* 每题二元化——greedy 是否答对 / 16 次采样里是否至少对一次（pass@k 的"能力上界"代理）；
* 统计两个方向的变化数 ``变对(b)`` 与 ``变错(c)``，在 H0（变化方向对称）下
  ``b ~ Binomial(b+c, 0.5)``，取双边精确 p 值。

这正是 SOP 想审计的问题：reward 上升到底是"模型变强了"，还是"同一批本来就会的题被
采样得更频繁"。如果 ``greedy`` 有显著净增、而 ``pass@16`` 净变化不显著，那结论就是
**分布锐化而非能力扩展**。

用法::

    python scripts/paired_compare.py \
        --baseline math_cold_cot_seed42 \
        --arms mini_rule_seed42 mini_rule_seed7 mini_rule_dynamic_seed42 mini_rule_len_seed42

输出一篇 markdown（默认 ``reports/paired_compare.md``）和一份同内容的 JSON，
所有数字都能从 ``runs/<tag>/eval.json`` 复算。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "paired_compare.md"
TIERS = ("easy", "medium", "hard")

Record = dict[str, Any]
#: 每题被二元化成"这个 run 在这题上算不算成功"，旗标函数就是二元化的定义。
Flag = Callable[[Record], bool]


def greedy_correct(record: Record) -> bool:
    """greedy 那一次 是否判对。"""

    return bool(record["greedy"]["correct"])


def any_sample_correct(record: Record) -> bool:
    """16 次采样里是否至少对一次（pass@16 的逐题版本 = "这题解题能力存在吗"）。"""

    return int(record["sampled_n_correct"]) > 0


def binomial_two_sided(b: int, c: int) -> float:
    """McNemar 精确检验的双边 p 值（H0: 变化方向对称）。"""

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_eval(run_dir: Path) -> dict[str, Record]:
    path = run_dir / "eval.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (run eval_heldout.py first)")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("samples")
    if not records:
        raise ValueError(f"{path} has no 'samples' section; cannot pair per question")
    return {str(row["sample_id"]): row for row in records}


def load_payload(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))


def pair(baseline: dict[str, Record], arm: dict[str, Record]) -> tuple[list[str], list[str]]:
    """返回 (共同题目 id, 只在一侧出现的 id)——缺题必须暴露出来，不能静默取交集。"""

    shared = sorted(set(baseline) & set(arm))
    only = sorted(set(baseline) ^ set(arm))
    if not shared:
        raise ValueError("no shared sample_id between the two runs")
    return shared, only


def mcnemar(baseline: dict[str, Record], arm: dict[str, Record], flag: Flag) -> dict[str, Any]:
    shared, only = pair(baseline, arm)
    became = sum(1 for key in shared if not flag(baseline[key]) and flag(arm[key]))
    broke = sum(1 for key in shared if flag(baseline[key]) and not flag(arm[key]))
    return {
        "paired": len(shared),
        "unpaired": len(only),
        "became_correct": became,
        "broke": broke,
        "net": became - broke,
        "p_value": round(binomial_two_sided(became, broke), 4),
    }


def tier_greedy(run_dir: Path, tier: str) -> tuple[int, float]:
    records = load_payload(run_dir)["samples"]
    rows = [row for row in records if row.get("difficulty") == tier]
    if not rows:
        return 0, 0.0
    return len(rows), sum(1 for row in rows if row["greedy"]["correct"]) / len(rows)


def overall_row(run_dir: Path) -> dict[str, Any]:
    payload = load_payload(run_dir)
    return {
        "run": run_dir.name,
        "greedy_accuracy": payload["greedy_accuracy"],
        "pass_at_1": payload["sampled_pass_at_1"],
        "pass_at_4": payload.get("pass_at_4"),
        "pass_at_8": payload.get("pass_at_8"),
        "pass_at_16": payload.get("pass_at_16"),
        "format_rate": payload["format_rate"],
        "mean_completion_tokens": payload.get("mean_completion_tokens"),
        "rows_evaluated": payload.get("rows_evaluated"),
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(runs_dir: Path, baseline_tag: str, arm_tags: Iterable[str]) -> dict[str, Any]:
    baseline_dir = runs_dir / baseline_tag
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"{baseline_dir} not found")
    baseline = load_eval(baseline_dir)

    arms: list[dict[str, Any]] = []
    for tag in arm_tags:
        arm_dir = runs_dir / tag
        if not arm_dir.is_dir():
            raise FileNotFoundError(f"{arm_dir} not found")
        arm = load_eval(arm_dir)
        _shared, only = pair(baseline, arm)
        arms.append({
            "run": tag,
            "missing_from_pairing": only,
            "overall": overall_row(arm_dir),
            "greedy_vs_baseline": mcnemar(baseline, arm, greedy_correct),
            "capability_vs_baseline": mcnemar(baseline, arm, any_sample_correct),
            "by_tier": {tier: tier_greedy(arm_dir, tier)[1] for tier in TIERS},
        })

    baseline_tiers = {tier: tier_greedy(baseline_dir, tier) for tier in TIERS}
    return {
        "baseline": baseline_tag,
        "baseline_overall": overall_row(baseline_dir),
        "baseline_by_tier": {tier: {"rows": n, "greedy_accuracy": acc} for tier, (n, acc) in baseline_tiers.items()},
        "arms": arms,
    }


def render_markdown(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    base_overall = report["baseline_overall"]
    lines: list[str] = [
        "# 逐题配对比较（McNemar 精确检验）",
        "",
        f"基线：`{baseline}`（{fmt(base_overall['rows_evaluated'])} 题）。",
        "`greedy 净变` = 从错变对的题数 − 从对变错的题数；`p` 是双边精确 p 值，**p < 0.05 才算显著**。",
        "`能力净变` 用 pass@16 的逐题版本（这题 16 次采样里至少对过一次），衡量的是**解题能力是否存在**，",
        "不是采样有多频繁——两者背离就是「分布锐化而非能力扩展」的直接证据。",
        "",
        "## 1. 总体指标",
        "",
        "| run | greedy | pass@1 | pass@4 | pass@8 | pass@16 | format | 平均长度 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows = [dict(base_overall, run=f"{baseline}（基线）")] + [arm["overall"] for arm in report["arms"]]
    for row in rows:
        lines.append(
            f"| `{row['run']}` | {fmt(row['greedy_accuracy'])} | {fmt(row['pass_at_1'])} "
            f"| {fmt(row['pass_at_4'])} | {fmt(row['pass_at_8'])} | {fmt(row['pass_at_16'])} "
            f"| {fmt(row['format_rate'])} | {fmt(row['mean_completion_tokens'], 0)} |"
        )

    lines += [
        "",
        "## 2. 配对检验（vs 基线，同一批题）",
        "",
        "| run | 配对题数 | greedy 变对 | greedy 变错 | greedy 净变 | p | 能力变对 | 能力变错 | 能力净变 | p |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in report["arms"]:
        greedy = arm["greedy_vs_baseline"]
        capability = arm["capability_vs_baseline"]
        lines.append(
            f"| `{arm['run']}` | {greedy['paired']} | {greedy['became_correct']} | {greedy['broke']} "
            f"| {greedy['net']:+d} | {greedy['p_value']:.4f} | {capability['became_correct']} "
            f"| {capability['broke']} | {capability['net']:+d} | {capability['p_value']:.4f} |"
        )
    unpaired = {arm["run"]: arm["missing_from_pairing"] for arm in report["arms"] if arm["missing_from_pairing"]}
    if unpaired:
        lines += ["", "> ⚠️ 有题没能配对（两侧 sample_id 不一致），上面的统计**不含**这些题："
                      + "; ".join(f"`{tag}` {len(ids)} 题" for tag, ids in unpaired.items())]

    lines += [
        "",
        "## 3. 分档 greedy 正确率",
        "",
        "| 档位 | 基线题数 | " + " | ".join(f"`{baseline}`" for _ in [0]) + " | "
        + " | ".join(f"`{arm['run']}`" for arm in report["arms"]) + " |",
        "|---" * (len(report["arms"]) + 2) + "|",
    ]
    for tier in TIERS:
        cells = [tier, str(report["baseline_by_tier"][tier]["rows"]),
                 f"{report['baseline_by_tier'][tier]['greedy_accuracy']:.3f}"]
        for arm in report["arms"]:
            delta = arm["by_tier"][tier] - report["baseline_by_tier"][tier]["greedy_accuracy"]
            cells.append(f"{arm['by_tier'][tier]:.3f} ({delta:+.3f})")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 4. 怎么读这张表",
        "",
        "- **greedy 显著为正、能力净变不显著**：RL 只是把已有的正确路径采样得更频繁，"
        "没有扩展能力上界（对应 SOP 参考文献 arXiv:2504.13837）。",
        "- **greedy 不显著、能力也不显著**：这个配置下 RL 没带来可测的变化，"
        "要去看 ACR（退化组比例）解释——梯度只来自「组内既有对又有错」的 prompt。",
        "- **多个种子同向**：单看一个种子 p=0.06 可能只是噪声，两个种子都指向同一方向才叫证据。",
        "",
        "所有数字都能从 `runs/<tag>/eval.json` 复算；本文件的 JSON 版本在 `reports/paired_compare.json`。",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default="math_cold_cot_seed42", help="冷启动基线 run_tag")
    parser.add_argument("--arms", nargs="+", required=True, help="要比对的 RL 臂 run_tag")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args.runs_dir, args.baseline, args.arms)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    markdown = render_markdown(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    json_path = args.json_output or args.output.with_suffix(".json")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(markdown)
    print(f"wrote {args.output} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
