#!/usr/bin/env python
"""Compare two runs' training trajectories, step by step.

两种用途：

1. **确定性对照**：同一条命令、同一个 seed 跑两遍。逐字段完全一致说明训练管线是确定性的；
   不一致则必须把"数值非确定性"写进结论局限，而不是把差异归因给超参。
2. **单变量消融**：只改一个超参时，看轨迹从第几步开始分岔、差异有多大、分岔是"数值噪声"
   还是"真的走了另一条路"。

    python scripts/compare_runs.py --a mini_rule_seed42 --b mini_rule_rerun_seed42
    python scripts/compare_runs.py --a mini_rule_seed42 --b mini_rule_len_seed42 --expect-identical
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DEFAULT_LOG_DIR = Path("logs")
SKIP_FIELDS = {"timestamp"}  # 两次运行的时间戳必然不同，不参与比较


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--a", required=True, help="基线 run_tag")
    parser.add_argument("--b", required=True, help="对照 run_tag")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="metrics CSV 所在目录")
    parser.add_argument("--expect-identical", action="store_true",
                        help="要求两者逐字段一致（确定性对照用）；有差异则退出码 1")
    parser.add_argument("--max-rows", type=int, default=8, help="最多列几条差异明细")
    return parser.parse_args(argv)


def load_metrics(path: Path) -> tuple[dict[str, dict[str, str]], int, int]:
    """Return (rows_by_step, total_rows, duplicate_steps)."""

    rows: dict[str, dict[str, str]] = {}
    duplicates = 0
    total = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            total += 1
            step = record.get("step", "")
            if step in rows:
                duplicates += 1
            rows[step] = {k: v for k, v in record.items() if k not in SKIP_FIELDS}
    return rows, total, duplicates


def numeric(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main(argv=None) -> int:
    args = parse_args(argv)
    log_dir = Path(args.log_dir)
    paths = {tag: log_dir / f"{tag}_metrics.csv" for tag in (args.a, args.b)}
    for tag, path in paths.items():
        if not path.is_file():
            print(f"FAIL 找不到 {tag} 的 metrics：{path}", file=sys.stderr)
            return 2

    a, a_total, a_dups = load_metrics(paths[args.a])
    b, b_total, b_dups = load_metrics(paths[args.b])
    print(f"{args.a}: {a_total} 行（重复 step {a_dups}） | {args.b}: {b_total} 行（重复 step {b_dups}）")

    steps = sorted(set(a) & set(b), key=int)
    only_a = sorted(set(a) - set(b), key=int)
    only_b = sorted(set(b) - set(a), key=int)
    print(f"共同 step: {len(steps)} | 只出现在 {args.a}: {len(only_a)} | 只出现在 {args.b}: {len(only_b)}")
    if not steps:
        print("FAIL 没有共同 step（log_interval 或 step 偏移不同），先确认两次运行用的是同一份配置", file=sys.stderr)
        return 2

    fields = [k for k in a[steps[0]] if k != "step"]
    differing = [s for s in steps if a[s] != b[s]]
    print(f"逐字段完全相同的 step: {len(steps) - len(differing)} / {len(steps)}")

    worst: dict[str, tuple[float, str, str, str]] = {}
    non_numeric: dict[str, int] = {}
    first_diff = None
    for step in steps:
        for key in fields:
            left, right = a[step].get(key), b[step].get(key)
            if left == right:
                continue
            first_diff = first_diff or step
            lnum, rnum = numeric(left), numeric(right)
            if lnum is None or rnum is None:
                non_numeric[key] = non_numeric.get(key, 0) + 1
                continue
            delta = abs(lnum - rnum)
            if key not in worst or delta > worst[key][0]:
                worst[key] = (delta, left, right, step)

    print()
    if first_diff is None:
        print("→ 逐 step、逐字段完全一致：训练管线是确定性的，两次运行是同一条轨迹。")
        return 0

    print(f"第一处差异出现在 step {first_diff}")
    print(f"{'字段':<28}{'最大绝对差':>14}   出现位置 / {args.a} vs {args.b}")
    for key, (delta, left, right, step) in sorted(worst.items(), key=lambda kv: -kv[1][0])[: args.max_rows]:
        print(f"{key:<28}{delta:>14.3e}   step {step}: {left} vs {right}")
    for key, count in sorted(non_numeric.items(), key=lambda kv: -kv[1])[: args.max_rows]:
        print(f"{key:<28}{'>非数值<':>14}   {count} 个 step 取值不同")
    print()
    print(f"→ {len(differing)}/{len(steps)} 个 step 存在差异。若这是确定性对照，"
          "说明训练管线存在数值非确定性（需写进局限）；若是消融对照，请结合差异首次出现的 step 解释。")
    return 1 if args.expect_identical else 0


if __name__ == "__main__":
    raise SystemExit(main())
