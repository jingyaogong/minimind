#!/usr/bin/env python
"""Classify raw rollouts into the SOP §10.4 error taxonomy.

输入是**原始 rollout 文件**（不是聚合指标）：

- 评估侧：``runs/<tag>/eval.jsonl``；``--mode greedy`` 用于人工直读，``sampled`` 用于分布统计；
- 训练侧：``logs/<tag>_samples.jsonl``。

输出两张表 + 抽样明细，供报告和人工审计引用：

1. **主类别**（互斥，按下面的优先级取第一个命中项）：
   ``correct`` > ``repetition`` > ``unclosed_think`` > ``redundant_answer`` >
   ``length_burn`` > ``no_answer`` > ``arithmetic_error`` > ``other``；
2. **标记**（非互斥）：同一行可以同时命中多项（例如"算错 + 没闭合 think"），
   主类别只保留最具体的那一项，标记表保留全部。

长度口径：``tokens >= 0.95 * max_new_tokens`` 记 ``length_burn``（被生成上限截断）。

think 口径：本 pipeline 的 prompt 末尾自带 ``<|im_start|>assistant\\n<think>\\n``
（``open_thinking=True``），所以补全里**只可能出现 ``</think>``**，没有开标签是正常的。
闭合与否以记录里的 ``think_closed`` 字段（``trainer/math_verifier.py`` 的 verifier 口径）
为准，缺失时才退回字符串判断：``"</think>" in text``。

    python scripts/error_analysis.py --run-tag mini_rule_seed42 --mode greedy
    python scripts/error_analysis.py --run-tag mini_rm_seed42 --mode greedy
    python scripts/error_analysis.py --path logs/mini_rm_seed42_samples.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_REPORT_DIR = Path("reports")
MARKER = "答案是"
REPETITION_MIN_PERIOD = 2  # 重复单元至少 2 个字符（单个字符连用另由更长的跨度兜住）
REPETITION_MAX_PERIOD = 64 # 重复单元最长 64 个字符（覆盖词、短语、整句级循环）
REPETITION_RUNS = 3        # 同一单元连续出现多少次算循环
REPETITION_SPAN = 12       # 且重复段总长 ≥12 字符，避免 "3，3，3" 这类正常重复被误报

PRIMARY_ORDER = (
    "correct",
    "repetition",
    "unclosed_think",
    "redundant_answer",
    "length_burn",
    "no_answer",
    "arithmetic_error",
    "other",
)

CATEGORY_NOTES = {
    "correct": "verifier 判定正确（仍可能同时命中重复等标记，见标记表）",
    "repetition": "同一单元（≤64 字符）连续重复 ≥3 次且跨度 ≥12 字符（退化循环）",
    "unclosed_think": "补全里没有 </think> 闭合（prompt 已开 think）—— 按口径 format_ok=False、不计分",
    "redundant_answer": "`答案是` 出现 ≥2 次（冗余候选）",
    "length_burn": "token 数达到生成上限的 95% 以上（被截断）",
    "no_answer": "verifier 抽不到数字",
    "arithmetic_error": "抽到数字但与 GT 不符",
    "other": "以上都不命中（不应出现；出现请查 verifier 口径）",
}

FLAG_NOTES = {
    "correct": "verifier 判定正确",
    "no_answer": "status == no_answer",
    "marker_unparsed": "`答案是` 出现在末段但数字解析失败（verifier 待复核）",
    "unclosed_think": "补全里没有 </think>（think 未闭合）",
    "repetition": "存在单元的连续重复（≥3 次且跨度 ≥12 字符）",
    "redundant_answer": "`答案是` ≥2 次",
    "length_burn": "接近生成上限",
    "arithmetic_error": "答案错",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-tag", help="自动定位 runs/<tag>/eval.jsonl，找不到再用 logs/<tag>_samples.jsonl")
    source.add_argument("--path", type=Path, help="显式指定 jsonl 文件")
    parser.add_argument("--mode", default="all", choices=["all", "greedy", "sampled"],
                        help="只统计某一种采样模式（默认全部）")
    parser.add_argument("--difficulty", default=None, help="只统计某个难度档")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="生成上限，用于判定 length_burn")
    parser.add_argument("--examples-per-category", type=int, default=2)
    parser.add_argument("--example-chars", type=int, default=400)
    parser.add_argument("--output", type=Path, default=None, help="默认 reports/error_analysis_<tag>.md")
    return parser.parse_args(argv)


def resolve_input(run_tag: str | None, path: Path | None) -> tuple[Path, str]:
    if path is not None:
        stem = path.stem
        tag = stem[: -len("_samples")] if stem.endswith("_samples") else stem
        return path, tag
    eval_path = DEFAULT_RUNS_DIR / run_tag / "eval.jsonl"
    if eval_path.is_file():
        return eval_path, run_tag
    samples_path = DEFAULT_LOG_DIR / f"{run_tag}_samples.jsonl"
    if samples_path.is_file():
        return samples_path, run_tag
    raise FileNotFoundError(f"找不到 {eval_path} 或 {samples_path}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def response_text(record: dict[str, Any]) -> str:
    return record.get("text") or record.get("completion") or ""


def token_count(record: dict[str, Any]) -> int:
    value = record.get("tokens")
    if value is None:
        value = record.get("completion_tokens")
    return int(value or 0)


def is_correct(record: dict[str, Any]) -> bool:
    if "correct" in record:
        return bool(record["correct"])
    return record.get("status") == "correct"


def think_closed(record: dict[str, Any], text: str) -> bool:
    """think 段是否闭合（口径与 trainer/math_verifier.py:split_think 一致）。

    prompt 末尾自带 ``<think>``（``open_thinking=True``），补全里只可能出现 ``</think>``，
    所以"没有开标签"绝不等于格式错误。优先信 verifier 写入的 ``think_closed`` 字段，
    字段缺失时才退回字符串判断。
    """

    value = record.get("think_closed")
    if value is None:
        return "</think>" in text
    return bool(value)


def final_segment(text: str) -> str:
    """``</think>`` 之后的作答段；未闭合时与 verifier 一致，不把推理段当作答。"""

    if "</think>" not in text:
        return "" if "<think>" in text else text
    return text.rsplit("</think>", 1)[-1]


def max_consecutive_repeats(text: str, *, max_period: int = REPETITION_MAX_PERIOD,
                            min_span: int = REPETITION_SPAN,
                            stop_at: int = REPETITION_RUNS) -> int:
    """最长"连续重复"次数。

    去掉空白后，枚举所有长度在 ``[REPETITION_MIN_PERIOD, max_period]`` 之间的候选
    单元，统计同一单元连续出现的最多次数 ``run``，只保留 ``run * 单元长度 >= min_span``
    的那一类（跨度太小不算退化循环）。找不到返回 1。

    单元长度不固定，是因为真实退化循环的粒度并不一致（词、短语、整句都见过）；
    找到 ``stop_at`` 次即提前返回——flag 判定只需要知道"够不够 3 次"。
    """

    compact = "".join(text.split())
    n = len(compact)
    best = 1
    limit = min(max_period, n // 2)
    for period in range(REPETITION_MIN_PERIOD, limit + 1):
        need = max(REPETITION_RUNS, -(-min_span // period))
        i = 0
        while i + need * period <= n:
            block = compact[i:i + period]
            run = 1
            cursor = i + period
            while run < need and cursor + period <= n and compact[cursor:cursor + period] == block:
                run += 1
                cursor += period
            if run >= need:
                while cursor + period <= n and compact[cursor:cursor + period] == block:
                    run += 1
                    cursor += period
                if run > best:
                    best = run
                    if best >= stop_at:
                        return best
                i = cursor
            else:
                i += 1
    return best


def collect_flags(record: dict[str, Any], *, max_new_tokens: int) -> set[str]:
    text = response_text(record)
    status = record.get("status")
    final = final_segment(text)
    flags = set()
    if is_correct(record):
        flags.add("correct")
    if status == "no_answer":
        flags.add("no_answer")
        if MARKER in final:
            flags.add("marker_unparsed")
    if not think_closed(record, text):
        flags.add("unclosed_think")
    if max_consecutive_repeats(text) >= REPETITION_RUNS:
        flags.add("repetition")
    if text.count(MARKER) >= 2:
        flags.add("redundant_answer")
    if token_count(record) >= 0.95 * max_new_tokens:
        flags.add("length_burn")
    if status == "incorrect":
        flags.add("arithmetic_error")
    return flags


def primary_category(flags: set[str]) -> str:
    for name in PRIMARY_ORDER:
        if name in flags:
            return name
    return "other"


def render_markdown(tag: str, path: Path, rows, args) -> str:
    counts = {name: 0 for name in PRIMARY_ORDER}
    flag_counts = {name: 0 for name in FLAG_NOTES}
    by_difficulty: dict[str, dict[str, int]] = {}
    examples: dict[str, list[dict[str, Any]]] = {}

    for record in rows:
        flags = collect_flags(record, max_new_tokens=args.max_new_tokens)
        category = primary_category(flags)
        counts[category] = counts.get(category, 0) + 1
        for name in flags:
            flag_counts[name] = flag_counts.get(name, 0) + 1
        difficulty = record.get("difficulty") or "n/a"
        bucket = by_difficulty.setdefault(difficulty, {})
        bucket[category] = bucket.get(category, 0) + 1
        if category != "correct" and len(examples.get(category, [])) < args.examples_per_category:
            examples.setdefault(category, []).append(record)

    total = len(rows) or 1
    lines = [
        f"# 错误类别分析：`{tag}`",
        "",
        f"- 输入：`{path}`（共 {len(rows)} 条）",
        f"- 过滤：mode={args.mode}, difficulty={args.difficulty or '全部'}",
        f"- 长度口径：tokens ≥ 0.95×{args.max_new_tokens} = {0.95 * args.max_new_tokens:.1f} 记 `length_burn`",
        f"- 复算命令：`python scripts/error_analysis.py --path {path} "
        f"--mode {args.mode} --max-new-tokens {args.max_new_tokens}`",
        "",
        "## 1. 主类别（互斥，按优先级）",
        "",
        "| 类别 | 条数 | 占比 | 说明 |",
        "|---|---:|---:|---|",
    ]
    for name in PRIMARY_ORDER:
        count = counts.get(name, 0)
        if count == 0:
            continue
        lines.append(f"| `{name}` | {count} | {count / total:.1%} | {CATEGORY_NOTES[name]} |")
    empty = [name for name in PRIMARY_ORDER if not counts.get(name)]
    if empty:
        lines += ["", "未命中（各 0 条，口径已检查）：" + "、".join(f"`{name}`" for name in empty)]
    lines += ["", "## 2. 标记（非互斥，同一行可命中多项）", "", "| 标记 | 条数 | 占比 | 说明 |", "|---|---:|---:|---|"]
    for name in FLAG_NOTES:
        count = flag_counts.get(name, 0)
        if count == 0:
            continue
        lines.append(f"| `{name}` | {count} | {count / total:.1%} | {FLAG_NOTES[name]} |")
    empty_flags = [name for name in FLAG_NOTES if not flag_counts.get(name)]
    if empty_flags:
        lines += ["", "未命中（各 0 条，口径已检查）：" + "、".join(f"`{name}`" for name in empty_flags)]

    if by_difficulty and set(by_difficulty) != {"n/a"}:
        lines += ["", "## 3. 难度 × 主类别", "", "| 难度 | 条数 | " + " | ".join(f"`{n}`" for n in PRIMARY_ORDER) + " |",
                  "|---|---:|" + "---:|" * len(PRIMARY_ORDER)]
        for difficulty in sorted(by_difficulty):
            bucket = by_difficulty[difficulty]
            row_total = sum(bucket.values())
            cells = " | ".join(str(bucket.get(name, 0)) if bucket.get(name) else "·" for name in PRIMARY_ORDER)
            lines.append(f"| {difficulty} | {row_total} | {cells} |")

    lines += ["", f"## 4. 抽样明细（每个失败类别最多 {args.examples_per_category} 条）", ""]
    for name in PRIMARY_ORDER:
        records = examples.get(name) or []
        if not records:
            continue
        lines += [f"### {name}", ""]
        for record in records:
            text = response_text(record)
            snippet = text[: args.example_chars]
            if len(text) > args.example_chars:
                snippet += f"…（共 {len(text)} 字符）"
            lines += [
                f"- `{record.get('difficulty', 'n/a')}` | gt=`{record.get('gt')}` | "
                f"extracted=`{record.get('extracted')}` | tokens={token_count(record)} | "
                f"status=`{record.get('status')}`",
                "",
                "```text",
                snippet,
                "```",
                "",
            ]
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        path, tag = resolve_input(args.run_tag, args.path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows = load_rows(path)
    if args.mode != "all":
        rows = [row for row in rows if row.get("mode", "greedy") == args.mode]
    if args.difficulty:
        rows = [row for row in rows if row.get("difficulty") == args.difficulty]
    if not rows:
        print("error: 过滤后没有数据行（检查 --mode / --difficulty）", file=sys.stderr)
        return 2

    markdown = render_markdown(tag, path, rows, args)
    output = args.output or DEFAULT_REPORT_DIR / f"error_analysis_{tag}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
