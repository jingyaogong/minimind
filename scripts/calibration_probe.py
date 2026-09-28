#!/usr/bin/env python
"""难度标定探针（difficulty calibration probe）。

为什么需要它
------------
RLVR 的成败取决于任务难度是否落在起点策略的能力边界内。若起点策略
pass@1 ≈ 0，则 GRPO 的同一组 rollout 奖励几乎全相同（组内方差为 0），
优势函数恒等于 0，策略梯度消失——即 MiniMind README 所说的「奖励稀疏」，
类比为「让小学生做高考数学题」。

本项目在 pilot 阶段实测到了这个失败模式：官方 `full_sft_768.pth` 在
`math_heldout.jsonl` 前 20 题（标称 easy，实际是 11–99 的两位数运算）上
greedy 全错、采样 pass@1 ≈ 0.05，且错误形态是「模板句 + 一个看着合理的数字」
（`66*23`→2888、`97*61`→4378）。所以正式跑 GRPO 之前必须先标定难度。

本脚本用**与主数据集完全相同的出题模板**（`synthetic_question`）生成一组
跨难度档的探针题，再用 `scripts/eval_heldout.py` 评测各档的 pass@1，
挑出落在 [0.2, 0.6] 的档位作为正式数据集的难度锚点。

用法::

    # 本地生成（torch-free，几秒）
    python scripts/calibration_probe.py --out dataset/probe/calibration_probe.jsonl

    # 上传后在 GPU 机器上评测（每档 20 题 × 8 采样）
    python scripts/eval_heldout.py \
        --data_path dataset/probe/calibration_probe.jsonl \
        --model_path out/math_cold_seed42_768.pth \
        --output runs/probe/calibration.json \
        --limit 0 --num_samples 8 --max_new_tokens 256

探针文件**只是诊断工具**：它的 sha256 不会写进 `data.sha256`，评测时
`data_matches_manifest` 必然是 `false`（脚本会打印 WARNING），这是预期行为。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.clean_math_data import (  # noqa: E402
    canonical_key,
    make_rlvr_record,
    safe_eval,
    synthetic_question,
)

PROBE_SOURCE = "calibration-probe"
QUESTION_PREFIX = "请计算 "
QUESTION_SUFFIX = "，并在"

TierGenerator = Callable[[random.Random], str]


def _gen_add_1x1(rng: random.Random) -> str:
    return f"{rng.randint(1, 9)}+{rng.randint(1, 9)}"


def _gen_sub_1x1(rng: random.Random) -> str:
    first = rng.randint(2, 18)
    return f"{first}-{rng.randint(1, first - 1)}"


def _gen_mul_1x1(rng: random.Random) -> str:
    return f"{rng.randint(1, 9)}*{rng.randint(1, 9)}"


def _gen_add_2x1(rng: random.Random) -> str:
    return f"{rng.randint(11, 99)}+{rng.randint(1, 9)}"


def _gen_mul_2x1(rng: random.Random) -> str:
    return f"{rng.randint(11, 99)}*{rng.randint(2, 9)}"


def _gen_add_2x2(rng: random.Random) -> str:
    return f"{rng.randint(11, 99)}+{rng.randint(11, 99)}"


TIERS: dict[str, tuple[TierGenerator, str]] = {
    "probe-add-1x1": (_gen_add_1x1, "一位数 + 一位数"),
    "probe-sub-1x1": (_gen_sub_1x1, "一位数 - 一位数（结果非负）"),
    "probe-mul-1x1": (_gen_mul_1x1, "一位数 × 一位数（九九表）"),
    "probe-add-2x1": (_gen_add_2x1, "两位数 + 一位数"),
    "probe-mul-2x1": (_gen_mul_2x1, "两位数 × 一位数"),
    "probe-add-2x2": (_gen_add_2x2, "两位数 + 两位数（当前 easy 的量级，对照锚点）"),
}


def expression_from_question(text: str) -> str:
    """把 ``请计算 43+38，并在…`` 还原成 ``43+38``（测试与人工审计用）。"""

    if not isinstance(text, str) or not text.startswith(QUESTION_PREFIX):
        raise ValueError(f"unexpected question wording: {text!r}")
    body = text[len(QUESTION_PREFIX):]
    if QUESTION_SUFFIX not in body:
        raise ValueError(f"unexpected question wording: {text!r}")
    return body.split(QUESTION_SUFFIX, 1)[0].strip()


def build_probe(per_tier: int = 20, seed: int = 42) -> list[dict[str, Any]]:
    """生成 ``per_tier`` × ``len(TIERS)`` 条探针，顺序与取值都由 seed 决定。"""

    if per_tier < 1:
        raise ValueError("per_tier must be >= 1")

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for tier, (generator, _description) in TIERS.items():
        pool: dict[str, dict[str, Any]] = {}
        stats: Counter = Counter()
        attempts = 0
        max_attempts = max(1000, per_tier * 400)
        while len(pool) < per_tier and attempts < max_attempts:
            attempts += 1
            expr = generator(rng)
            try:
                gt = safe_eval(expr)
            except ValueError as exc:
                stats[f"rejected:{type(exc).__name__}"] += 1
                continue
            question = synthetic_question(expr)
            key = canonical_key(question)
            if key in pool:
                continue
            pool[key] = {
                "key": key,
                "question": question,
                "gt": gt,
                "difficulty": tier,
                "source": PROBE_SOURCE,
                "expression": expr,
            }
        if len(pool) < per_tier:
            raise RuntimeError(
                f"could not generate enough {tier} rows: {len(pool)}/{per_tier}"
            )
        records.extend(make_rlvr_record(item) for item in pool.values())

    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="build the difficulty calibration probe")
    parser.add_argument("--out", type=Path,
                        default=Path("dataset/probe/calibration_probe.jsonl"),
                        help="输出 jsonl 路径")
    parser.add_argument("--per-tier", type=int, default=20, help="每档题数")
    parser.add_argument("--seed", type=int, default=42, help="生成 seed（决定题目与顺序）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = build_probe(per_tier=args.per_tier, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} rows to {args.out}")
    for tier, (_generator, description) in TIERS.items():
        print(f"  {tier:<16} {description}")
    print("提示：探针的 sha256 不写进 data.sha256，评测时 data_matches_manifest=false 是预期的。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
