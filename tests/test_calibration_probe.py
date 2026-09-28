import json
import re
import tempfile
import unittest
from pathlib import Path

from scripts.calibration_probe import (
    PROBE_SOURCE,
    TIERS,
    build_probe,
    expression_from_question,
    main,
)
from scripts.clean_math_data import safe_eval, synthetic_question

EXPR_RE = re.compile(r"^(\d+)([+\-*])(\d+)$")


def expressions_by_tier(records):
    grouped = {}
    for record in records:
        expr = expression_from_question(record["conversations"][0]["content"])
        grouped.setdefault(record["difficulty"], []).append(expr)
    return grouped


class ProbeShapeTests(unittest.TestCase):
    def test_all_tiers_present_with_requested_count(self):
        records = build_probe(per_tier=5, seed=42)
        self.assertEqual(len(records), len(TIERS) * 5)
        counts = {}
        for record in records:
            counts[record["difficulty"]] = counts.get(record["difficulty"], 0) + 1
        self.assertEqual(counts, {tier: 5 for tier in TIERS})

    def test_deterministic_for_same_seed(self):
        first = build_probe(per_tier=5, seed=42)
        second = build_probe(per_tier=5, seed=42)
        self.assertEqual(first, second)

    def test_different_seed_changes_questions(self):
        first = [r["sample_id"] for r in build_probe(per_tier=5, seed=42)]
        second = [r["sample_id"] for r in build_probe(per_tier=5, seed=7)]
        self.assertNotEqual(first, second)

    def test_question_wording_matches_the_main_dataset_template(self):
        for record in build_probe(per_tier=3, seed=42):
            expr = expression_from_question(record["conversations"][0]["content"])
            self.assertEqual(record["conversations"][0]["content"], synthetic_question(expr))

    def test_ground_truth_is_the_evaluated_expression(self):
        for record in build_probe(per_tier=5, seed=42):
            expr = expression_from_question(record["conversations"][0]["content"])
            self.assertEqual(record["gt"], safe_eval(expr), expr)

    def test_sample_ids_unique(self):
        records = build_probe(per_tier=5, seed=42)
        ids = [record["sample_id"] for record in records]
        self.assertEqual(len(ids), len(set(ids)))
        for record in records:
            self.assertTrue(record["sample_id"].startswith(record["difficulty"] + "-"))

    def test_rejects_non_positive_per_tier(self):
        with self.assertRaises(ValueError):
            build_probe(per_tier=0)


class ProbeSchemaTests(unittest.TestCase):
    """探针必须能被 scripts/eval_heldout.read_rows 直接吃下。"""

    def test_rows_have_the_rlvr_schema(self):
        for record in build_probe(per_tier=3, seed=42):
            self.assertEqual(
                set(record),
                {"conversations", "gt", "source", "difficulty", "sample_id"},
            )
            self.assertEqual(record["source"], PROBE_SOURCE)
            conversations = record["conversations"]
            self.assertEqual(len(conversations), 1)
            self.assertEqual(conversations[0]["role"], "user")
            self.assertIsInstance(conversations[0]["content"], str)

    def test_expression_from_question_rejects_foreign_wording(self):
        with self.assertRaises(ValueError):
            expression_from_question("43+38 等于几？")
        with self.assertRaises(ValueError):
            expression_from_question("请计算 43+38 吧")

    def test_no_duplicate_questions_within_a_tier(self):
        for tier, exprs in expressions_by_tier(build_probe(per_tier=20, seed=42)).items():
            self.assertEqual(len(exprs), len(set(exprs)), tier)


class TierRangeTests(unittest.TestCase):
    """每个档位的操作数必须落在该档声明的范围内（防止模板被改坏）。"""

    def test_declared_operand_ranges(self):
        grouped = expressions_by_tier(build_probe(per_tier=20, seed=42))

        for expr in grouped["probe-add-1x1"]:
            left, op, right = EXPR_RE.match(expr).groups()
            self.assertEqual(op, "+")
            self.assertTrue(1 <= int(left) <= 9 and 1 <= int(right) <= 9, expr)

        for expr in grouped["probe-sub-1x1"]:
            left, op, right = EXPR_RE.match(expr).groups()
            self.assertEqual(op, "-")
            self.assertGreaterEqual(int(left), 2)
            self.assertLess(int(right), int(left))

        for expr in grouped["probe-mul-1x1"]:
            left, op, right = EXPR_RE.match(expr).groups()
            self.assertEqual(op, "*")
            self.assertTrue(1 <= int(left) <= 9 and 1 <= int(right) <= 9, expr)

        for expr in grouped["probe-add-2x1"]:
            left, op, right = EXPR_RE.match(expr).groups()
            self.assertEqual(op, "+")
            self.assertTrue(11 <= int(left) <= 99 and 1 <= int(right) <= 9, expr)

        for expr in grouped["probe-mul-2x1"]:
            left, op, right = EXPR_RE.match(expr).groups()
            self.assertEqual(op, "*")
            self.assertTrue(11 <= int(left) <= 99 and 2 <= int(right) <= 9, expr)

        for expr in grouped["probe-add-2x2"]:
            left, op, right = EXPR_RE.match(expr).groups()
            self.assertEqual(op, "+")
            self.assertTrue(11 <= int(left) <= 99 and 11 <= int(right) <= 99, expr)


class ProbeCliTests(unittest.TestCase):
    def test_main_writes_parseable_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "probe.jsonl"
            self.assertEqual(main(["--out", str(out), "--per-tier", "4", "--seed", "11"]), 0)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), len(TIERS) * 4)
        self.assertEqual({r["difficulty"] for r in rows}, set(TIERS))


if __name__ == "__main__":
    unittest.main()
