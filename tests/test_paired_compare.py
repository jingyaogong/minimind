import json
import tempfile
import unittest
from pathlib import Path

from scripts.paired_compare import (
    any_sample_correct,
    binomial_two_sided,
    build_report,
    greedy_correct,
    load_eval,
    mcnemar,
    pair,
    render_markdown,
)


def make_record(sample_id, difficulty, greedy_ok, n_correct, samples=16, tokens=120):
    return {
        "sample_id": sample_id,
        "difficulty": difficulty,
        "gt": "1",
        "greedy": {"correct": greedy_ok, "format_ok": True, "tokens": tokens},
        "sampled_n_correct": n_correct,
        "sampled": [{"correct": index < n_correct, "format_ok": True, "tokens": tokens}
                    for index in range(samples)],
    }


def write_eval(run_dir: Path, records, **overrides):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows_evaluated": len(records),
        "greedy_accuracy": sum(1 for r in records if r["greedy"]["correct"]) / len(records),
        "sampled_pass_at_1": sum(r["sampled_n_correct"] for r in records) / (len(records) * 16),
        "pass_at_4": 0.5, "pass_at_8": 0.6, "pass_at_16": 0.7,
        "format_rate": 1.0, "no_answer_rate": 0.0, "mean_completion_tokens": 120.0,
        "samples": records,
    }
    payload.update(overrides)
    (run_dir / "eval.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


class McNemarTests(unittest.TestCase):
    def test_exact_p_values_match_the_hand_computed_reference(self):
        """这些数就是本项目真实结果：seed42 是 (14,5)，能力层是 (19,17)。

        如果这个测试挂了，说明 p 值的定义被改了——报告里的显著性结论会跟着失效。
        """

        for (b, c), expected in {(14, 5): 0.0636, (19, 17): 0.8679, (9, 0): 0.0039,
                                 (0, 0): 1.0, (30, 10): 0.0022}.items():
            self.assertAlmostEqual(round(binomial_two_sided(b, c), 4), expected, places=4,
                                   msg=f"b={b} c={c}")

    def test_p_value_is_symmetric_in_b_and_c(self):
        self.assertEqual(binomial_two_sided(12, 3), binomial_two_sided(3, 12))

    def test_no_change_is_never_significant(self):
        self.assertEqual(binomial_two_sided(0, 0), 1.0)


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "a": make_record("a", "easy", True, 4),
            "b": make_record("b", "easy", False, 0),
            "c": make_record("c", "medium", False, 2),
        }

    def test_counts_both_directions_and_leaves_other_questions_alone(self):
        arm = {
            "a": make_record("a", "easy", False, 4),   # 变错
            "b": make_record("b", "easy", True, 1),    # 变对
            "c": make_record("c", "medium", False, 2),  # 不变
        }
        result = mcnemar(self.baseline, arm, greedy_correct)
        self.assertEqual(result["paired"], 3)
        self.assertEqual(result["became_correct"], 1)
        self.assertEqual(result["broke"], 1)
        self.assertEqual(result["net"], 0)

    def test_capability_flag_uses_the_sampling_budget_not_the_greedy_pass(self):
        """同一对记录，两个旗标可以给出相反的结论——这正是报告要区分的那件事。"""

        baseline = {"c": make_record("c", "medium", greedy_ok=False, n_correct=0)}
        arm = {"c": make_record("c", "medium", greedy_ok=False, n_correct=3)}
        # 两次 greedy 都错，但 arm 在这题 16 次采样里对过 3 次 → 只有"能力"旗标能看到它
        self.assertEqual(mcnemar(baseline, arm, any_sample_correct)["net"], 1)
        self.assertEqual(mcnemar(baseline, arm, greedy_correct)["net"], 0)

    def test_unpaired_questions_are_reported_not_silently_dropped(self):
        arm = {"a": make_record("a", "easy", True, 4), "zz": make_record("zz", "hard", True, 9)}
        shared, only = pair(self.baseline, arm)
        self.assertEqual(shared, ["a"])
        self.assertEqual(only, ["b", "c", "zz"])

    def test_pair_raises_when_nothing_is_shared(self):
        with self.assertRaises(ValueError):
            pair({"a": make_record("a", "easy", True, 1)}, {"b": make_record("b", "easy", True, 1)})


class LoadTests(unittest.TestCase):
    def test_load_eval_rejects_a_payload_without_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "x"
            run_dir.mkdir()
            (run_dir / "eval.json").write_text(json.dumps({"greedy_accuracy": 1.0}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_eval(run_dir)

    def test_load_eval_reports_a_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_eval(Path(tmp) / "nope")


class ReportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.runs_dir = Path(self._tmp.name) / "runs"
        baseline = [make_record(f"{tier}-{i}", tier, greedy_ok=(i % 2 == 0), n_correct=i % 5)
                    for tier in ("easy", "medium", "hard") for i in range(6)]
        write_eval(self.runs_dir / "cold", baseline)
        # 3 题由错变对、1 题由对变错，另外把一题从能力集合里拿掉
        arm = [dict(record) for record in baseline]
        for index in (1, 3, 5):
            arm[index] = make_record(arm[index]["sample_id"], arm[index]["difficulty"], True, 3)
        arm[0] = make_record(arm[0]["sample_id"], arm[0]["difficulty"], True, 0)
        write_eval(self.runs_dir / "arm", arm)

    def tearDown(self):
        self._tmp.cleanup()

    def test_report_pairs_by_sample_id_and_reports_missing_ones(self):
        report = build_report(self.runs_dir, "cold", ["arm"])
        self.assertEqual(report["arms"][0]["greedy_vs_baseline"]["paired"], 18)
        self.assertEqual(report["arms"][0]["missing_from_pairing"], [])
        self.assertEqual(report["baseline_by_tier"]["easy"]["rows"], 6)

    def test_unpaired_ids_show_up_in_the_markdown(self):
        payload = json.loads((self.runs_dir / "arm" / "eval.json").read_text(encoding="utf-8"))
        payload["samples"] = payload["samples"][:-1]
        (self.runs_dir / "arm" / "eval.json").write_text(json.dumps(payload), encoding="utf-8")
        report = build_report(self.runs_dir, "cold", ["arm"])
        self.assertEqual(len(report["arms"][0]["missing_from_pairing"]), 1)
        self.assertIn("有题没能配对", render_markdown(report))

    def test_markdown_contains_every_arm_and_the_reading_guide(self):
        markdown = render_markdown(build_report(self.runs_dir, "cold", ["arm"]))
        self.assertIn("`arm`", markdown)
        self.assertIn("## 2. 配对检验", markdown)
        self.assertIn("分布锐化而非能力扩展", markdown)


if __name__ == "__main__":
    unittest.main()
