import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.compare_runs import load_metrics, main

FIELDS = ["timestamp", "step", "seed", "avg_reward", "correct_rate",
          "completion_tokens", "valid_groups", "adv_std"]


def write_run(log_dir: Path, tag: str, rows) -> None:
    with (log_dir / f"{tag}_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for step, reward, tokens, groups in rows:
            writer.writerow({
                "timestamp": f"2026-09-27T10:00:{step % 60:02d}+00:00",
                "step": step, "seed": 42, "avg_reward": reward, "correct_rate": 0.5,
                "completion_tokens": tokens, "valid_groups": groups, "adv_std": 1.0,
            })


class CompareRunsTests(unittest.TestCase):
    def test_load_metrics_ignores_timestamps_and_counts_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            write_run(log_dir, "a", [(5, 0.5, 100, "1/2")])
            rows, total, dups = load_metrics(log_dir / "a_metrics.csv")
        self.assertEqual((total, dups), (1, 0))
        self.assertNotIn("timestamp", rows["5"])

    def test_identical_runs_report_full_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            rows = [(5, 0.5, 100, "1/2"), (10, 0.6, 110, "2/2")]
            write_run(log_dir, "a", rows)
            write_run(log_dir, "b", rows)
            with redirect_stdout(io.StringIO()) as out:
                code = main(["--a", "a", "--b", "b", "--log-dir", str(log_dir)])
        self.assertEqual(code, 0)
        self.assertIn("逐字段完全一致", out.getvalue())
        self.assertIn("2 / 2", out.getvalue())

    def test_differences_report_first_step_and_magnitude(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            write_run(log_dir, "a", [(5, 0.5, 100, "1/2"), (10, 0.5, 100, "1/2")])
            write_run(log_dir, "b", [(5, 0.5, 100, "1/2"), (10, 1.5, 130, "0/2")])
            with redirect_stdout(io.StringIO()) as out:
                code = main(["--a", "a", "--b", "b", "--log-dir", str(log_dir), "--expect-identical"])
        self.assertEqual(code, 1)
        text = out.getvalue()
        self.assertIn("第一处差异出现在 step 10", text)
        self.assertIn("avg_reward", text)
        self.assertIn("valid_groups", text)   # 非数值字段单独统计

    def test_missing_metrics_file_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            write_run(log_dir, "a", [(5, 0.5, 100, "1/2")])
            with redirect_stdout(io.StringIO()):
                code = main(["--a", "a", "--b", "nope", "--log-dir", str(log_dir)])
        self.assertEqual(code, 2)

    def test_disjoint_steps_fail_instead_of_pretending_to_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            write_run(log_dir, "a", [(5, 0.5, 100, "1/2")])
            write_run(log_dir, "b", [(7, 0.5, 100, "1/2")])
            with redirect_stdout(io.StringIO()):
                code = main(["--a", "a", "--b", "b", "--log-dir", str(log_dir)])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
