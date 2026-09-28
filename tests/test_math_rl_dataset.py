import json
import tempfile
import unittest
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.math_rl_dataset import MathRLDataset
from scripts.clean_math_data import main as clean_main

TOKENIZER = None


def tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(Path(__file__).resolve().parents[1] / "model")
    return TOKENIZER


def write_rows(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def row(question: str, gt, **extra) -> dict:
    payload = {
        "conversations": [{"role": "user", "content": question}],
        "gt": gt,
        "source": "unit",
        "difficulty": "easy",
        "sample_id": f"unit-{abs(hash(question)) % 10**8}",
    }
    payload.update(extra)
    return payload


QUESTION = "请计算 43+38，并在最后一行以‘答案是 数字’输出。"


class MathRLDatasetTests(unittest.TestCase):
    def build(self, tmp: str, rows: list[dict], **kwargs) -> MathRLDataset:
        path = write_rows(Path(tmp) / "rlvr.jsonl", rows)
        return MathRLDataset(path, tokenizer(), **kwargs)

    def test_item_exposes_prompt_gt_difficulty_and_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self.build(tmp, [row(QUESTION, "81")])
            item = dataset[0]
            self.assertEqual(sorted(item), ["difficulty", "gt", "prompt", "sample_id"])
            self.assertEqual(item["gt"], "81")
            self.assertEqual(item["difficulty"], "easy")
            self.assertIn("请计算 43+38", item["prompt"])
            self.assertTrue(item["prompt"].endswith("<think>\n"))

    def test_decimal_gt_is_normalised(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self.build(tmp, [row(QUESTION, "81.0"), row(QUESTION, 81)])
            self.assertEqual([dataset[i]["gt"] for i in range(2)], ["81", "81"])

    def test_dataloader_collates_prompt_and_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self.build(tmp, [row(QUESTION, "81"), row(QUESTION.replace("43+38", "1+1"), "2")])
            batch = next(iter(DataLoader(dataset, batch_size=2)))
            self.assertEqual(len(batch["prompt"]), 2)
            self.assertEqual(batch["gt"], ["81", "2"])
            self.assertTrue(all(isinstance(p, str) for p in batch["prompt"]))

    def test_thinking_ratio_controls_the_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            on = self.build(tmp, [row(QUESTION, "81")], thinking_ratio=1.0)
            self.assertTrue(on.render_prompt(0).endswith("<think>\n"))
            self.assertNotIn("</think>", on.render_prompt(0))
            off = self.build(tmp, [row(QUESTION, "81")], thinking_ratio=0.0)
            self.assertTrue(off.render_prompt(0).endswith("<think>\n\n</think>\n\n"))

    def test_audit_lengths_reports_truncation_risk(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self.build(
                tmp, [row(QUESTION, "81"), row(QUESTION.replace("43+38", "1+1"), "2")],
                max_length=8,
            )
            audit = dataset.audit_lengths()
            self.assertEqual(audit["rows"], 2)
            for label in ("thinking_on", "thinking_off"):
                self.assertEqual(audit[label]["overlong"], 2)
                self.assertEqual(audit[label]["overlong_fraction"], 1.0)
                self.assertEqual(audit[label]["budget_tokens"], 8)
            self.assertGreater(audit["thinking_off"]["p50_tokens"], 0)
            self.assertLess(audit["thinking_on"]["p50_tokens"], 10_000)

    def test_audit_lengths_can_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [row(QUESTION, "81"), row(QUESTION.replace("43+38", "1+1"), "2")]
            dataset = self.build(tmp, rows)
            self.assertEqual(dataset.audit_lengths(sample_size=1)["rows"], 1)

    def test_difficulty_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                row(QUESTION, "81", difficulty="easy"),
                row(QUESTION.replace("43+38", "1+1"), "2", difficulty="hard"),
                row(QUESTION.replace("43+38", "2+2"), "4", difficulty="hard"),
            ]
            dataset = self.build(tmp, rows)
            self.assertEqual(dataset.difficulty_counts(), {"easy": 1, "hard": 2})


class RejectionTests(unittest.TestCase):
    def build(self, tmp: str, rows: list[dict], **kwargs):
        path = write_rows(Path(tmp) / "rlvr.jsonl", rows)
        return MathRLDataset(path, tokenizer(), **kwargs)

    def test_rejects_sft_style_rows_with_a_pointer_to_the_right_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                {
                    "conversations": [
                        {"role": "user", "content": QUESTION, "reasoning_content": "", "tools": "", "tool_calls": ""},
                        {"role": "assistant", "content": "答案是 81", "reasoning_content": "算一下", "tools": "", "tool_calls": ""},
                    ]
                }
            ]
            with self.assertRaises(ValueError) as ctx:
                self.build(tmp, rows)
            message = str(ctx.exception)
            self.assertIn("user-only", message)
            self.assertIn("clean_math_data.py", message)

    def test_rejects_missing_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                self.build(tmp, [{"conversations": [{"role": "user", "content": QUESTION}]}])
            self.assertIn("missing gt", str(ctx.exception))

    def test_rejects_non_numeric_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                self.build(tmp, [row(QUESTION, "答案是 81")])
            self.assertIn("not a normalisable number", str(ctx.exception))

    def test_rejects_marker_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                self.build(tmp, [row("请计算 43+38，最后输出答案。", "81")])
            self.assertIn("marker", str(ctx.exception))

    def test_rejects_multiple_user_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            conversations = [
                {"role": "user", "content": "你好"},
                {"role": "user", "content": QUESTION},
            ]
            with self.assertRaises(ValueError) as ctx:
                self.build(tmp, [{"conversations": conversations, "gt": "81"}])
            self.assertIn("exactly one user message", str(ctx.exception))

    def test_rejects_empty_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = write_rows(Path(tmp) / "empty.jsonl", [])
            with self.assertRaises(ValueError):
                MathRLDataset(empty, tokenizer())
            with self.assertRaises(FileNotFoundError):
                MathRLDataset(Path(tmp) / "nope.jsonl", tokenizer())

    def test_rejects_bad_constructor_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.build(tmp, [row(QUESTION, "81")], thinking_ratio=1.5)
            with self.assertRaises(ValueError):
                self.build(tmp, [row(QUESTION, "81")], max_length=0)


class CleanerIntegrationTests(unittest.TestCase):
    def test_loads_the_file_produced_by_the_cleaner(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dataset"
            self.assertEqual(
                clean_main(["--synthetic-only", "--train-size", "12", "--heldout-size", "4",
                            "--out-dir", str(out)]),
                0,
            )
            dataset = MathRLDataset(out / "math_rlvr_train.jsonl", tokenizer(), max_length=768)
            self.assertEqual(len(dataset), 12)
            self.assertEqual(sum(dataset.difficulty_counts().values()), 12)
            audit = dataset.audit_lengths()
            self.assertEqual(audit["thinking_on"]["overlong"], 0)
            for index in range(len(dataset)):
                item = dataset[index]
                self.assertTrue(item["sample_id"])
                self.assertTrue(item["gt"].lstrip("-").isdigit())


if __name__ == "__main__":
    unittest.main()
