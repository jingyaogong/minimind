import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scripts.error_analysis as error_analysis

from scripts.error_analysis import (
    collect_flags,
    is_correct,
    load_rows,
    main,
    max_consecutive_repeats,
    primary_category,
    render_markdown,
    resolve_input,
)

MAX_NEW = 512


def row(text, status="incorrect", tokens=50, **extra):
    record = {"text": text, "status": status, "tokens": tokens, "difficulty": "easy",
              "gt": "19", "extracted": None, "think_closed": None, "mode": "greedy"}
    record.update(extra)
    return record


class FlagTests(unittest.TestCase):
    def test_clean_wrong_answer_is_an_arithmetic_error(self):
        flags = collect_flags(row("<think>\n1+1\n</think>\n\n答案是 18"), max_new_tokens=MAX_NEW)
        self.assertEqual(primary_category(flags), "arithmetic_error")

    def test_correct_row_wins_over_other_flags(self):
        flags = collect_flags(row("<think>\nx\n</think>\n答案是 19", status="correct", correct=True),
                              max_new_tokens=MAX_NEW)
        self.assertEqual(primary_category(flags), "correct")
        self.assertTrue(is_correct({"status": "correct"}))

    def test_unclosed_think_is_format_failure(self):
        flags = collect_flags(row("<think>\n算到一半就断了"), max_new_tokens=MAX_NEW)
        self.assertEqual(primary_category(flags), "unclosed_think")

    def test_prompt_opened_think_is_not_a_format_failure(self):
        # prompt 末尾自带 "<think>\n"（open_thinking=True），正常补全只剩 "</think>"，
        # 绝不能把"没有开标签"当成模板没被遵循
        flags = collect_flags(row("32-27：结果是 10。\n</think>\n\n答案是 10"), max_new_tokens=MAX_NEW)
        self.assertNotIn("unclosed_think", flags)
        self.assertEqual(primary_category(flags), "arithmetic_error")

    def test_no_closing_tag_is_unclosed_think(self):
        flags = collect_flags(row("算到一半就断了，还没给出结果"), max_new_tokens=MAX_NEW)
        self.assertIn("unclosed_think", flags)
        self.assertEqual(primary_category(flags), "unclosed_think")

    def test_verifier_field_overrides_text_sniffing(self):
        flags = collect_flags(row("</think>\n\n答案是 18", think_closed=False), max_new_tokens=MAX_NEW)
        self.assertIn("unclosed_think", flags)

    def test_marker_without_parseable_number_is_flagged_for_review(self):
        flags = collect_flags(row("<think>\nx\n</think>\n\n答案是 一千", status="no_answer"),
                              max_new_tokens=MAX_NEW)
        self.assertIn("marker_unparsed", flags)
        self.assertEqual(primary_category(flags), "no_answer")

    def test_marker_inside_unclosed_think_is_not_marker_unparsed(self):
        # 带 <think> 且未闭合时 verifier 的 final 段为空，标记出现在推理里不算"抽取失败"
        flags = collect_flags(row("<think>\n答案是 一千 但我还在想", status="no_answer"), max_new_tokens=MAX_NEW)
        self.assertIn("unclosed_think", flags)
        self.assertNotIn("marker_unparsed", flags)

    def test_degenerate_loop_is_repetition(self):
        text = "<think>\n" + "97乘61可以分解为97乘以61这样可以简化计算。" * 5 + "\n</think>\n\n答案是 1"
        flags = collect_flags(row(text), max_new_tokens=MAX_NEW)
        self.assertIn("repetition", flags)
        self.assertEqual(primary_category(flags), "repetition")

    def test_redundant_answer_markers(self):
        flags = collect_flags(row("<think>\nx\n</think>\n答案是 18\n答案是 19"), max_new_tokens=MAX_NEW)
        self.assertIn("redundant_answer", flags)
        self.assertEqual(primary_category(flags), "redundant_answer")

    def test_truncated_generation_is_length_burn(self):
        flags = collect_flags(row("<think>\n" + "。".join(["继续算"] * 40), tokens=int(0.96 * MAX_NEW)),
                              max_new_tokens=MAX_NEW)
        self.assertIn("length_burn", flags)

    def test_flags_are_not_exclusive(self):
        flags = collect_flags(row("<think>\nx\n</think>\n答案是 18\n答案是 19", status="incorrect"),
                              max_new_tokens=MAX_NEW)
        self.assertTrue({"redundant_answer", "arithmetic_error"} <= flags)


class RepetitionTests(unittest.TestCase):
    def test_whitespace_is_ignored_and_normal_text_is_not_repetition(self):
        self.assertEqual(max_consecutive_repeats("甲乙丙丁戊己庚辛壬癸" * 4), 4)
        plain = "我需要先算 32-27，然后核对一遍，最后给出答案。"
        self.assertLess(max_consecutive_repeats(plain), 3)


class RenderTests(unittest.TestCase):
    def test_render_contains_tables_and_examples(self):
        args = type("A", (), {"mode": "greedy", "difficulty": None, "max_new_tokens": MAX_NEW,
                              "examples_per_category": 1, "example_chars": 50})()
        rows = [row("<think>\nx\n</think>\n答案是 18"), row("<think>\nx\n</think>\n答案是 19",
                                                          status="correct", correct=True)]
        markdown = render_markdown("tag", Path("runs/tag/eval.jsonl"), rows, args)
        self.assertIn("## 1. 主类别", markdown)
        self.assertIn("| `arithmetic_error` | 1 | 50.0% |", markdown)
        self.assertIn("## 3. 难度 × 主类别", markdown)
        self.assertIn("```text", markdown)
        self.assertNotIn("### correct", markdown)   # 正确样本不进人工审计抽样

    def test_training_samples_without_mode_column_still_render(self):
        args = type("A", (), {"mode": "all", "difficulty": None, "max_new_tokens": MAX_NEW,
                              "examples_per_category": 0, "example_chars": 50})()
        record = {"completion": "<think>\nx\n</think>\n答案是 18", "completion_tokens": 40,
                  "status": "incorrect", "difficulty": "hard"}
        rows = [record]
        markdown = render_markdown("t", Path("logs/t_samples.jsonl"), rows, args)
        self.assertIn("| `arithmetic_error` | 1 |", markdown)
        self.assertNotIn("| n/a |", markdown)

    def test_zero_count_categories_are_listed_explicitly(self):
        # DoD 要求"错误分类至少包括 …"，所以 0 命中的类别也要写出来，证明确实检查过
        args = type("A", (), {"mode": "greedy", "difficulty": None, "max_new_tokens": MAX_NEW,
                              "examples_per_category": 0, "example_chars": 50})()
        rows = [row("<think>\nx\n</think>\n答案是 19", status="correct", correct=True)]
        markdown = render_markdown("t", Path("runs/t/eval.jsonl"), rows, args)
        self.assertIn("未命中（各 0 条，口径已检查）：`repetition`", markdown)
        self.assertIn("`arithmetic_error`", markdown)


class CliTests(unittest.TestCase):
    def test_resolve_prefers_eval_jsonl_then_falls_back_to_training_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runs" / "tag").mkdir(parents=True)
            (root / "logs").mkdir(parents=True)
            with mock.patch.object(error_analysis, "DEFAULT_RUNS_DIR", root / "runs"), \
                 mock.patch.object(error_analysis, "DEFAULT_LOG_DIR", root / "logs"):
                (root / "runs" / "tag" / "eval.jsonl").write_text("", encoding="utf-8")
                path, tag = resolve_input("tag", None)
                self.assertEqual((path, tag), (root / "runs" / "tag" / "eval.jsonl", "tag"))

                (root / "runs" / "tag" / "eval.jsonl").unlink()
                (root / "logs" / "tag_samples.jsonl").write_text("", encoding="utf-8")
                path, tag = resolve_input("tag", None)
                self.assertEqual((path, tag), (root / "logs" / "tag_samples.jsonl", "tag"))

                with self.assertRaises(FileNotFoundError):
                    resolve_input("missing", None)

    def test_main_writes_report_and_rejects_empty_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runs" / "tag").mkdir(parents=True)
            target = root / "runs" / "tag" / "eval.jsonl"
            with target.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(row("<think>\nx\n</think>\n答案是 18")) + "\n")
                handle.write(json.dumps(row("<think>\nx\n</think>\n答案是 19",
                                            status="correct", correct=True, mode="sampled")) + "\n")
            out = root / "report.md"
            code = main(["--path", str(target), "--mode", "greedy", "--output", str(out)])
            self.assertEqual(code, 0)
            text = out.read_text(encoding="utf-8")
            self.assertIn("共 1 条", text)
            self.assertEqual(load_rows(target)[0]["gt"], "19")
            self.assertEqual(main(["--path", str(target), "--difficulty", "hard"]), 2)


if __name__ == "__main__":
    unittest.main()
