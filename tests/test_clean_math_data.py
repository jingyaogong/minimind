import json
import random
import tempfile
import unittest
from pathlib import Path

from scripts.clean_math_data import (
    AUDIT_REL,
    DIFFICULTIES,
    GENERATORS,
    HELDOUT_FILE,
    RLVR_FILE,
    SFT_FILE,
    canonical_key,
    main,
    normalize_question,
    plan_split,
    safe_eval,
    sha256_file,
)
from trainer.math_verifier import verdict


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class SafeEvalTests(unittest.TestCase):
    def test_whitelisted_arithmetic(self):
        cases = {
            "17*23": "391",
            "94-35": "59",
            "(17+23)*4-9": "151",
            "(4**3)+(20**2)-17": "447",
            "-(3+4)": "-7",
            "100/4": "25",
        }
        for expr, expected in cases.items():
            with self.subTest(expr=expr):
                self.assertEqual(safe_eval(expr), expected)

    def test_rejects_non_whitelisted_nodes(self):
        for expr in (
            "__import__('os').system('ls')",
            "open('x')",
            "1.5+2",
            "x+1",
            "'a'*3",
            "2**99",
            "10**9",
            "7/2",
            "1/0",
            "[1,2][0]",
            "True",
            "(lambda: 1)()",
        ):
            with self.subTest(expr=expr):
                with self.assertRaises(ValueError):
                    safe_eval(expr)

    def test_rejects_overlong_and_deep_expressions(self):
        with self.assertRaises(ValueError):
            safe_eval("1" * 200)
        with self.assertRaises(ValueError):
            safe_eval("+".join(["1"] * 100))


class GeneratorCrossCheckTests(unittest.TestCase):
    def test_generated_expressions_are_whitelisted_only(self):
        rng = random.Random(0)
        for difficulty in DIFFICULTIES:
            for _ in range(50):
                expr = GENERATORS[difficulty](rng)
                self.assertRegex(expr, r"^[0-9+*()\-]+$")

    def test_safe_eval_matches_cpython_arithmetic(self):
        # Independent implementation: CPython evaluates the same whitelisted
        # expression, so a traversal bug in safe_eval cannot hide here.
        rng = random.Random(1234)
        for difficulty in DIFFICULTIES:
            for _ in range(50):
                expr = GENERATORS[difficulty](rng)
                self.assertRegex(expr, r"^[0-9+*()\-]+$")
                self.assertEqual(safe_eval(expr), str(eval(expr)))  # noqa: S307


class AuditSheetTests(unittest.TestCase):
    def _build(self, tmp: str) -> Path:
        out = Path(tmp) / "dataset"
        self.assertEqual(
            main(["--synthetic-only", "--train-size", "30", "--heldout-size", "10",
                  "--audit-sample", "50", "--out-dir", str(out)]),
            0,
        )
        return out

    def test_sheet_lists_known_ids_across_both_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._build(tmp)
            rows = (out / AUDIT_REL).read_text(encoding="utf-8").splitlines()
            rows = [row for row in rows if row.strip()]
            self.assertEqual(rows[0], "sample_id\tdifficulty\tquestion\tgt\thuman_checked")
            self.assertEqual(len(rows) - 1, 40)
            known = {
                record["sample_id"]
                for name in (RLVR_FILE, HELDOUT_FILE)
                for record in read_jsonl(out / name)
            }
            for row in rows[1:]:
                fields = row.split("\t")
                self.assertEqual(len(fields), 5)
                self.assertIn(fields[0], known)
                self.assertEqual(verdict(f"答案是 {fields[3]}", fields[3])["status"], "correct")

    def test_sheet_is_reproducible_and_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self._build(Path(tmp) / "a")
            second = self._build(Path(tmp) / "b")
            self.assertEqual(
                (first / AUDIT_REL).read_bytes(), (second / AUDIT_REL).read_bytes()
            )
            out = Path(tmp) / "c" / "dataset"
            main(["--synthetic-only", "--train-size", "10", "--heldout-size", "4",
                  "--audit-sample", "0", "--out-dir", str(out)])
            self.assertFalse((out / AUDIT_REL).exists())

    def test_check_only_rejects_stale_audit_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._build(tmp)
            path = out / AUDIT_REL
            rows = [r for r in path.read_text(encoding="utf-8").splitlines() if r.strip()]
            rows.append("easy-deadbeefdeadbeef\teasy\tquestion\t1\t")
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            self.assertEqual(main(["--check-only", "--out-dir", str(out)]), 1)


class NormalizeQuestionTests(unittest.TestCase):
    def test_collapses_whitespace_and_appends_one_instruction(self):
        question = normalize_question("请  计算\n17*23")
        self.assertEqual(question.count("答案是"), 1)
        self.assertIn("17*23", question)
        self.assertEqual(normalize_question(question), question)


class PlanSplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        candidates = [
            {"key": f"k{i}", "difficulty": "easy"} for i in range(30)
        ] + [{"key": f"m{i}", "difficulty": "medium"} for i in range(20)]
        train_a, heldout_a = plan_split(candidates, 10, seed=42)
        train_b, heldout_b = plan_split(candidates, 10, seed=42)
        self.assertEqual([c["key"] for c in heldout_a], [c["key"] for c in heldout_b])
        self.assertEqual(len(heldout_a), 10)
        self.assertEqual(len(train_a), 40)
        self.assertFalse({c["key"] for c in train_a} & {c["key"] for c in heldout_a})

    def test_split_is_stratified_by_difficulty(self):
        candidates = [
            {"key": f"e{i}", "difficulty": "easy"} for i in range(50)
        ] + [{"key": f"h{i}", "difficulty": "hard"} for i in range(50)]
        _, heldout = plan_split(candidates, 10, seed=7)
        buckets = {c["difficulty"] for c in heldout}
        self.assertEqual(buckets, {"easy", "hard"})

    def test_rejects_impossible_heldout_target(self):
        with self.assertRaises(ValueError):
            plan_split([{"key": "a", "difficulty": "easy"}], 1, seed=0)


class SyntheticBuildTests(unittest.TestCase):
    def _build(self, directory: str, train: int = 30, heldout: int = 10) -> Path:
        out = Path(directory) / "dataset"
        code = main(
            [
                "--synthetic-only",
                "--train-size",
                str(train),
                "--heldout-size",
                str(heldout),
                "--seed",
                "42",
                "--out-dir",
                str(out),
            ]
        )
        self.assertEqual(code, 0)
        return out

    def test_builds_three_files_with_exact_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._build(tmp)
            sft = read_jsonl(out / SFT_FILE)
            rlvr = read_jsonl(out / RLVR_FILE)
            heldout = read_jsonl(out / HELDOUT_FILE)
            self.assertEqual((len(sft), len(rlvr), len(heldout)), (30, 30, 10))
            manifest = json.loads((out / "manifests" / "math_manifest.json").read_text())
            self.assertEqual(manifest["overlap_count"], 0)
            self.assertEqual(manifest["train_count"], 30)
            self.assertEqual(sum(manifest["difficulty_counts"]["heldout"].values()), 10)

    def test_gold_answers_verify_as_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._build(tmp)
            for record in read_jsonl(out / SFT_FILE):
                conversations = record["conversations"]
                assistant = next(m for m in conversations if m["role"] == "assistant")
                gt = assistant["content"].split()[-1]
                response = f"<think>{assistant['reasoning_content']}</think>\n{assistant['content']}"
                result = verdict(response, gt)
                self.assertEqual(result["status"], "correct", record)

    def test_rlvr_prompts_carry_gt_and_no_assistant(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._build(tmp)
            for path in (out / RLVR_FILE, out / HELDOUT_FILE):
                for record in read_jsonl(path):
                    self.assertEqual(len(record["conversations"]), 1)
                    self.assertEqual(record["conversations"][0]["role"], "user")
                    self.assertEqual(
                        verdict("答案是 " + record["gt"], record["gt"])["status"], "correct"
                    )

    def test_sample_ids_are_unique_across_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._build(tmp)
            ids = [
                record["sample_id"]
                for name in (RLVR_FILE, HELDOUT_FILE)
                for record in read_jsonl(out / name)
            ]
            self.assertEqual(len(ids), len(set(ids)))

    def test_same_seed_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self._build(Path(tmp) / "a")
            second = self._build(Path(tmp) / "b")
            for name in (SFT_FILE, RLVR_FILE, HELDOUT_FILE):
                self.assertEqual(sha256_file(first / name), sha256_file(second / name))

    def test_different_seed_changes_the_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self._build(Path(tmp) / "a")
            out = Path(tmp) / "b" / "dataset"
            code = main(
                [
                    "--synthetic-only", "--train-size", "30", "--heldout-size", "10",
                    "--seed", "7", "--out-dir", str(out),
                ]
            )
            self.assertEqual(code, 0)
            self.assertNotEqual(
                sha256_file(first / HELDOUT_FILE), sha256_file(out / HELDOUT_FILE)
            )


class RawInputTests(unittest.TestCase):
    def _write_raw(self, path: Path) -> None:
        rows = [
            # usable single-turn row
            {"conversations": [{"role": "user", "content": "请计算 12+30，并在最后一行以‘答案是 数字’输出。"}],
             "gt": "42", "source": "unit", "difficulty": "easy"},
            # duplicate of the first row (only punctuation/whitespace differ)
            {"conversations": [{"role": "user", "content": "请计算  12+30，并在最后一行以‘答案是 数字’输出。 "}],
             "gt": "42.0", "source": "unit", "difficulty": "easy"},
            # numeric gt of a non-string type
            {"messages": [{"role": "user", "content": "请计算 5*5，并在最后一行以‘答案是 数字’输出。"}],
             "ground_truth": 25, "difficulty": "medium"},
            # multi-turn: the last user turn wins
            {"conversations": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好，有什么可以帮你？"},
                {"role": "user", "content": "请计算 7+8，并在最后一行以‘答案是 数字’输出。"}],
             "gt": "15", "difficulty": "easy"},
            # tool trajectory -> discarded
            {"conversations": [{"role": "user", "content": "请计算 1+1，并在最后一行以‘答案是 数字’输出。"}],
             "gt": "2", "tools": "[{\"name\": \"calc\"}]"},
            {"conversations": [{"role": "user", "content": "请计算 1+1。"},
                               {"role": "tool", "content": "2"}], "gt": "2"},
            # missing / unparsable gt -> discarded
            {"conversations": [{"role": "user", "content": "请计算 3+4。"}]},
            {"conversations": [{"role": "user", "content": "请计算 3+4。"}], "gt": "答案是 7"},
            # no user message -> discarded
            {"conversations": [{"role": "assistant", "content": "答案是 1"}], "gt": "1"},
        ]
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.write("{not json}\n")

    def test_cleaning_discards_and_dedupes_with_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.jsonl"
            self._write_raw(raw)
            out = Path(tmp) / "dataset"
            code = main(
                [
                    "--input", str(raw), "--heldout-size", "1", "--train-size", "0",
                    "--out-dir", str(out),
                ]
            )
            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifests" / "math_manifest.json").read_text())
            self.assertEqual(manifest["train_count"] + manifest["heldout_count"], 3)
            self.assertEqual(manifest["duplicate_count"], 1)
            self.assertEqual(manifest["bad_line_numbers"], [10])
            reasons = manifest["discard_reasons"]
            self.assertEqual(reasons["tool_trajectory"], 2)
            self.assertEqual(reasons["missing_gt"], 1)
            self.assertEqual(reasons["unparsable_gt"], 1)
            self.assertEqual(reasons["missing_user"], 1)
            self.assertEqual(manifest["overlap_count"], 0)

    def test_multi_turn_row_is_collapsed_to_the_last_user_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.jsonl"
            self._write_raw(raw)
            out = Path(tmp) / "dataset"
            main(["--input", str(raw), "--heldout-size", "1", "--out-dir", str(out)])
            keys = {canonical_key(r["conversations"][0]["content"]) for r in read_jsonl(out / RLVR_FILE)}
            for record in read_jsonl(out / HELDOUT_FILE):
                keys.add(canonical_key(record["conversations"][0]["content"]))
            self.assertTrue(any("请计算7+8" in key for key in keys))
            self.assertFalse(any("你好" in key for key in keys))


class CheckOnlyTests(unittest.TestCase):
    def _build(self, directory: str) -> Path:
        out = Path(directory) / "dataset"
        self.assertEqual(
            main(["--synthetic-only", "--train-size", "20", "--heldout-size", "8",
                  "--out-dir", str(out)]),
            0,
        )
        return out

    def test_check_only_passes_on_fresh_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._build(tmp)
            self.assertEqual(main(["--check-only", "--out-dir", str(out)]), 0)

    def test_check_only_detects_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._build(tmp)
            train = read_jsonl(out / RLVR_FILE)
            heldout = read_jsonl(out / HELDOUT_FILE)
            heldout[0] = train[0]
            with (out / HELDOUT_FILE).open("w", encoding="utf-8") as handle:
                for record in heldout:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.assertEqual(main(["--check-only", "--out-dir", str(out)]), 1)

    def test_check_only_detects_schema_damage(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._build(tmp)
            sft = read_jsonl(out / SFT_FILE)
            del sft[0]["conversations"][0]["tools"]
            with (out / SFT_FILE).open("w", encoding="utf-8") as handle:
                for record in sft:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.assertEqual(main(["--check-only", "--out-dir", str(out)]), 1)

    def test_check_only_detects_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._build(tmp)
            manifest_path = out / "manifests" / "math_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["train_count"] = 999
            manifest_path.write_text(json.dumps(manifest))
            self.assertEqual(main(["--check-only", "--out-dir", str(out)]), 1)


class CliContractTests(unittest.TestCase):
    def test_requires_exactly_one_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(["--out-dir", tmp]), 2)
            self.assertEqual(
                main(["--out-dir", tmp, "--synthetic-only", "--input", "x.jsonl"]), 2
            )

    def test_missing_input_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                main(["--input", str(Path(tmp) / "nope.jsonl"), "--out-dir", tmp]), 2
            )


if __name__ == "__main__":
    unittest.main()
