import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.eval_heldout import (
    completion_length,
    data_provenance,
    main,
    parse_args,
    pass_at_k,
    read_rows,
    score_completion,
    summarize,
    summarize_group,
)
from trainer.math_verifier import ANSWER_MARKER


def item(correct, status, tokens=10, format_ok=True, marker_hit=True):
    return {
        "text": "x",
        "status": status,
        "extracted": None,
        "correct": correct,
        "think_closed": format_ok,
        "marker_hit": marker_hit,
        "format_ok": format_ok,
        "tokens": tokens,
    }


def row(difficulty, greedy, sampled):
    return {
        "sample_id": f"{difficulty}-1",
        "difficulty": difficulty,
        "gt": "1",
        "prompt": "p",
        "greedy": greedy,
        "sampled": sampled,
        "sampled_n_correct": sum(1 for value in sampled if value["correct"]),
    }


class PassAtKTests(unittest.TestCase):
    def test_hand_computed_values(self):
        self.assertAlmostEqual(pass_at_k(16, 1, 1), 1 - 15 / 16)
        self.assertAlmostEqual(pass_at_k(4, 2, 2), 1 - 1 / 6)
        self.assertAlmostEqual(pass_at_k(16, 2, 4), 1 - 1001 / 1820)

    def test_bounds(self):
        self.assertEqual(pass_at_k(16, 0, 1), 0.0)
        self.assertEqual(pass_at_k(16, 16, 16), 1.0)
        self.assertEqual(pass_at_k(3, 1, 3), 1.0)

    def test_insufficient_budget_is_none_not_zero(self):
        self.assertIsNone(pass_at_k(2, 1, 16))
        self.assertIsNone(pass_at_k(8, 8, 16))

    def test_is_monotone_in_k(self):
        values = [pass_at_k(16, 4, k) for k in (1, 2, 4, 8, 16)]
        self.assertEqual(values, sorted(values))


class SummarizeTests(unittest.TestCase):
    def build_rows(self):
        return [
            row("easy", item(True, "correct"), [item(True, "correct"), item(False, "incorrect")]),
            row("hard", item(False, "no_answer"), [item(False, "no_answer"), item(False, "no_answer")]),
        ]

    def test_overall_metrics(self):
        summary = summarize(self.build_rows(), [1, 2, 16])
        self.assertAlmostEqual(summary["greedy_accuracy"], 0.5)
        self.assertAlmostEqual(summary["sampled_pass_at_1"], 0.25)
        self.assertAlmostEqual(summary["pass_at_2"], 0.5)
        self.assertIsNone(summary["pass_at_16"])
        self.assertAlmostEqual(summary["no_answer_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_completion_tokens"], 10.0)
        self.assertEqual(summary["sample_budget"], 2)
        self.assertEqual(summary["status_counts"]["correct"], 1)
        self.assertEqual(summary["status_counts"]["no_answer"], 2)
        self.assertEqual(summary["status_counts"]["invalid_gt"], 0)

    def test_greedy_and_sampled_are_reported_separately(self):
        rows = [row("easy", item(True, "correct"), [item(False, "incorrect")] * 4)]
        summary = summarize(rows, [1])
        self.assertAlmostEqual(summary["greedy_accuracy"], 1.0)
        self.assertAlmostEqual(summary["sampled_pass_at_1"], 0.0)

    def test_by_difficulty_breakdown(self):
        summary = summarize(self.build_rows(), [1])
        self.assertEqual(set(summary["by_difficulty"]), {"easy", "hard"})
        self.assertAlmostEqual(summary["by_difficulty"]["easy"]["greedy_accuracy"], 1.0)
        self.assertAlmostEqual(summary["by_difficulty"]["hard"]["greedy_accuracy"], 0.0)
        self.assertEqual(summary["by_difficulty"]["easy"]["rows"], 1)

    def test_empty_group_does_not_crash(self):
        summary = summarize_group([], [1, 16])
        self.assertEqual(summary["rows"], 0)
        self.assertEqual(summary["greedy_accuracy"], 0.0)
        self.assertIsNone(summary["pass_at_16"])


class CompletionLengthTests(unittest.TestCase):
    def test_stops_at_first_eos(self):
        self.assertEqual(completion_length(torch.tensor([5, 6, 2, 2, 2]), 2), 3)
        self.assertEqual(completion_length(torch.tensor([5, 6]), 2), 2)
        self.assertEqual(completion_length(torch.tensor([2]), 2), 1)
        self.assertEqual(completion_length(torch.tensor([]), 2), 0)

    def test_no_eos_id_disables_the_scan(self):
        self.assertEqual(completion_length(torch.tensor([5, 6]), None), 2)


class ScoreCompletionTests(unittest.TestCase):
    def test_uses_the_shared_verifier(self):
        good = score_completion(f"<think>算一下</think>\n{ANSWER_MARKER} 391", "391", ANSWER_MARKER, 12)
        self.assertTrue(good["correct"])
        self.assertEqual(good["status"], "correct")
        self.assertTrue(good["format_ok"])
        self.assertEqual(good["extracted"], "391")

        bad = score_completion("我算不出来。", "391", ANSWER_MARKER, 5)
        self.assertEqual(bad["status"], "no_answer")
        self.assertFalse(bad["format_ok"])

    def test_answer_only_after_the_last_marker_wins(self):
        result = score_completion(
            f"<think>{ANSWER_MARKER} 1</think>\n{ANSWER_MARKER} 391", "391", ANSWER_MARKER, 9
        )
        self.assertTrue(result["correct"])
        self.assertEqual(result["extracted"], "391")


class ReadRowsTests(unittest.TestCase):
    def write(self, tmp, rows):
        path = Path(tmp) / "heldout.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        return path

    def test_rejects_assistant_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, [{
                "conversations": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ], "gt": "1",
            }])
            with self.assertRaises(ValueError):
                read_rows(path)

    def test_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_rows(path)

    def test_reads_user_only_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, [{"conversations": [{"role": "user", "content": "q"}], "gt": "1"}])
            self.assertEqual(len(read_rows(path)), 1)


class ProvenanceTests(unittest.TestCase):
    def test_repository_dataset_matches_its_manifest(self):
        root = Path(__file__).resolve().parents[1]
        data = root / "dataset" / "math_heldout.jsonl"
        manifest = root / "dataset" / "manifests" / "math_manifest.json"
        info = data_provenance(data, manifest)
        self.assertTrue(info["data_matches_manifest"])
        self.assertEqual(len(info["data_manifest_sha256"]), 64)
        self.assertEqual(len(info["verifier_sha256"]), 64)

    def test_tampered_data_is_detected(self):
        root = Path(__file__).resolve().parents[1]
        manifest = root / "dataset" / "manifests" / "math_manifest.json"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "math_heldout.jsonl"
            path.write_text('{"conversations":[{"role":"user","content":"q"}],"gt":"1"}\n', encoding="utf-8")
            info = data_provenance(path, manifest)
            self.assertFalse(info["data_matches_manifest"])

    def test_missing_manifest_is_not_an_error(self):
        root = Path(__file__).resolve().parents[1]
        info = data_provenance(root / "dataset" / "math_heldout.jsonl", Path("/nonexistent/manifest.json"))
        self.assertIsNone(info["data_manifest_sha256"])
        self.assertIsNone(info["data_matches_manifest"])


class CliTests(unittest.TestCase):
    def test_defaults_are_recorded_in_the_payload_contract(self):
        args = parse_args(["--model_path", "x.pth"])
        self.assertEqual(args.num_samples, 16)
        self.assertEqual(args.marker, ANSWER_MARKER)
        self.assertEqual(args.max_new_tokens, 512)
        self.assertEqual(args.temperature, 0.8)
        self.assertEqual(args.top_p, 0.85)
        self.assertEqual(args.top_k, 50)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.hidden_size, 768)
        self.assertEqual(args.num_hidden_layers, 8)
        self.assertEqual(args.limit, 0)

    def test_model_path_is_required(self):
        with self.assertRaises(SystemExit):
            parse_args([])

    def test_rejects_zero_samples_before_touching_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                main(["--model_path", "x.pth", "--num_samples", "0", "--output", str(Path(tmp) / "o.json")]),
                2,
            )


if __name__ == "__main__":
    unittest.main()
