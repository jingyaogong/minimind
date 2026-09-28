import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.experiment_registry import is_baseline_row  # noqa: E402
from scripts.archive_run import (
    REQUIRED_FILES,
    archive_run,
    command_lead_line,
    changed_columns,
    check_archive,
    checkpoints_index,
    collect_environment,
    data_manifest_payload,
    discover_tags,
    eval_digest,
    main,
    metrics_digest,
    reference_row,
    registry_row,
    render_checkpoints,
    render_config_yaml,
    sample_status_counts,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "configs" / "experiment_registry.csv"
DATASET = REPO_ROOT / "dataset"
TAG = "mini_rule_seed42"

METRIC_HEADER = [
    "timestamp", "step", "seed", "avg_reward", "format_rate", "correct_rate", "no_answer_rate",
    "avg_completion_tokens", "median_completion_tokens", "kl_ref", "entropy", "adv_std",
    "policy_loss", "valid_groups", "all_correct_groups", "all_wrong_groups", "filtered_fraction",
]


def write_metrics(path: Path, rows: int = 2, steps=(5, 10)) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_HEADER)
        writer.writeheader()
        for index in range(rows):
            writer.writerow({
                "timestamp": f"2026-09-24T0{index}:00:00+00:00",
                "step": steps[index],
                "seed": 42,
                "avg_reward": 0.1 * (index + 1),
                "format_rate": 0.4,
                "correct_rate": 0.2,
                "no_answer_rate": 0.3,
                "avg_completion_tokens": 100 + index,
                "median_completion_tokens": 96,
                "kl_ref": -1.5,
                "entropy": 8.5,
                "adv_std": 0.01,
                "policy_loss": 0.05,
                "valid_groups": 3,
                "all_correct_groups": 1,
                "all_wrong_groups": 2,
                "filtered_fraction": 0.5,
            })


def write_samples(path: Path) -> None:
    records = [
        {"step": 5, "completions": [{"status": "correct"}, {"status": "no_answer"}]},
        {"step": 5, "completions": [{"status": "incorrect"}]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def write_config(path: Path) -> None:
    path.write_text(json.dumps({"created_at": "now", "args": {"seed": 42}}), encoding="utf-8")


class HashTests(unittest.TestCase):
    def test_sha256_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.bin"
            path.write_bytes(b"minimind" * 1000)
            self.assertEqual(sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest())


class RegistryHelperTests(unittest.TestCase):
    def test_registry_row_found_and_missing(self):
        row = registry_row(REGISTRY, TAG)
        self.assertIsNotNone(row)
        self.assertEqual(row["reward_mode"], "rule")
        self.assertIsNone(registry_row(REGISTRY, "does_not_exist"))

    def _rows(self):
        header_rows = []
        with REGISTRY.open(encoding="utf-8", newline="") as handle:
            header_rows = list(csv.DictReader(handle))
        return header_rows

    def test_reference_row_is_the_first_baseline_arm(self):
        rows = self._rows()
        # the plain rule arm is the reference, so asking for it returns None ...
        self.assertIsNone(reference_row(rows, TAG))
        # ... and every other arm is diffed against it.
        for tag in ("mini_rule_seed7", "mini_rule_len_seed42", "mini_mix_seed42"):
            reference = reference_row(rows, tag)
            self.assertIsNotNone(reference, tag)
            self.assertEqual(reference["run_tag"], TAG)
            self.assertTrue(is_baseline_row(reference))

    def test_changed_columns_reports_only_knob_differences(self):
        rows = {row["run_tag"]: row for row in self._rows()}
        changes = changed_columns(rows["mini_rule_dynamic_seed42"], rows[TAG])
        self.assertEqual(changes, [("dynamic_sampling", "0", "1")])
        self.assertEqual(changed_columns(rows["mini_mix_seed42"], rows[TAG]),
                         [("reward_mode", "rule", "mix")])
        self.assertEqual(changed_columns(rows[TAG], rows[TAG]), [])
        self.assertEqual(changed_columns(rows[TAG], None), [])


class CommandLeadLineTests(unittest.TestCase):
    def test_picks_the_python_line(self):
        text = "set -o pipefail\nmkdir -p ../runs/x\ncd trainer\npython train_grpo.py \\\n  --seed 42\n"
        self.assertEqual(command_lead_line(text), "python train_grpo.py \\")

    def test_warning_only_file_returns_none_marker(self):
        self.assertEqual(command_lead_line(None), "n/a")
        self.assertEqual(command_lead_line("# not registered\n"), "n/a")


class CheckpointTests(unittest.TestCase):
    def test_only_prefix_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "mini_rule_seed42_768.pth").write_bytes(b"a")
            (out / "mini_rule_seed42_768_resume.pth").write_bytes(b"bb")
            (out / "mini_rule_seed7_768.pth").write_bytes(b"ccc")
            entries = checkpoints_index("mini_rule_seed42", [out])
            self.assertEqual([Path(e["path"]).name for e in entries],
                             ["mini_rule_seed42_768.pth", "mini_rule_seed42_768_resume.pth"])
            self.assertEqual(entries[0]["bytes"], 1)
            self.assertEqual(len(entries[0]["sha256"]), 64)

    def test_missing_directory_is_not_an_error(self):
        self.assertEqual(checkpoints_index("tag", [Path("/nonexistent/dir")]), [])

    def test_render_notes_empty_case(self):
        text = render_checkpoints("tag", [], ["no checkpoint file found"])
        self.assertIn("(none found", text)
        self.assertIn("# note: no checkpoint file found", text)


class MetricsDigestTests(unittest.TestCase):
    def test_digest_reports_first_last_min_max(self):
        rows = [
            {"step": "5", "avg_reward": "0.1", "format_rate": "0.4"},
            {"step": "10", "avg_reward": "0.3", "format_rate": "0.6"},
        ]
        digest = metrics_digest(["step", "avg_reward", "format_rate"], rows)
        self.assertEqual(digest["rows"], 2)
        self.assertEqual(digest["first_step"], 5.0)
        self.assertEqual(digest["last_step"], 10.0)
        self.assertEqual(digest["first"]["avg_reward"], 0.1)
        self.assertEqual(digest["last"]["avg_reward"], 0.3)
        self.assertEqual(digest["min"]["avg_reward"], 0.1)
        self.assertEqual(digest["max"]["avg_reward"], 0.3)

    def test_empty_rows_are_reported_not_guessed(self):
        digest = metrics_digest(["step", "avg_reward"], [])
        self.assertEqual(digest["rows"], 0)
        self.assertEqual(digest["first"], {})

    def test_blank_and_garbage_values_do_not_crash(self):
        rows = [{"step": "5", "avg_reward": "", "entropy": "nan-ish"}]
        digest = metrics_digest(["step", "avg_reward", "entropy"], rows)
        self.assertIsNone(digest["first"]["avg_reward"])
        self.assertIsNone(digest["first"]["entropy"])


class SampleStatusTests(unittest.TestCase):
    def test_counts_statuses_and_bad_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl"
            write_samples(path)
            with path.open("a", encoding="utf-8") as handle:
                handle.write("{not json}\n")
            counts = sample_status_counts(path)
            self.assertEqual(counts["correct"], 1)
            self.assertEqual(counts["no_answer"], 1)
            self.assertEqual(counts["incorrect"], 1)
            self.assertEqual(counts["<unparseable>"], 1)

    def test_missing_file_returns_empty(self):
        self.assertEqual(sample_status_counts(Path("/nonexistent/samples.jsonl")), {})


class EvalDigestTests(unittest.TestCase):
    def test_pending_when_absent(self):
        self.assertIn("pending", eval_digest(None))

    def test_renders_scalars_and_difficulty_breakdown(self):
        payload = {
            "model": "out/x.pth",
            "seed": 42,
            "temperature": 0.8,
            "greedy_accuracy": 0.25,
            "sampled_pass_at_1": 0.3,
            "pass_at_16": None,
            "by_difficulty": {
                "easy": {"greedy_accuracy": 0.5, "sampled_pass_at_1": 0.5, "rows_evaluated": 133},
                "hard": 0.1,
            },
        }
        text = eval_digest(payload)
        self.assertIn("- greedy_accuracy: 0.25", text)
        self.assertIn("- pass_at_16: n/a", text)
        self.assertIn("easy: greedy=0.5", text)
        self.assertIn("hard: 0.1", text)


class ConfigAndEnvironmentTests(unittest.TestCase):
    def test_config_yaml_round_trip(self):
        import yaml

        payload = {"created_at": "now", "args": {"seed": 42, "reward_marker": "答案是"}}
        self.assertEqual(yaml.safe_load(render_config_yaml(payload)), payload)

    def test_environment_contains_core_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = collect_environment("mini_x", repo_root=Path(tmp), names=[])
        for key in ["captured_at", "platform", "python", "cuda_available", "gpu", "git_commit", "nvidia-smi", "packages"]:
            self.assertIn(f"{key}", text)
        self.assertIn("torch==", collect_environment("mini_x", repo_root=REPO_ROOT, names=["torch"]))


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.log_dir = self.root / "logs"
        self.log_dir.mkdir()
        self.runs_dir = self.root / "runs"
        write_metrics(self.log_dir / f"{TAG}_metrics.csv")
        write_samples(self.log_dir / f"{TAG}_samples.jsonl")
        write_config(self.log_dir / f"{TAG}_config.json")
        self.env = "python: 3.13.5\nplatform: test\ngpu: none\ngit_commit: deadbeef\nhost: testhost\n"

    def tearDown(self):
        self._tmp.cleanup()

    def archive(self, tag=TAG, **kwargs):
        return archive_run(
            tag,
            log_dir=self.log_dir,
            runs_dir=self.runs_dir,
            out_dir=self.root / "out",
            checkpoints_dir=self.root / "checkpoints",
            dataset_dir=DATASET,
            registry_path=REGISTRY,
            repo_root=self.root,
            environment=self.env,
            **kwargs,
        )

    def test_archive_without_eval_is_partial_but_complete_on_files(self):
        manifest = self.archive()
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual(manifest["missing"], ["eval.json"])
        self.assertEqual(manifest["metrics_rows"], 2)
        run_dir = self.runs_dir / TAG
        for name in REQUIRED_FILES:
            if name == "eval.json":
                continue
            self.assertTrue((run_dir / name).is_file(), name)
        self.assertIn("eval missing", manifest["notes"][0])

    def test_prerequisite_run_is_strict_clean_without_trainer_metrics(self):
        """冷启动这类**未登记**的 run 不该因为缺 metrics/samples/config 被判 FAIL。

        MiniMind 的 ``train_full_sft.py`` 不写这三样（runbook §6），所以它们"缺失"是设计
        使然。``status``/``missing`` 仍按完整清单算，保持 ``[partial]`` 的诚实表述；但
        ``--check --strict`` 依据该 run 自己的 ``required_files``，否则一个行为完全符合
        文档的 run 会被闸门判死。
        """

        tag = "math_cold_cot_seed42"
        self.archive(tag)
        run_dir = self.runs_dir / tag
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["prerequisite"])
        for name in ("metrics.csv", "samples.jsonl", "config.yaml"):
            self.assertNotIn(name, manifest["required_files"])
            self.assertIn(name, manifest["missing"])
        self.assertEqual(manifest["status"], "partial")

        (run_dir / "train.log").write_text("log\n", encoding="utf-8")
        problems = check_archive(run_dir, strict=True)
        for name in ("metrics.csv", "samples.jsonl", "config.yaml"):
            self.assertFalse([p for p in problems if p.startswith(name)], problems)

    def test_registered_arm_is_still_required_to_have_metrics(self):
        """对照组：登记表里的臂缺 metrics 必须继续判 FAIL，别把口子开太大。"""

        self.archive()
        run_dir = self.runs_dir / TAG
        (run_dir / "metrics.csv").unlink()
        problems = check_archive(run_dir, strict=True)
        self.assertTrue([p for p in problems if p.startswith("metrics.csv")], problems)

    def test_command_txt_is_registry_derived_and_archival(self):
        self.archive()
        text = (self.runs_dir / TAG / "command.txt").read_text(encoding="utf-8")
        self.assertIn("set -o pipefail", text)
        self.assertIn(f"tee ../runs/{TAG}/train.log", text)
        self.assertIn("--num_generations 6", text)
        self.assertIn("--reward_mode rule", text)
        self.assertNotIn("--reward_model_path", text)

    def test_rm_arm_command_includes_reward_model_path(self):
        write_metrics(self.log_dir / "mini_rm_seed42_metrics.csv")
        self.archive("mini_rm_seed42")
        text = (self.runs_dir / "mini_rm_seed42" / "command.txt").read_text(encoding="utf-8")
        self.assertIn("--reward_model_path", text)

    def test_raw_evidence_is_copied_byte_for_byte(self):
        self.archive()
        for name in ("metrics.csv", "samples.jsonl"):
            source = (self.log_dir / f"{TAG}_{name}").read_bytes()
            self.assertEqual((self.runs_dir / TAG / name).read_bytes(), source)

    def test_data_manifest_records_matching_hashes(self):
        self.archive()
        payload = json.loads((self.runs_dir / TAG / "data_manifest.json").read_text(encoding="utf-8"))
        entry = payload["files"][str(DATASET / "math_rlvr_train.jsonl")]
        self.assertTrue(entry["exists"])
        self.assertIs(entry["matches_manifest"], True)
        self.assertEqual(payload["registry_row"]["run_tag"], TAG)

    def test_data_manifest_flags_modified_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            (dataset / "manifests").mkdir(parents=True)
            (dataset / "math_rlvr_train.jsonl").write_text("tampered\n", encoding="utf-8")
            (dataset / "manifests" / "math_manifest.json").write_text(
                json.dumps({"outputs": {"math_rlvr_train.jsonl": "0" * 64}}), encoding="utf-8"
            )
            payload = data_manifest_payload(
                run_tag=TAG,
                row=None,
                manifest_path=dataset / "manifests" / "math_manifest.json",
                data_paths=[dataset / "math_rlvr_train.jsonl"],
            )
            entry = payload["files"][str(dataset / "math_rlvr_train.jsonl")]
            self.assertIs(entry["matches_manifest"], False)

    def test_analysis_quotes_hypothesis_and_leaves_conclusions_open(self):
        self.archive()
        text = (self.runs_dir / TAG / "analysis.md").read_text(encoding="utf-8")
        row = registry_row(REGISTRY, TAG)
        self.assertIn(row["hypothesis"], text)
        self.assertIn("## 6. 结论边界", text)
        self.assertIn("TODO", text)
        self.assertIn("- steps logged: 2 (step 5 -> 10)", text)
        self.assertIn("- command: `python -u train_grpo.py \\`", text)
        self.assertIn("- dynamic sampling: filtered_fraction 0.5", text)

    def test_analysis_lists_the_single_changed_factor(self):
        write_metrics(self.log_dir / "mini_rule_dynamic_seed42_metrics.csv")
        self.archive("mini_rule_dynamic_seed42")
        text = (self.runs_dir / "mini_rule_dynamic_seed42" / "analysis.md").read_text(encoding="utf-8")
        self.assertIn("- dynamic_sampling: reference=0 -> this run=1", text)

    def test_analysis_marks_the_baseline_arm(self):
        self.archive()
        text = (self.runs_dir / TAG / "analysis.md").read_text(encoding="utf-8")
        self.assertIn("reference arm itself", text)

    def test_eval_json_is_attached_and_flips_status_to_complete(self):
        eval_path = self.root / "eval.json"
        eval_path.write_text(json.dumps({
            "model": "out/x.pth", "seed": 42, "temperature": 0.8,
            "greedy_accuracy": 0.25, "sampled_pass_at_1": 0.3,
            "format_rate": 0.9, "by_difficulty": {"easy": 0.5},
        }), encoding="utf-8")
        Path(str(eval_path) + "l").write_text("{}\n", encoding="utf-8")
        manifest = self.archive(eval_json=eval_path)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["missing"], [])
        self.assertTrue((self.runs_dir / TAG / "eval.jsonl").is_file())
        self.assertIn("- greedy_accuracy: 0.25",
                      (self.runs_dir / TAG / "analysis.md").read_text(encoding="utf-8"))

    def test_unregistered_tag_gets_an_honest_command_file(self):
        write_metrics(self.log_dir / "unregistered_tag_metrics.csv")
        manifest = self.archive("unregistered_tag")
        text = (self.runs_dir / "unregistered_tag" / "command.txt").read_text(encoding="utf-8")
        self.assertIn("not registered", text)
        self.assertIn("experiment_registry.csv", text)  # names the registry missing the row
        self.assertTrue(any("not present" in note for note in manifest["notes"]))

    def test_placeholder_tag_is_refused(self):
        with self.assertRaises(ValueError):
            self.archive("unnamed")

    def test_existing_archive_needs_force(self):
        self.archive()
        with self.assertRaises(FileExistsError):
            self.archive()
        manifest = self.archive(force=True)
        self.assertEqual(manifest["run_tag"], TAG)

    def test_missing_logs_are_reported_not_invented(self):
        manifest = self.archive("mini_rule_len_seed42")
        self.assertTrue(manifest["missing"])
        self.assertIn("eval.json", manifest["missing"])
        self.assertTrue(any("metrics missing" in note for note in manifest["notes"]))
        self.assertEqual(manifest["metrics_rows"], 0)


class CheckArchiveTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.log_dir = self.root / "logs"
        self.log_dir.mkdir()
        self.runs_dir = self.root / "runs"
        write_metrics(self.log_dir / f"{TAG}_metrics.csv")
        write_samples(self.log_dir / f"{TAG}_samples.jsonl")
        write_config(self.log_dir / f"{TAG}_config.json")
        self.eval_path = self.root / "eval.json"
        self.eval_path.write_text(json.dumps({
            "model": "out/x.pth", "seed": 42, "temperature": 0.8,
            "greedy_accuracy": 0.25, "sampled_pass_at_1": 0.3,
            "format_rate": 0.9, "by_difficulty": {},
        }), encoding="utf-8")
        archive_run(
            TAG, log_dir=self.log_dir, runs_dir=self.runs_dir, out_dir=self.root / "out",
            checkpoints_dir=self.root / "checkpoints", dataset_dir=DATASET, registry_path=REGISTRY,
            repo_root=self.root, environment="python: 3.13.5\n", eval_json=self.eval_path,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_fresh_archive_has_no_problems(self):
        self.assertEqual(check_archive(self.runs_dir / TAG), [])

    def test_strict_complains_about_the_missing_train_log(self):
        problems = check_archive(self.runs_dir / TAG, strict=True)
        self.assertTrue(any("train.log" in problem for problem in problems))

    def test_tampering_is_detected(self):
        (self.runs_dir / TAG / "metrics.csv").write_text("x\n", encoding="utf-8")
        problems = check_archive(self.runs_dir / TAG)
        self.assertTrue(any("sha256 changed" in problem for problem in problems))

    def test_header_only_metrics_is_rejected(self):
        with (self.runs_dir / TAG / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_HEADER)
            writer.writeheader()
        problems = check_archive(self.runs_dir / TAG)
        self.assertTrue(any("no data rows" in problem for problem in problems))

    def test_eval_without_required_fields_is_rejected(self):
        (self.runs_dir / TAG / "eval.json").write_text(json.dumps({"model": "x"}), encoding="utf-8")
        problems = check_archive(self.runs_dir / TAG)
        self.assertTrue(any("greedy_accuracy" in problem for problem in problems))

    def test_missing_manifest_is_reported(self):
        (self.runs_dir / TAG / "manifest.json").unlink()
        problems = check_archive(self.runs_dir / TAG)
        self.assertTrue(any("manifest.json missing" in problem for problem in problems))


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.log_dir = self.root / "logs"
        self.log_dir.mkdir()
        self.runs_dir = self.root / "runs"
        write_metrics(self.log_dir / f"{TAG}_metrics.csv")
        write_samples(self.log_dir / f"{TAG}_samples.jsonl")
        write_config(self.log_dir / f"{TAG}_config.json")
        (self.log_dir / "mini_rule_seed7_metrics.csv").write_text(
            ",".join(METRIC_HEADER) + "\n", encoding="utf-8"
        )

    def tearDown(self):
        self._tmp.cleanup()

    def base(self, *extra):
        return [
            "--log-dir", str(self.log_dir),
            "--runs-dir", str(self.runs_dir),
            "--out-dir", str(self.root / "out"),
            "--checkpoints-dir", str(self.root / "checkpoints"),
            "--dataset-dir", str(DATASET),
            "--registry", str(REGISTRY),
            *extra,
        ]

    def test_no_target_returns_usage_error(self):
        self.assertEqual(main(self.base()), 2)

    def test_archive_one_then_check(self):
        self.assertEqual(main(self.base("--run-tag", TAG, "--quiet")), 0)
        self.assertTrue((self.runs_dir / TAG / "analysis.md").is_file())
        self.assertEqual(main(self.base("--run-tag", TAG, "--check", "--quiet")), 0)

    def test_check_without_strict_tolerates_a_missing_eval(self):
        self.assertEqual(main(self.base("--run-tag", TAG, "--quiet")), 0)
        self.assertEqual(main(self.base("--run-tag", TAG, "--check", "--quiet")), 0)
        self.assertEqual(main(self.base("--run-tag", TAG, "--check", "--strict", "--quiet")), 1)

    def test_check_reports_a_broken_archive(self):
        main(self.base("--run-tag", TAG, "--quiet"))
        (self.runs_dir / TAG / "samples.jsonl").unlink()
        self.assertEqual(main(self.base("--run-tag", TAG, "--check", "--quiet")), 1)

    def test_all_archives_only_registered_and_logged_runs(self):
        write_metrics(self.log_dir / "stranger_metrics.csv")
        self.assertEqual(main(self.base("--all", "--quiet")), 0)
        self.assertTrue((self.runs_dir / TAG / "command.txt").is_file())
        self.assertTrue((self.runs_dir / "mini_rule_seed7").is_dir())  # header-only metrics still archived
        self.assertFalse((self.runs_dir / "stranger").exists())

    def test_all_without_logs_returns_one(self):
        empty = self.root / "empty_logs"
        empty.mkdir()
        args = self.base("--all", "--quiet")
        args[args.index(str(self.log_dir))] = str(empty)
        self.assertEqual(main(args), 1)

    def test_all_and_run_tag_conflict(self):
        self.assertEqual(main(self.base("--all", "--run-tag", TAG)), 2)

    def test_force_is_required_to_refresh(self):
        main(self.base("--run-tag", TAG, "--quiet"))
        self.assertEqual(main(self.base("--run-tag", TAG, "--quiet")), 1)
        self.assertEqual(main(self.base("--run-tag", TAG, "--force", "--quiet")), 0)


if __name__ == "__main__":
    unittest.main()
