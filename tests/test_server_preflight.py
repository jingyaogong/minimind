import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.server_preflight import (
    FAIL,
    OK,
    WARN,
    check_dataset,
    check_disk,
    check_layout,
    check_packages,
    check_python,
    check_torch,
    check_weights,
    check_writable,
    collect,
    main,
    read_recorded_hashes,
    render,
    sha256_file,
    summarize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class PythonCheckTests(unittest.TestCase):
    def test_current_interpreter_passes(self):
        self.assertEqual(check_python()["status"], OK)

    def test_old_interpreter_fails(self):
        result = check_python((3, 9))
        self.assertEqual(result["status"], FAIL)
        self.assertIn("3.9", result["detail"])


class PackageCheckTests(unittest.TestCase):
    def test_missing_required_is_a_blocker_and_missing_optional_is_a_warning(self):
        checks = {item["name"]: item for item in check_packages(versions={"torch": "2.6.0"})}
        self.assertEqual(checks["package:torch"]["status"], OK)
        self.assertEqual(checks["package:transformers"]["status"], FAIL)
        self.assertEqual(checks["package:matplotlib"]["status"], WARN)
        self.assertIn("pip install", checks["package:datasets"]["detail"])


class TorchCheckTests(unittest.TestCase):
    def test_cpu_only_is_a_warning_by_default_and_a_blocker_when_expected(self):
        cuda = {"version": "2.6.0", "available": False, "count": 0}
        self.assertEqual(check_torch(cuda, expect_gpu=False)[1]["status"], WARN)
        self.assertEqual(check_torch(cuda, expect_gpu=True)[1]["status"], FAIL)

    def test_small_vram_and_missing_bf16_are_warned_about(self):
        cuda = {"version": "2.6.0", "available": True, "count": 1, "name": "T4",
                "memory_gb": 16.0, "capability": (7, 5), "bf16": False}
        checks = {item["name"]: item for item in check_torch(cuda, expect_gpu=True)}
        self.assertEqual(checks["torch/cuda"]["status"], OK)
        self.assertEqual(checks["dtype"]["status"], WARN)
        self.assertEqual(checks["vram"]["status"], WARN)

    def test_healthy_gpu_is_clean(self):
        cuda = {"version": "2.6.0", "available": True, "count": 1, "name": "RTX 4090",
                "memory_gb": 24.0, "capability": (8, 9), "bf16": True}
        checks = check_torch(cuda, expect_gpu=True)
        self.assertTrue(all(item["status"] == OK for item in checks))


class LayoutCheckTests(unittest.TestCase):
    def test_missing_files_are_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            checks = {item["name"]: item for item in check_layout(Path(tmp))}
        self.assertEqual(checks["trainer/train_grpo.py"]["status"], FAIL)
        self.assertEqual(checks["dataset/manifests/data.sha256"]["status"], WARN)

    def test_present_files_are_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in ("model/tokenizer.json", "trainer/train_grpo.py"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")
            checks = {item["name"]: item for item in check_layout(root)}
        self.assertEqual(checks["model/tokenizer.json"]["status"], OK)
        self.assertEqual(checks["configs/experiment_registry.csv"]["status"], FAIL)


class DatasetCheckTests(unittest.TestCase):
    def test_matching_hashes_pass_and_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            for name in ("math_sft_train.jsonl", "math_rlvr_train.jsonl", "math_heldout.jsonl"):
                (dataset / name).write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
            recorded = {name: sha256_file(dataset / name)
                        for name in ("math_sft_train.jsonl", "math_rlvr_train.jsonl")}
            recorded["math_heldout.jsonl"] = "0" * 64
            checks = {item["name"]: item for item in check_dataset(dataset, recorded)}
        self.assertEqual(checks["math_sft_train.jsonl"]["status"], OK)
        self.assertIn("2 rows", checks["math_sft_train.jsonl"]["detail"])
        self.assertEqual(checks["math_heldout.jsonl"]["status"], FAIL)
        self.assertIn("changed after the registry was written", checks["math_heldout.jsonl"]["detail"])

    def test_missing_file_points_at_the_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            checks = {item["name"]: item for item in check_dataset(Path(tmp), {})}
        self.assertEqual(checks["math_rlvr_train.jsonl"]["status"], FAIL)
        self.assertIn("clean_math_data.py", checks["math_rlvr_train.jsonl"]["detail"])

    def test_read_recorded_hashes_parses_shasum_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.sha256"
            path.write_text("abc  *math_sft_train.jsonl\ndef  math_heldout.jsonl\n", encoding="utf-8")
            recorded = read_recorded_hashes(path)
        self.assertEqual(recorded["math_sft_train.jsonl"], "abc")
        self.assertEqual(recorded["math_heldout.jsonl"], "def")
        self.assertEqual(read_recorded_hashes(Path("/nonexistent/data.sha256")), {})


class WeightCheckTests(unittest.TestCase):
    def test_missing_checkpoint_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = check_weights(Path(tmp))
        self.assertEqual(result["status"], FAIL)
        self.assertIn("strict=False", result["detail"])

    def test_present_checkpoint_reports_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full_sft_768.pth"
            path.write_bytes(b"0" * 2048)
            result = check_weights(Path(tmp))
        self.assertEqual(result["status"], OK)
        self.assertIn("GB", result["detail"])


class DiskAndWritableTests(unittest.TestCase):
    def test_implausible_disk_requirement_fails(self):
        checks = check_disk({"repo": REPO_ROOT}, min_gb=10**7)
        self.assertEqual(checks[0]["status"], FAIL)

    def test_nested_missing_path_resolves_to_an_existing_ancestor(self):
        checks = check_disk({"out": REPO_ROOT / "out" / "deep" / "deeper"}, min_gb=0.0)
        self.assertEqual(checks[0]["status"], OK)

    @unittest.skipIf(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        "root 有 CAP_DAC_OVERRIDE，会绕过 chmod 权限位；FAIL 分支由 "
        "test_writable_fails_when_parent_is_a_file 覆盖",
    )
    def test_read_only_directory_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "readonly"
            target.mkdir()
            target.chmod(0o500)
            try:
                checks = check_writable([target])
            finally:
                target.chmod(0o700)
        self.assertEqual(checks[0]["status"], FAIL)

    def test_writable_fails_when_parent_is_a_file(self):
        # 与 uid 无关的 FAIL 用例：父路径是普通文件时 mkdir(parents=True) 必然抛 OSError，
        # 所以在 root（服务器）和非 root（本地）下行为一致
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "afile"
            blocker.write_text("x", encoding="utf-8")
            checks = check_writable([blocker / "child"])
        self.assertEqual(checks[0]["status"], FAIL)

    def test_writable_directory_is_created_on_demand(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "runs"
            checks = check_writable([target])
            self.assertEqual(checks[0]["status"], OK)
            self.assertTrue(target.is_dir())


class AggregateTests(unittest.TestCase):
    def test_summarize_counts_each_status(self):
        checks = [{"status": OK, "name": "a", "detail": ""},
                  {"status": WARN, "name": "b", "detail": ""},
                  {"status": FAIL, "name": "c", "detail": ""}]
        self.assertEqual(summarize(checks), {OK: 1, WARN: 1, FAIL: 1})

    def test_render_mentions_blockers(self):
        text = render([{"status": FAIL, "name": "weights:x.pth", "detail": "missing"}])
        self.assertIn("FAIL", text)
        self.assertIn("1 blockers", text)
        self.assertIn("Do not start training", text)

    def test_collect_reports_the_expected_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            checks = collect(repo_root=Path(tmp), dataset_dir=Path(tmp) / "dataset",
                             out_dir=Path(tmp) / "out", min_disk_gb=0.0)
        names = {item["name"] for item in checks}
        for expected in ("python", "torch/cuda", "trainer/train_grpo.py", "math_heldout.jsonl",
                         "weights:full_sft_768.pth", "writable:" + str(Path(tmp) / "runs")):
            self.assertIn(expected, names)


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "model").mkdir()
        (self.root / "model" / "tokenizer.json").write_text("{}", encoding="utf-8")
        self.dataset = self.root / "dataset"
        (self.dataset / "manifests").mkdir(parents=True)
        for name in ("math_sft_train.jsonl", "math_rlvr_train.jsonl", "math_heldout.jsonl"):
            (self.dataset / name).write_text('{"a": 1}\n', encoding="utf-8")
        (self.dataset / "manifests" / "data.sha256").write_text(
            "".join(f"{sha256_file(self.dataset / name)}  {name}\n"
                    for name in ("math_sft_train.jsonl", "math_rlvr_train.jsonl", "math_heldout.jsonl")),
            encoding="utf-8")
        (self.root / "out").mkdir()
        (self.root / "out" / "full_sft_768.pth").write_bytes(b"0" * 16)

    def tearDown(self):
        self._tmp.cleanup()

    def args(self, *extra):
        return ["--repo-root", str(self.root), "--dataset-dir", str(self.dataset),
                "--out-dir", str(self.root / "out"), "--min-disk-gb", "0", *extra]

    def test_exit_zero_when_only_warnings_remain(self):
        with mock.patch("scripts.server_preflight.check_packages", return_value=[]), \
             mock.patch("scripts.server_preflight.check_layout", return_value=[]), \
             mock.patch("scripts.server_preflight.check_torch",
                        return_value=[{"status": WARN, "name": "gpu", "detail": "cpu only"}]):
            self.assertEqual(main(self.args("--quiet")), 0)

    def test_exit_one_when_a_blocker_remains(self):
        (self.root / "out" / "full_sft_768.pth").unlink()
        with mock.patch("scripts.server_preflight.check_packages", return_value=[]), \
             mock.patch("scripts.server_preflight.check_layout", return_value=[]), \
             mock.patch("scripts.server_preflight.check_torch",
                        return_value=[{"status": OK, "name": "gpu", "detail": "ok"}]):
            self.assertEqual(main(self.args("--quiet")), 1)

    def test_json_output_is_written(self):
        out = self.root / "preflight.json"
        with mock.patch("scripts.server_preflight.check_packages", return_value=[]), \
             mock.patch("scripts.server_preflight.check_layout", return_value=[]), \
             mock.patch("scripts.server_preflight.check_torch",
                        return_value=[{"status": OK, "name": "gpu", "detail": "ok"}]):
            code = main(self.args("--quiet", "--json", str(out)))
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertIn("summary", payload)
        self.assertTrue(any(item["name"].startswith("weights:") for item in payload["checks"]))
        self.assertIn("dataset", " ".join(item["name"] for item in payload["checks"]))


class RepoLayoutSmokeTest(unittest.TestCase):
    """The real repository must already satisfy the static part of the checklist."""

    def test_repo_files_are_complete(self):
        checks = {item["name"]: item for item in check_layout(REPO_ROOT)}
        failures = [name for name, item in checks.items() if item["status"] == FAIL]
        self.assertEqual(failures, [])

    def test_repo_dataset_matches_data_sha256(self):
        recorded = read_recorded_hashes(REPO_ROOT / "dataset" / "manifests" / "data.sha256")
        checks = check_dataset(REPO_ROOT / "dataset", recorded)
        self.assertTrue(all(item["status"] == OK for item in checks), checks)


if __name__ == "__main__":
    unittest.main()
