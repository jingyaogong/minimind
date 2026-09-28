import csv
import shlex
import tempfile
import unittest
from pathlib import Path

from scripts.experiment_registry import (
    COLUMNS,
    DEFAULT_REGISTRY,
    DEFAULT_TRAINER,
    build_args,
    check_registry,
    load_registry,
    main,
    render_command,
    save_command,
    trainer_flags,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def base_row(**overrides):
    row = {
        "run_tag": "mini_test_seed42",
        "model": "MiniMind-64M",
        "reward_mode": "rule",
        "loss_type": "grpo",
        "seed": "42",
        "G": "6",
        "max_gen_len": "512",
        "beta": "0.1",
        "dynamic_sampling": "0",
        "len_penalty_coef": "0",
        "alpha": "0.5",
        "beta_mix": "0.5",
        "hypothesis": "unit test hypothesis",
    }
    row.update(overrides)
    return row


class RegistryFileTests(unittest.TestCase):
    def test_header_matches_the_sop_schema(self):
        header, rows = load_registry(DEFAULT_REGISTRY)
        self.assertEqual(header, COLUMNS)
        self.assertEqual(len(rows), 6)

    def test_preregistered_runs_are_consistent(self):
        header, rows = load_registry(DEFAULT_REGISTRY)
        problems = check_registry(header, rows, trainer_flags(DEFAULT_TRAINER))
        self.assertEqual(problems, [])

    def test_run_tags_are_unique(self):
        _, rows = load_registry(DEFAULT_REGISTRY)
        tags = [row["run_tag"] for row in rows]
        self.assertEqual(len(tags), len(set(tags)))

    def test_arms_differ_by_one_factor_from_the_baseline(self):
        _, rows = load_registry(DEFAULT_REGISTRY)
        baseline = rows[0]
        self.assertEqual(baseline["run_tag"], "mini_rule_seed42")
        self.assertEqual(baseline["dynamic_sampling"], "0")
        self.assertEqual(baseline["len_penalty_coef"], "0")
        for row in rows[1:]:
            differing = [c for c in COLUMNS if row[c] != baseline[c]]
            self.assertLessEqual(
                len(differing), 3, f"{row['run_tag']} changes too many factors: {differing}"
            )


class CheckRegistryTests(unittest.TestCase):
    def setUp(self):
        self.flags = trainer_flags(DEFAULT_TRAINER)

    def test_duplicate_run_tag_is_rejected(self):
        rows = [base_row(), base_row()]
        problems = check_registry(COLUMNS, rows, self.flags)
        self.assertTrue(any("duplicated run_tag" in p for p in problems))

    def test_unknown_reward_mode_is_rejected(self):
        problems = check_registry(COLUMNS, [base_row(reward_mode="verifier")], self.flags)
        self.assertTrue(any("reward_mode" in p for p in problems))

    def test_group_size_one_is_rejected(self):
        problems = check_registry(COLUMNS, [base_row(G="1")], self.flags)
        self.assertTrue(any("G=1" in p for p in problems))

    def test_kl_disabled_is_rejected(self):
        problems = check_registry(COLUMNS, [base_row(beta="0")], self.flags)
        self.assertTrue(any("beta=0.0" in p for p in problems))

    def test_header_mismatch_is_rejected(self):
        problems = check_registry(COLUMNS[:-1], [base_row()], self.flags)
        self.assertTrue(any("header mismatch" in p for p in problems))

    def test_missing_trainer_flag_is_rejected(self):
        problems = check_registry(COLUMNS, [base_row()], set())
        self.assertTrue(any("train_grpo.py does not define" in p for p in problems))

    def test_dynamic_sampling_without_verifier_is_rejected(self):
        problems = check_registry(
            COLUMNS, [base_row(reward_mode="official", dynamic_sampling="1")], self.flags
        )
        self.assertTrue(any("dynamic_sampling" in p for p in problems))

    def test_two_factors_at_once_is_only_a_warning(self):
        warnings: list[str] = []
        problems = check_registry(
            COLUMNS, [base_row(dynamic_sampling="1", len_penalty_coef="0.05")], self.flags, warnings
        )
        self.assertEqual(problems, [])
        self.assertTrue(any("one factor per run" in w for w in warnings))

    def test_empty_hypothesis_is_rejected(self):
        problems = check_registry(COLUMNS, [base_row(hypothesis="")], self.flags)
        self.assertTrue(any("hypothesis" in p for p in problems))


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.flags = trainer_flags(DEFAULT_TRAINER)

    def test_every_generated_flag_exists_in_the_trainer(self):
        _, rows = load_registry(DEFAULT_REGISTRY)
        for row in rows:
            args = build_args(row)
            for flag in args[::2]:
                self.assertIn(flag, self.flags, f"{row['run_tag']} generated unknown flag {flag}")

    def test_rule_arm_never_passes_a_reward_model(self):
        args = build_args(base_row(reward_mode="rule"))
        self.assertNotIn("--reward_model_path", args)

    def test_rm_and_mix_arms_pass_the_reward_model(self):
        for mode in ("rm", "mix"):
            args = build_args(base_row(reward_mode=mode), reward_model_path="/abs/rm")
            self.assertIn("--reward_model_path", args)
            self.assertEqual(args[args.index("--reward_model_path") + 1], "/abs/rm")

    def test_registered_factors_reach_the_command(self):
        args = build_args(base_row(dynamic_sampling="1", G="4", seed="7", max_gen_len="256"))
        self.assertEqual(args[args.index("--dynamic_sampling") + 1], "1")
        self.assertEqual(args[args.index("--num_generations") + 1], "4")
        self.assertEqual(args[args.index("--seed") + 1], "7")
        self.assertEqual(args[args.index("--max_gen_len") + 1], "256")

    def test_run_tag_drives_save_weight_and_logs(self):
        args = build_args(base_row(run_tag="mini_x_seed42"))
        self.assertEqual(args[args.index("--save_weight") + 1], "mini_x_seed42")
        block = render_command("mini_x_seed42", args)
        self.assertIn("runs/mini_x_seed42/train.log", block)
        self.assertIn("mkdir -p runs/mini_x_seed42", block)

    def test_mkdir_creates_the_directory_tee_writes_into(self):
        """回归：``mkdir`` 在 ``cd trainer`` 之前执行，``tee`` 之后执行。

        两条路径的基准目录不同（仓库根 vs ``trainer/``），所以必须解析后指向同一个
        ``runs/<tag>/``。旧版本写的是 ``mkdir -p ../runs/<tag>``，会在仓库上一层建目录，
        tee 随即写进一个不存在的目录并失败——训练照跑但永远没有 ``train.log``。
        """

        tag = "mini_x_seed42"
        block = render_command(tag, build_args(base_row(run_tag=tag)))
        lines = block.splitlines()
        mkdir_target = Path(next(l for l in lines if l.startswith("mkdir -p ")).split("mkdir -p ")[-1].strip())
        tee_target = Path(next(l for l in lines if "tee " in l).split("tee ")[-1].strip()).parent
        root = Path("/repo")
        self.assertEqual((root / mkdir_target).resolve(), (root / "trainer" / tee_target).resolve())
        self.assertEqual(tee_target.name, tag)

    def test_continuation_backslashes_keep_it_one_command(self):
        args = build_args(base_row())
        block = render_command("t", args, archival=False)
        lines = [line for line in block.splitlines() if line.strip()]
        self.assertTrue(lines[0].startswith("python -u train_grpo.py"),
                        "the archival form pipes into tee, so stdout must be unbuffered")
        for line in lines[:-1]:
            self.assertTrue(line.endswith("\\"), f"line would break the command: {line!r}")
        joined = " ".join(line.rstrip("\\").strip() for line in lines)
        tokens = shlex.split(joined)
        self.assertEqual(tokens[:3], ["python", "-u", "train_grpo.py"])
        self.assertEqual(tokens[3::2], args[::2])
        # shlex unquotes the marker, so the parsed values must round-trip exactly
        self.assertEqual(tokens[4::2], args[1::2])

    def test_save_command_writes_the_archive_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_command("mini_x_seed42", build_args(base_row()), run_dir=Path(tmp))
            self.assertTrue(path.is_file())
            self.assertIn("train_grpo.py", path.read_text(encoding="utf-8"))


class CliTests(unittest.TestCase):
    def test_check_passes(self):
        self.assertEqual(main(["--check"]), 0)

    def test_unknown_run_tag_exits_two(self):
        self.assertEqual(main(["--print-command", "nope"]), 2)

    def test_print_command_for_each_registered_run(self):
        _, rows = load_registry(DEFAULT_REGISTRY)
        for row in rows:
            self.assertEqual(main(["--plain", "--print-command", row["run_tag"]]), 0)

    def test_save_command_uses_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                main(["--save-command", "mini_rule_seed42", "--run-dir", tmp]), 0
            )
            self.assertTrue((Path(tmp) / "mini_rule_seed42" / "command.txt").is_file())

    def test_tampered_registry_fails_the_check(self):
        header, rows = load_registry(DEFAULT_REGISTRY)
        rows[0]["beta"] = "0"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(main(["--check", "--registry", str(path)]), 1)

    def test_missing_registry_is_reported(self):
        self.assertEqual(main(["--check", "--registry", "/nonexistent/registry.csv"]), 2)


if __name__ == "__main__":
    unittest.main()
