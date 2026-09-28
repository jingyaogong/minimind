import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.archive_run import archive_run
from scripts.experiment_registry import load_registry
from scripts.summarize_runs import (
    ARM_KNOBS,
    CSV_COLUMNS,
    EXTRA_TAGS,
    MATRIX_ROWS,
    add_cross_run_columns,
    build_row,
    collect_run,
    decision_tree,
    reward_gain_cell,
    reward_trend_cell,
    fig_reward_eval_gap,
    figure_caption,
    first_last,
    fmt,
    fmt_delta,
    format_cell,
    length_samples,
    main,
    numeric_rows,
    render_matrix,
    render_seed_aggregate,
    render_summary_md,
    run_rows,
    series,
    trend_endpoints,
    trend_window,
    write_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "configs" / "experiment_registry.csv"
DATASET = REPO_ROOT / "dataset"

METRIC_HEADER = [
    "timestamp", "step", "seed", "avg_reward", "format_rate", "correct_rate", "no_answer_rate",
    "avg_completion_tokens", "median_completion_tokens", "kl_ref", "entropy", "adv_std",
    "policy_loss", "valid_groups", "all_correct_groups", "all_wrong_groups", "filtered_fraction",
]


def write_metrics(path: Path, *, reward=(0.1, 0.3), correct=(0.1, 0.2), format_rate=(0.4, 0.5),
                  tokens=(100, 80), groups=(3, 1, 2)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_HEADER)
        writer.writeheader()
        for index in range(2):
            writer.writerow({
                "timestamp": f"2026-09-24T0{index}:00:00+00:00", "step": 5 + index * 5, "seed": 42,
                "avg_reward": reward[index], "format_rate": format_rate[index],
                "correct_rate": correct[index], "no_answer_rate": 0.1,
                "avg_completion_tokens": tokens[index], "median_completion_tokens": tokens[index] - 4,
                "kl_ref": -1.5, "entropy": 8.0 + index, "adv_std": 0.2, "policy_loss": 0.05,
                "valid_groups": groups[0], "all_correct_groups": groups[1], "all_wrong_groups": groups[2],
                "filtered_fraction": 0.5,
            })


def eval_payload(**overrides) -> dict:
    payload = {
        "model": "out/x_768.pth", "seed": 42, "temperature": 0.8, "num_samples": 16,
        "greedy_accuracy": 0.30, "sampled_pass_at_1": 0.35, "pass_at_4": 0.5,
        "pass_at_8": 0.6, "pass_at_16": 0.7, "format_rate": 0.9, "no_answer_rate": 0.05,
        "marker_hit_rate": 0.95, "mean_completion_tokens": 180.0, "median_completion_tokens": 176.0,
        "rows_evaluated": 400, "data_matches_manifest": True,
        "by_difficulty": {
            "easy": {"greedy_accuracy": 0.5, "sampled_pass_at_1": 0.55},
            "medium": {"greedy_accuracy": 0.3, "sampled_pass_at_1": 0.33},
            "hard": {"greedy_accuracy": 0.1, "sampled_pass_at_1": 0.12},
        },
    }
    payload.update(overrides)
    return payload


class LayoutFixture(unittest.TestCase):
    """A temp repo layout with real (archive_run-built) run archives."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.log_dir = self.root / "logs"
        self.runs_dir = self.root / "runs"
        self.out_dir = self.root / "reports"
        self.log_dir.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def make_run(self, tag: str, *, archive: bool = True, eval_data: dict | None = None,
                 eval_jsonl: bool = True, **metric_kwargs):
        write_metrics(self.log_dir / f"{tag}_metrics.csv", **metric_kwargs)
        (self.log_dir / f"{tag}_config.json").write_text(
            json.dumps({"created_at": "now", "args": {"run_tag": tag}}), encoding="utf-8")
        (self.log_dir / f"{tag}_samples.jsonl").write_text(
            json.dumps({"step": 5, "status": "correct", "completion_tokens": 90}) + "\n", encoding="utf-8")
        if eval_data is not None:
            (self.log_dir / f"{tag}_eval.json").write_text(
                json.dumps(eval_data, ensure_ascii=False, indent=2), encoding="utf-8")
            if eval_jsonl:
                (self.log_dir / f"{tag}_eval.jsonl").write_text(
                    "".join(json.dumps({"sample_id": f"s{i}", "mode": "sampled", "tokens": 100 + i,
                                        "status": "correct"}) + "\n" for i in range(5)),
                    encoding="utf-8")
        if archive:
            archive_run(
                tag, log_dir=self.log_dir, runs_dir=self.runs_dir, out_dir=self.root / "out",
                checkpoints_dir=self.root / "checkpoints", dataset_dir=DATASET,
                registry_path=REGISTRY, repo_root=self.root, environment="python: test\n",
            )

    def summarize(self, *extra):
        return main([
            "--runs-dir", str(self.runs_dir), "--log-dir", str(self.log_dir),
            "--out-dir", str(self.out_dir), "--registry", str(REGISTRY), *extra,
        ])

    def rows(self, tags=None):
        rows, _notes, _runs = run_rows(
            runs_dir=self.runs_dir, log_dir=self.log_dir, registry_path=REGISTRY, tags=tags,
        )
        return rows


class FormattingTests(unittest.TestCase):
    def test_format_cell(self):
        self.assertEqual(format_cell(None), "")
        self.assertEqual(format_cell(True), "true")
        self.assertEqual(format_cell(False), "false")
        self.assertEqual(format_cell(0.1000000001), "0.1")
        self.assertEqual(format_cell("rule"), "rule")

    def test_fmt_and_delta(self):
        self.assertEqual(fmt(None), "—")
        self.assertEqual(fmt(0.5), "0.500")
        self.assertEqual(fmt(0.5, 1), "0.5")
        self.assertEqual(fmt_delta(None), "—")
        self.assertEqual(fmt_delta(0.125), "+0.125")
        self.assertEqual(fmt_delta(-0.125), "-0.125")

    def test_numeric_rows_and_series_tolerate_garbage(self):
        header = ["step", "avg_reward", "note"]
        rows = numeric_rows(header, [{"step": "5", "avg_reward": "", "note": "x"},
                                     {"step": "10", "avg_reward": "0.25", "note": "oops"}])
        self.assertEqual(rows[0]["avg_reward"], None)
        self.assertEqual(rows[1]["avg_reward"], 0.25)
        self.assertEqual(rows[0]["note"], None)
        self.assertEqual(series(rows, "avg_reward"), [None, 0.25])

    def test_first_last_skips_blanks(self):
        self.assertEqual(first_last([None, 1.0, None, 3.0]), (1.0, 3.0))
        self.assertEqual(first_last([None, None]), (None, None))

    def test_trend_endpoints_average_a_window_instead_of_a_single_step(self):
        """单点太抖：长序列必须用首/末窗口均值。

        真实数字：`mini_rule_seed42` 第 5 步 correct_rate=0.75、第 1000 步=0.50，
        只看首末两点会得出"训练在退步 -0.25"的结论（决策树随即误报），
        而两端各 10 个点的均值都在 0.31~0.33 附近。
        """

        noisy = [0.75] + [0.32] * 27 + [0.50]
        self.assertEqual(trend_window(len(noisy)), 5)
        start, end = trend_endpoints(noisy)
        self.assertAlmostEqual(start, sum(noisy[:5]) / 5)
        self.assertAlmostEqual(end, sum(noisy[-5:]) / 5)
        single_point_gap = abs(noisy[0] - noisy[-1])
        self.assertAlmostEqual(single_point_gap, 0.25)
        self.assertLess(abs(end - start), single_point_gap / 3,
                        "首末窗口均值应该把单点噪声稀释掉，而不是原样喂给决策树")


class CollectRunTests(LayoutFixture):
    def test_archived_run_reads_archive_and_eval(self):
        self.make_run("mini_rule_seed42", eval_data=eval_payload())
        run = collect_run("mini_rule_seed42", runs_dir=self.runs_dir, log_dir=self.log_dir,
                          registry_row=None)
        self.assertEqual(run["evidence"], "archived")
        self.assertEqual(run["archive_status"], "complete")
        self.assertEqual(len(run["rows"]), 2)
        self.assertEqual(run["warnings"], [])
        self.assertIsNotNone(run["eval"])

    def test_logs_only_run_is_flagged(self):
        self.make_run("mini_rule_seed42", archive=False, eval_data=eval_payload())
        run = collect_run("mini_rule_seed42", runs_dir=self.runs_dir, log_dir=self.log_dir,
                          registry_row=None)
        self.assertEqual(run["evidence"], "logs-only")
        self.assertTrue(any("not archived" in warning for warning in run["warnings"]))

    def test_missing_run_reports_no_evidence(self):
        run = collect_run("nope", runs_dir=self.runs_dir, log_dir=self.log_dir, registry_row=None)
        self.assertEqual(run["evidence"], "missing")
        self.assertEqual(run["metrics_path"], None)
        self.assertTrue(any("no held-out evaluation" in warning for warning in run["warnings"]))

    def test_single_step_run_is_flagged_as_trendless(self):
        write_metrics(self.log_dir / "single_metrics.csv")
        header = METRIC_HEADER
        with (self.log_dir / "single_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerow({"step": 5, "avg_reward": 0.1, "format_rate": 0.1, "correct_rate": 0.1,
                             "avg_completion_tokens": 100, "valid_groups": 1, "all_correct_groups": 0,
                             "all_wrong_groups": 1})
        run = collect_run("single", runs_dir=self.runs_dir, log_dir=self.log_dir, registry_row=None)
        self.assertTrue(any("no trend" in warning for warning in run["warnings"]))


class BuildRowTests(LayoutFixture):
    def test_row_carries_registry_knobs_and_gains(self):
        self.make_run("mini_rule_seed42", eval_data=eval_payload())
        rows = self.rows(["mini_rule_seed42"])
        row = rows[0]
        self.assertEqual(row["reward_mode"], "rule")
        self.assertAlmostEqual(row["reward_gain"], 0.2)
        self.assertAlmostEqual(row["correct_rate_gain"], 0.1)
        self.assertAlmostEqual(row["format_rate_gain"], 0.1)
        self.assertEqual(row["completion_tokens_last"], 80.0)
        self.assertEqual(row["eval_greedy_accuracy"], 0.30)
        self.assertEqual(row["eval_hard_pass1"], 0.12)
        self.assertEqual(row["evidence"], "archived")

    def test_acr_is_a_share_of_groups(self):
        self.make_run("mini_rule_seed42", eval_data=eval_payload(), groups=(3, 1, 2))
        row = self.rows(["mini_rule_seed42"])[0]
        self.assertAlmostEqual(row["acr_wrong_last"], 2 / 6)
        self.assertAlmostEqual(row["acr_correct_last"], 1 / 6)

    def test_acr_is_none_without_group_counters(self):
        self.make_run("mini_rule_seed42", eval_data=eval_payload(), groups=(None, None, None))
        row = self.rows(["mini_rule_seed42"])[0]
        self.assertIsNone(row["acr_wrong_last"])

    def test_unregistered_run_has_empty_knobs(self):
        self.make_run("unregistered", eval_data=eval_payload(), archive=False)
        row = self.rows(["unregistered"])[0]
        self.assertEqual(row["reward_mode"], "")
        self.assertEqual(row["hypothesis"], "")
        self.assertEqual(row["evidence"], "logs-only")


class CrossRunTests(LayoutFixture):
    def test_heldout_gain_is_relative_to_the_reference_arm(self):
        self.make_run("mini_rule_seed42", eval_data=eval_payload())
        self.make_run("mini_rm_seed42", eval_data=eval_payload(greedy_accuracy=0.20))
        rows = {row["run_tag"]: row for row in self.rows(["mini_rule_seed42", "mini_rm_seed42"])}
        self.assertAlmostEqual(rows["mini_rule_seed42"]["heldout_gain"], 0.0)
        self.assertAlmostEqual(rows["mini_rm_seed42"]["heldout_gain"], -0.1)

    def test_reference_arm_rows_have_no_heldout_gain_when_baseline_is_missing_eval(self):
        self.make_run("mini_rule_seed42")
        self.make_run("mini_rm_seed42", eval_data=eval_payload())
        rows = {row["run_tag"]: row for row in self.rows(["mini_rule_seed42", "mini_rm_seed42"])}
        self.assertIsNone(rows["mini_rm_seed42"]["heldout_gain"])
        self.assertIsNone(rows["mini_rm_seed42"]["reward_eval_gap"])

    def test_hacking_suspect_needs_reward_up_without_correctness_or_heldout_movement(self):
        self.make_run("mini_rule_seed42", eval_data=eval_payload())
        self.make_run("mini_rm_seed42", reward=(0.1, 0.9), correct=(0.0, 0.0),
                      eval_data=eval_payload(greedy_accuracy=0.30))
        rows = {row["run_tag"]: row for row in self.rows(["mini_rule_seed42", "mini_rm_seed42"])}
        self.assertTrue(rows["mini_rm_seed42"]["hacking_suspect"])
        self.assertFalse(rows["mini_rule_seed42"]["hacking_suspect"])

    def test_passk_inversion_needs_pass1_up_and_pass16_down(self):
        self.make_run("mini_rule_seed42", eval_data=eval_payload())
        self.make_run("mini_rm_seed42", eval_data=eval_payload(sampled_pass_at_1=0.45, pass_at_16=0.55))
        rows = {row["run_tag"]: row for row in self.rows(["mini_rule_seed42", "mini_rm_seed42"])}
        self.assertTrue(rows["mini_rm_seed42"]["passk_inversion"])
        self.assertFalse(rows["mini_rule_seed42"]["passk_inversion"])

    def test_add_cross_run_columns_is_idempotent(self):
        rows = [{
            "run_tag": "a", "reward_gain": 0.2, "correct_rate_gain": 0.0,
            "eval_greedy_accuracy": 0.3, "eval_pass_at_1": 0.3, "eval_pass_at_16": 0.6,
        }, {
            "run_tag": "b", "reward_gain": 0.1, "correct_rate_gain": 0.05,
            "eval_greedy_accuracy": 0.4, "eval_pass_at_1": 0.4, "eval_pass_at_16": 0.7,
        }]
        add_cross_run_columns(rows, "a")
        first_pass = [dict(row) for row in rows]
        add_cross_run_columns(rows, "a")
        self.assertEqual([dict(row) for row in rows], first_pass)


class DecisionTreeTests(unittest.TestCase):
    def base(self, **overrides):
        row = {
            "metrics_rows": 3, "reward_gain": 0.2, "correct_rate_gain": 0.0,
            "format_rate_gain": 0.0, "heldout_gain": 0.0,
            "completion_tokens_first": 100.0, "completion_tokens_last": 100.0,
            "acr_wrong_last": 0.2,
        }
        row.update(overrides)
        return row

    def test_insufficient_data(self):
        self.assertIn("数据不足", decision_tree(self.base(metrics_rows=1)))

    def test_reward_flat(self):
        self.assertIn("查 reward 范围", decision_tree(self.base(reward_gain=0.0)))

    def test_train_correctness_up_with_flat_heldout_is_a_gap_not_a_win(self):
        verdict = decision_tree(self.base(correct_rate_gain=0.05, heldout_gain=0.0))
        self.assertIn("reward-eval gap", verdict)

    def test_train_and_heldout_up_together_is_a_signal_candidate(self):
        verdict = decision_tree(self.base(correct_rate_gain=0.05, heldout_gain=0.03))
        self.assertIn("真实任务信号候选", verdict)

    def test_train_correctness_up_without_an_eval_is_deferred(self):
        verdict = decision_tree(self.base(correct_rate_gain=0.05, heldout_gain=None))
        self.assertIn("还没有 held-out 数字", verdict)

    def test_heldout_up_is_a_signal_candidate(self):
        self.assertIn("真实任务信号候选", decision_tree(self.base(heldout_gain=0.02)))

    def test_format_absorption(self):
        self.assertIn("格式吸收 reward", decision_tree(self.base(format_rate_gain=0.2)))

    def test_length_hacking(self):
        self.assertIn("长度偏置", decision_tree(self.base(completion_tokens_last=160.0)))

    def test_capacity_branch(self):
        self.assertIn("容量", decision_tree(self.base(acr_wrong_last=0.8)))

    def test_fallback_branch(self):
        self.assertIn("奖励分辨率", decision_tree(self.base()))

    def test_mix_reward_is_not_read_as_a_flat_reward(self):
        verdict = decision_tree(self.base(reward_mode="mix", reward_gain=0.0))
        self.assertNotIn("查 reward 范围", verdict)
        self.assertIn("标准化", verdict)

    def test_rm_arm_heldout_drop_is_reward_hacking_not_length_bias(self):
        # 回归：mini_rm_seed42 实测 reward 升、正确率降、held-out −0.263，
        # 旧排序会因为长度窗口均值上升而误报"长度偏置 / 过长 hack"。
        verdict = decision_tree(self.base(
            reward_mode="rm", reward_gain=0.327, correct_rate_gain=-0.208,
            heldout_gain=-0.263, completion_tokens_last=110.0,
        ))
        self.assertIn("背离", verdict)
        self.assertNotIn("长度偏置", verdict)

    def test_mix_arm_heldout_drop_blames_the_rm_component(self):
        verdict = decision_tree(self.base(
            reward_mode="mix", reward_gain=0.0, correct_rate_gain=0.142, heldout_gain=-0.080,
        ))
        self.assertIn("污染", verdict)

    def test_rule_arm_heldout_drop_is_a_plain_negative_result(self):
        verdict = decision_tree(self.base(reward_mode="rule", heldout_gain=-0.05))
        self.assertIn("负结果", verdict)
        self.assertNotIn("污染", verdict)

    def test_heldout_drop_wording_never_claims_significance(self):
        # 回归：本函数不做配对检验，所以措辞不能出现"显著"——显著性是 paired_compare 的活。
        # 反例：mini_rule_dynamic_seed42 对参考臂是 −0.005，正落在噪声地板内（report §4.7）。
        for mode in ("rule", "rm", "mix"):
            verdict = decision_tree(self.base(reward_mode=mode, heldout_gain=-0.005))
            self.assertNotIn("显著下降", verdict)     # 方向 ≠ 显著；显著性留给 paired_compare
            self.assertNotIn("显著上升", verdict)
            self.assertIn("paired_compare", verdict)


class RewardCellTests(unittest.TestCase):
    def test_mix_reward_cells_are_marked_not_applicable(self):
        row = {"reward_mode": "mix", "reward_first": -0.0, "reward_last": 0.0, "reward_gain": 0.0}
        self.assertIn("n/a", reward_trend_cell(row))
        self.assertIn("n/a", reward_gain_cell(row))

    def test_rule_reward_cells_keep_the_numbers(self):
        row = {"reward_mode": "rule", "reward_first": 0.317, "reward_last": 0.542, "reward_gain": 0.225}
        self.assertIn("0.317", reward_trend_cell(row))
        self.assertIn("+0.225", reward_gain_cell(row))


class LengthSampleTests(LayoutFixture):
    def test_reads_eval_jsonl(self):
        self.make_run("mini_rule_seed42", eval_data=eval_payload())
        buckets = length_samples(self.runs_dir / "mini_rule_seed42", None)
        self.assertEqual(len(buckets["sampled"]), 5)

    def test_falls_back_to_eval_json_samples(self):
        payload = eval_payload(samples=[{"sampled": [{"tokens": 10}, {"tokens": 20}], "greedy": {"tokens": 30}}])
        (self.runs_dir / "tag").mkdir(parents=True, exist_ok=True)
        buckets = length_samples(self.runs_dir / "tag", payload)
        self.assertEqual(sorted(buckets["sampled"]), [10, 20])
        self.assertEqual(buckets["greedy"], [30])

    def test_absent_data_is_empty(self):
        self.assertEqual(length_samples(self.runs_dir / "nope", None), {})


class RenderTests(LayoutFixture):
    def test_csv_header_matches_the_declared_columns(self):
        self.make_run("mini_rule_seed42", eval_data=eval_payload())
        rows = self.rows(["mini_rule_seed42"])
        path = self.out_dir / "summary.csv"
        write_csv(rows, path)
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(list(reader.fieldnames), CSV_COLUMNS)
            record = next(reader)
        self.assertEqual(record["run_tag"], "mini_rule_seed42")
        self.assertEqual(record["eval_greedy_accuracy"], "0.3")
        self.assertTrue(record["hypothesis"].startswith("rule signal baseline"))
        path.unlink()

    def test_matrix_marks_missing_and_unarchived_runs(self):
        rows = [{
            "run_tag": "mini_rule_seed42", "evidence": "logs-only", "eval_greedy_accuracy": 0.3,
            "eval_pass_at_1": 0.3, "eval_pass_at_16": 0.6, "eval_format_rate": 0.9,
            "eval_mean_tokens": 180.0, "eval_easy_greedy": 0.5, "eval_medium_greedy": 0.3,
            "eval_hard_greedy": 0.1,
        }]
        text = "\n".join(render_matrix(rows))
        self.assertIn("rule GRPO", text)
        self.assertIn("(`mini_rule_seed42`) *", text)  # logs-only marker
        self.assertIn("RM-only", text)
        self.assertIn("| — | — |", text)  # missing runs still have a row
        self.assertEqual(len([line for line in text.splitlines() if line.startswith("|")]),
                         2 + len(MATRIX_ROWS))
        for _label, tag in MATRIX_ROWS:
            self.assertIn(tag, text)

    def test_matrix_flags_evals_with_fewer_questions(self):
        """20 题的 eval 和 400 题并排时必须有可见警告（SOP §10.1）。

        真实事故：pilot 的 `math_cold_seed42` 只跑了 20 题（greedy 0.000），却出现在
        400 题的矩阵里当"官方/冷启动 SFT"基线，很容易被当成可比的数字引用。
        """

        base = {"evidence": "archived", "eval_format_rate": 1.0, "eval_mean_tokens": 115.0,
                "eval_easy_greedy": 0.5, "eval_medium_greedy": 0.1, "eval_hard_greedy": 0.4,
                "eval_pass_at_1": 0.3, "eval_pass_at_16": 0.5}
        rows = [
            dict(base, run_tag="mini_rule_seed42", eval_greedy_accuracy=0.4, eval_rows_evaluated=400),
            dict(base, run_tag="math_cold_cot_seed42", eval_greedy_accuracy=0.37, eval_rows_evaluated=400),
            dict(base, run_tag="math_cold_seed42", eval_greedy_accuracy=0.0, eval_rows_evaluated=20),
        ]
        text = "\n".join(render_matrix(rows))
        self.assertIn("| 题数 |", text)
        self.assertIn("(`math_cold_seed42`) ⚠️", text)
        self.assertNotIn("(`mini_rule_seed42`) ⚠️", text)
        self.assertIn("题数少于本表基准（400 题）", text)

    def test_matrix_includes_the_cot_v2_cold_start(self):
        """RL 的起点是 CoT v2 冷启动，它必须出现在 §10.3 矩阵里。"""

        tags = [tag for _label, tag in MATRIX_ROWS]
        self.assertIn("math_cold_cot_seed42", tags)
        self.assertLess(tags.index("math_cold_cot_seed42"), tags.index("mini_rule_seed42"))

    def test_seed_aggregate_reports_mean_and_range(self):
        rows = [
            {"run_tag": "a", "seed_group": "g|rule", "seed": "42", "eval_greedy_accuracy": 0.30,
             "eval_pass_at_1": 0.30},
            {"run_tag": "b", "seed_group": "g|rule", "seed": "7", "eval_greedy_accuracy": 0.40,
             "eval_pass_at_1": 0.44},
        ]
        text = "\n".join(render_seed_aggregate(rows))
        self.assertIn("0.350 [0.300, 0.400]", text)
        self.assertIn("42, 7", text)

    def test_seed_aggregate_admits_a_single_seed(self):
        rows = [{"run_tag": "a", "seed_group": "g|rule", "seed": "42",
                 "eval_greedy_accuracy": 0.30, "eval_pass_at_1": 0.30}]
        text = "\n".join(render_seed_aggregate(rows))
        self.assertIn("无", text)
        self.assertIn("§10.4", text)

    def test_summary_md_has_every_declared_section(self):
        self.make_run("mini_rule_seed42", eval_data=eval_payload())
        self.make_run("mini_rm_seed42", eval_data=eval_payload(greedy_accuracy=0.2))
        rows = self.rows(["mini_rule_seed42", "mini_rm_seed42"])
        text = render_summary_md(rows, notes=["reference arm: mini_rule_seed42"], figures=[],
                                 data_sources={"registry": "configs/experiment_registry.csv",
                                               "runs": "runs", "logs": "logs",
                                               "heldout": "dataset/math_heldout.jsonl"})
        for heading in ["## 1. 训练动态", "## 2. Held-out 评估矩阵（SOP §10.3）", "## 3. 按难度的 held-out 正确率",
                        "## 4. seed 汇总（SOP §10.4）", "## 5. 图表清单（caption 全文）", "## 6. 警告与缺口"]:
            self.assertIn(heading, text)
        self.assertIn("reference arm: mini_rule_seed42", text)
        self.assertIn("未生成图表", text)

    def test_caption_names_every_required_dimension(self):
        rows = [{"model": "MiniMind-64M", "seed": "42", "eval_num_samples": 16.0,
                 "eval_temperature": 0.8}]
        caption = figure_caption(rows, "source: test")
        for fragment in ["MiniMind-64M", "dataset/math_rlvr_train.jsonl", "dataset/math_heldout.jsonl",
                         "seeds: 42", "16 completions per problem", "temperature 0.8",
                         "length convention", "source: test"]:
            self.assertIn(fragment, caption)

    def test_caption_handles_a_run_without_an_eval(self):
        caption = figure_caption([{"model": "MiniMind-64M"}], "source: test")
        self.assertIn("sampling: n/a", caption)


class FigureTests(LayoutFixture):
    def test_figures_are_written_with_captions(self):
        try:
            import matplotlib  # noqa: F401
        except Exception:
            self.skipTest("matplotlib not installed")
        self.make_run("mini_rule_seed42", eval_data=eval_payload())
        self.make_run("mini_rm_seed42", eval_data=eval_payload(greedy_accuracy=0.2))
        self.assertEqual(self.summarize("--quiet"), 0)
        figures = sorted((self.out_dir / "figures").glob("fig*.png"))
        self.assertEqual(len(figures), 6)
        manifest = json.loads((self.out_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["figures"]), 6)
        for figure in manifest["figures"]:
            self.assertIn("length convention", figure["caption"])

    def test_missing_eval_reports_note_and_still_writes_figures(self):
        try:
            import matplotlib  # noqa: F401
        except Exception:
            self.skipTest("matplotlib not installed")
        self.make_run("mini_rule_seed42")
        self.assertEqual(self.summarize("--quiet"), 0)
        manifest = json.loads((self.out_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(any("not plotted" in note for note in manifest["notes"]))
        self.assertTrue((self.out_dir / "figures" / "fig1_training_dynamics.png").is_file())

    def test_matplotlib_absence_is_not_fatal(self):
        self.make_run("mini_rule_seed42", eval_data=eval_payload())
        with mock.patch("scripts.summarize_runs.ensure_mpl", return_value=None):
            self.assertEqual(self.summarize("--quiet"), 0)
        manifest = json.loads((self.out_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["figures"], [])
        self.assertTrue(any("matplotlib" in note for note in manifest["notes"]))

    def test_empty_state_writes_placeholder_figures_and_a_note(self):
        if not _has_mpl():
            self.skipTest("matplotlib not installed")
        self.assertEqual(self.summarize("--run-tag", "mini_rule_seed42", "--quiet"), 0)
        manifest = json.loads((self.out_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(any("placeholder" in note for note in manifest["notes"]))
        self.assertEqual(manifest["missing_eval"], ["mini_rule_seed42"])

    def test_no_figures_flag_skips_the_directory(self):
        self.make_run("mini_rule_seed42", eval_data=eval_payload())
        self.assertEqual(self.summarize("--no-figures", "--quiet"), 0)
        self.assertFalse((self.out_dir / "figures").exists())

    def test_gap_figure_notes_unplotted_runs(self):
        plt = __import__("matplotlib.pyplot", fromlist=["pyplot"]) if _has_mpl() else None
        if plt is None:
            self.skipTest("matplotlib not installed")
        rows = [{"run_tag": "with_eval", "reward_gain": 0.2, "heldout_gain": 0.05},
                {"run_tag": "no_eval", "reward_gain": 0.2, "heldout_gain": None}]
        notes = fig_reward_eval_gap(plt, rows, {"runs": {}, "runs_dir": self.root}, self.out_dir, "caption", 80)
        self.assertTrue(any("no_eval" in note for note in notes))


def _has_mpl() -> bool:
    try:
        import matplotlib  # noqa: F401
    except Exception:
        return False
    return True


class CliTests(LayoutFixture):
    def test_summary_writes_all_three_outputs(self):
        self.make_run("mini_rule_seed42", eval_data=eval_payload())
        self.assertEqual(self.summarize("--run-tag", "mini_rule_seed42", "--quiet"), 0)
        for name in ("summary.md", "summary.csv", "manifest.json"):
            self.assertTrue((self.out_dir / name).is_file(), name)
        manifest = json.loads((self.out_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([entry["run_tag"] for entry in manifest["runs"]][0], "mini_rule_seed42")
        self.assertEqual(manifest["missing_eval"], [])

    def test_require_eval_fails_when_an_arm_lacks_a_heldout_number(self):
        self.make_run("mini_rule_seed42", eval_data=eval_payload())
        self.assertEqual(
            self.summarize("--run-tag", "mini_rule_seed42", "--quiet", "--require-eval"), 0)
        self.make_run("mini_rm_seed42")  # metrics only, no evaluation yet
        rc = self.summarize("--run-tag", "mini_rule_seed42", "--run-tag", "mini_rm_seed42",
                            "--quiet", "--require-eval")
        self.assertEqual(rc, 1)

    def test_default_tag_list_is_the_registry_plus_the_cold_start(self):
        rows = self.rows()
        tags = [row["run_tag"] for row in rows]
        _header, registry_rows = load_registry(REGISTRY)
        self.assertEqual(tags, [row["run_tag"] for row in registry_rows] + list(EXTRA_TAGS))
        self.assertTrue(all(row["evidence"] == "missing" for row in rows))

    def test_run_tag_subset(self):
        self.make_run("mini_rule_seed42", eval_data=eval_payload())
        self.make_run("mini_rm_seed42", eval_data=eval_payload())
        rows = self.rows(["mini_rm_seed42"])
        self.assertEqual([row["run_tag"] for row in rows], ["mini_rm_seed42"])

    def test_arm_knobs_do_not_include_the_seed(self):
        self.assertNotIn("seed", ARM_KNOBS)


if __name__ == "__main__":
    unittest.main()
