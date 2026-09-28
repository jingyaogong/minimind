import re
import types
import unittest

import torch

import trainer.train_grpo as grpo
from trainer.math_verifier import ANSWER_MARKER


class FakeRewardModel:
    """Stands in for LMForRewardModel without loading a 1.8B checkpoint."""

    def __init__(self, score=0.0):
        self.score = score
        self.calls = []

    def get_score(self, messages, response):
        self.calls.append({"messages": messages, "response": response})
        return self.score


def set_args(**overrides):
    defaults = {
        "reward_mode": "rule",
        "reward_marker": ANSWER_MARKER,
        "verdict_tier": "binary",
        "num_generations": 2,
        "mix_alpha": 0.5,
        "mix_beta": 0.5,
    }
    defaults.update(overrides)
    grpo.args = types.SimpleNamespace(**defaults)
    return grpo.args


CORRECT = f"<think>先算 17*20 再算 17*3，相加。</think>\n{ANSWER_MARKER} 391"
WRONG = f"<think>随便算算。</think>\n{ANSWER_MARKER} 392"
NO_ANSWER = "<think>先看一下题目。</think>\n我算不出来。"


class RuleRewardTests(unittest.TestCase):
    def test_scores_follow_the_verdict(self):
        scores, stats = grpo.rule_rewards([CORRECT, WRONG, NO_ANSWER, CORRECT], ["391", "391", "391", "391"])
        self.assertEqual(scores, [1.0, 0.0, -0.5, 1.0])
        self.assertEqual(stats["correct"], 2)
        self.assertEqual(stats["incorrect"], 1)
        self.assertEqual(stats["no_answer"], 1)
        self.assertEqual(stats["correct_flags"], [True, False, False, True])

    def test_rates_are_reported(self):
        _, stats = grpo.rule_rewards([CORRECT, WRONG], ["391", "391"])
        self.assertAlmostEqual(stats["correct_rate"], 0.5)
        self.assertAlmostEqual(stats["format_rate"], 1.0)
        self.assertAlmostEqual(stats["no_answer_rate"], 0.0)
        self.assertAlmostEqual(stats["mean_rule_reward"], 0.5)

    def test_tiered_reward_differs_from_binary(self):
        binary, _ = grpo.rule_rewards([WRONG, NO_ANSWER], ["391", "391"], tier="binary")
        tiered, _ = grpo.rule_rewards([WRONG, NO_ANSWER], ["391", "391"], tier="tiered")
        self.assertEqual(binary, [0.0, -0.5])
        self.assertEqual(tiered, [-0.5, -1.0])

    def test_off_by_group_ground_truth_is_rejected(self):
        # The classic bug: B ground truths zipped with B*G completions.
        with self.assertRaises(ValueError) as ctx:
            grpo.rule_rewards([CORRECT, WRONG, CORRECT, WRONG], ["391", "391"])
        self.assertIn("length mismatch", str(ctx.exception))

    def test_custom_marker(self):
        response = "推理过程略。#### 391"
        scores, stats = grpo.rule_rewards([response], ["391"], marker="####")
        self.assertEqual(scores, [1.0])
        self.assertTrue(stats["correct_flags"][0])

    def test_empty_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            grpo.rule_rewards([], [])


def reference_official_reward(prompts, responses, reward_model, num_generations):
    """The pre-change reward, copied verbatim from the original loop.

    Kept in the test suite as the regression reference: if someone edits
    ``official_rm_rewards`` the diff shows up here instead of silently changing
    the parity baseline.
    """

    def rep_penalty(text, n=3, cap=0.5):
        toks = re.findall(r"\w+|[^\w\s]", text.lower())
        grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

    rewards = []
    for i in range(len(prompts)):
        for j in range(num_generations):
            response = responses[i * num_generations + j]
            prompt = prompts[i]
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            answer = response
            score = 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            if '</think>' in response:
                thinking_content, answer_content = response.split('</think>', 1)
                score += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                score += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer_content.strip()
            score -= rep_penalty(answer)
            score += reward_model.get_score(messages, answer)
            rewards.append(score)
    return rewards


PROMPT = f"<|im_start|>user\n请计算 17*23，并在最后一行以‘{ANSWER_MARKER} 数字’输出。<|im_end|>\n<|im_start|>assistant\n"


class OfficialRewardRegressionTests(unittest.TestCase):
    def test_matches_the_pre_change_formula(self):
        responses = [
            CORRECT,
            WRONG,
            NO_ANSWER,
            "短",
            "<think>" + "啰嗦的思考" * 80 + "</think>\n" + ANSWER_MARKER + " 391",
            "<think>重复重复重复重复重复重复重复重复</think>\n" + ANSWER_MARKER + " 1",
            "<think>两个闭合</think>中段</think>\n" + ANSWER_MARKER + " 391",
            "x" * 900,
        ]
        prompts = [PROMPT] * (len(responses) // 2)
        rm = FakeRewardModel(score=1.25)
        set_args(reward_mode="rm", num_generations=2)
        actual, _ = grpo.official_rm_rewards(prompts, responses, rm)
        expected = reference_official_reward(prompts, responses, FakeRewardModel(score=1.25), 2)
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            self.assertAlmostEqual(got, want, places=9)

    def test_heuristic_terms_still_present(self):
        set_args(reward_mode="rm", num_generations=2)
        # "短" is under 20 chars -> -0.5 even though the RM is neutral.
        scores, _ = grpo.official_rm_rewards([PROMPT], ["短", CORRECT], FakeRewardModel(score=0.0))
        self.assertAlmostEqual(scores[0], -0.5)
        # whole response +0.5, thinking +1.0, single </think> +0.25, no repetition, RM 0.0
        self.assertAlmostEqual(scores[1], 0.5 + 1.0 + 0.25)

    def test_pure_rm_drops_the_heuristics(self):
        rm = FakeRewardModel(score=0.75)
        scores, _ = grpo.pure_rm_rewards([PROMPT], ["短", CORRECT], rm)
        self.assertEqual(scores, [0.75, 0.75])
        # the RM sees the post-</think> answer, not the whole completion
        self.assertEqual(rm.calls[1]["response"], f"{ANSWER_MARKER} 391")
        self.assertEqual(rm.calls[0]["messages"][0]["role"], "user")


class MixRewardTests(unittest.TestCase):
    def test_weights_are_applied_after_standardisation(self):
        set_args(num_generations=2, mix_alpha=1.0, mix_beta=0.0)
        rule = [1.0, 0.0, 0.0, 1.0]
        rm = [0.3, -0.2, 0.9, 0.4]
        mixed, stats = grpo.mix_rewards(rule, rm, 2, 1.0, 0.0)
        expected = grpo.normalize_group(torch.tensor(rule).view(-1, 2)).reshape(-1).tolist()
        for got, want in zip(mixed, expected):
            self.assertAlmostEqual(got, want, places=6)
        self.assertEqual(stats["component_degenerate_groups"], 0)

    def test_component_scale_cannot_change_the_mix(self):
        rule = [1.0, 0.0, 0.5, 0.25, 0.0, 1.0]
        rm = [0.3, -0.2, 0.9, 0.4, 0.1, 0.2]
        base, _ = grpo.mix_rewards(rule, rm, 3, 0.5, 0.5)
        shifted = [1000.0 * value + 500.0 for value in rm]
        scaled, _ = grpo.mix_rewards(rule, shifted, 3, 0.5, 0.5)
        for got, want in zip(scaled, base):
            # float32 round-trip through two different normalisations
            self.assertAlmostEqual(got, want, places=4)

    def test_degenerate_component_is_counted_not_nan(self):
        rule = [1.0, 0.0, 0.25, 0.25]  # second group is flat
        rm = [0.3, -0.2, 0.9, 0.4]
        mixed, stats = grpo.mix_rewards(rule, rm, 2, 0.5, 0.5)
        self.assertEqual(stats["component_degenerate_groups"], 1)
        self.assertFalse(any(value != value for value in mixed))
        # a flat rule group contributes exactly zero; the RM half still carries signal
        rule_only, _ = grpo.mix_rewards(rule, rm, 2, 1.0, 0.0)
        self.assertEqual(rule_only[2:], [0.0, 0.0])
        self.assertAlmostEqual(mixed[2], 0.5 * 0.0 + 0.5 * 1.0, places=5)

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            grpo.mix_rewards([1.0, 0.0], [1.0], 2, 0.5, 0.5)

    def test_indivisible_group_size_is_rejected(self):
        with self.assertRaises(ValueError):
            grpo.mix_rewards([1.0, 0.0, 0.5], [1.0, 0.0, 0.5], 2, 0.5, 0.5)


class NormalizeGroupTests(unittest.TestCase):
    def test_flat_group_maps_to_zero_without_nan(self):
        flat = torch.tensor([[2.0, 2.0], [1.0, 2.0]])
        out = grpo.normalize_group(flat)
        self.assertEqual(out[0].tolist(), [0.0, 0.0])
        self.assertFalse(torch.isnan(out).any())

    def test_standardisation_statistics(self):
        out = grpo.normalize_group(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        self.assertAlmostEqual(out.mean().item(), 0.0, places=6)
        self.assertAlmostEqual(out.std(unbiased=False).item(), 1.0, places=5)


class DispatchTests(unittest.TestCase):
    def test_rule_mode_uses_the_verifier_and_refuses_a_reward_model(self):
        set_args(reward_mode="rule")
        scores, stats = grpo.calculate_rewards(
            [PROMPT], [CORRECT, WRONG], gt_for_rollouts=["391", "391"], reward_model=None
        )
        self.assertEqual(scores, [1.0, 0.0])
        self.assertIn("verdicts", stats)
        with self.assertRaises(AssertionError):
            grpo.calculate_rewards([PROMPT], [CORRECT, WRONG],
                                   gt_for_rollouts=["391", "391"], reward_model=FakeRewardModel())

    def test_rule_mode_requires_ground_truth(self):
        set_args(reward_mode="rule")
        with self.assertRaises(ValueError) as ctx:
            grpo.calculate_rewards([PROMPT], [CORRECT, WRONG])
        self.assertIn("ground-truth", str(ctx.exception))

    def test_rm_mode_rewards_from_rm_but_still_reports_correctness(self):
        set_args(reward_mode="rm")
        rm = FakeRewardModel(score=0.5)
        scores, stats = grpo.calculate_rewards(
            [PROMPT], [CORRECT, WRONG], gt_for_rollouts=["391", "391"], reward_model=rm
        )
        self.assertAlmostEqual(scores[0], 0.5 + 1.0 + 0.25 + 0.5)
        self.assertEqual(len(rm.calls), 2)
        # the verifier still runs, so an RM arm is comparable on accuracy
        self.assertAlmostEqual(stats["correct_rate"], 0.5)
        self.assertEqual(stats["correct_flags"], [True, False])

    def test_mix_mode_combines_standardised_components(self):
        set_args(reward_mode="mix", mix_alpha=0.3, mix_beta=0.7, num_generations=2)
        rm = FakeRewardModel(score=0.9)
        scores, stats = grpo.calculate_rewards(
            [PROMPT], [CORRECT, WRONG], gt_for_rollouts=["391", "391"], reward_model=rm
        )
        rule_scores, _ = grpo.rule_rewards([CORRECT, WRONG], ["391", "391"])
        rm_scores, _ = grpo.pure_rm_rewards([PROMPT], [CORRECT, WRONG], rm)
        expected, _ = grpo.mix_rewards(rule_scores, rm_scores, 2, 0.3, 0.7)
        for got, want in zip(scores, expected):
            self.assertAlmostEqual(got, want, places=6)
        self.assertAlmostEqual(stats["mix_alpha"], 0.3)
        self.assertAlmostEqual(stats["correct_rate"], 0.5)

    def test_official_mode_ignores_ground_truth(self):
        set_args(reward_mode="official")
        scores, stats = grpo.calculate_rewards([PROMPT], [CORRECT, WRONG], reward_model=FakeRewardModel(score=0.0))
        self.assertEqual(len(scores), 2)
        self.assertNotIn("correct_flags", stats)

    def test_unknown_mode_is_rejected(self):
        set_args(reward_mode="nonsense")
        with self.assertRaises(ValueError):
            grpo.calculate_rewards([PROMPT], [CORRECT, WRONG], gt_for_rollouts=["391", "391"])


class MetricFieldsTests(unittest.TestCase):
    def test_csv_schema_matches_the_sop(self):
        self.assertEqual(grpo.METRIC_FIELDS, [
            "timestamp", "step", "seed", "avg_reward", "format_rate", "correct_rate",
            "no_answer_rate", "avg_completion_tokens", "median_completion_tokens",
            "kl_ref", "entropy", "adv_std", "policy_loss", "valid_groups",
            "all_correct_groups", "all_wrong_groups", "filtered_fraction",
        ])


if __name__ == "__main__":
    unittest.main()
