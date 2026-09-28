import unittest

from trainer.math_verifier import (
    ANSWER_MARKER,
    extract_answer_text,
    normalize_number,
    reward_tier,
    split_think,
    verdict,
)


class SplitThinkTests(unittest.TestCase):
    def test_closed_think_is_split(self):
        thinking, final, closed = split_think("<think>先算乘法</think>\n答案是 391")
        self.assertEqual(thinking, "先算乘法")
        self.assertEqual(final, "答案是 391")
        self.assertTrue(closed)

    def test_final_only_response_is_allowed_but_not_closed(self):
        thinking, final, closed = split_think("答案是 391")
        self.assertEqual(thinking, "")
        self.assertEqual(final, "答案是 391")
        self.assertFalse(closed)

    def test_unclosed_think_cannot_hide_an_answer(self):
        thinking, final, closed = split_think("<think>答案是 391")
        self.assertEqual((thinking, final, closed), ("", "", False))

    def test_last_closed_block_determines_final_text(self):
        _, final, closed = split_think(
            "<think>第一段</think>中间<think>第二段</think>答案是 391"
        )
        self.assertEqual(final, "答案是 391")
        self.assertTrue(closed)


class NormalizeNumberTests(unittest.TestCase):
    def test_equivalent_decimal_forms(self):
        self.assertEqual(normalize_number("59"), "59")
        self.assertEqual(normalize_number("59.0"), "59")
        self.assertEqual(normalize_number("59.000"), "59")

    def test_integer_zero_and_signs(self):
        self.assertEqual(normalize_number("-0"), "0")
        self.assertEqual(normalize_number("+3"), "3")
        self.assertEqual(normalize_number("-3"), "-3")
        self.assertEqual(normalize_number("1000"), "1000")

    def test_fullwidth_digits_and_valid_thousands_separator(self):
        self.assertEqual(normalize_number("３９１"), "391")
        self.assertEqual(normalize_number("１，０００.０"), "1000")
        self.assertEqual(normalize_number("1,000"), "1000")

    def test_invalid_number_grammar(self):
        for value in (None, "", "1,00", "1e3", "三", "59元", "1/2", ".5"):
            with self.subTest(value=value):
                self.assertIsNone(normalize_number(value))

    def test_many_integer_round_trips(self):
        for number in range(-100, 100):
            with self.subTest(number=number):
                self.assertEqual(normalize_number(str(number)), str(number))


class ExtractionTests(unittest.TestCase):
    def test_marker_in_thinking_is_not_answer(self):
        response = "<think>答案是 391</think>\n最终没有格式"
        self.assertIsNone(extract_answer_text(response))

    def test_last_marker_wins_and_only_first_line_is_read(self):
        response = "答案是 1\n答案是 391\n附加说明"
        self.assertEqual(extract_answer_text(response), "391")

    def test_custom_marker(self):
        self.assertEqual(extract_answer_text("<think>x</think>\n#### 42", "####"), "42")

    def test_empty_marker_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_answer_text("答案是 1", "")


class VerdictTests(unittest.TestCase):
    def test_correct_and_incorrect(self):
        self.assertEqual(verdict("<think>x</think>\n答案是 59", "59")["status"], "correct")
        self.assertEqual(verdict("<think>x</think>\n答案是 1959", "59")["status"], "incorrect")

    def test_no_answer_cases(self):
        cases = (
            "没有最终答案",
            "<think>答案是 59",
            "<think>答案是 59</think>\n没有 marker",
        )
        for response in cases:
            with self.subTest(response=response):
                result = verdict(response, "59")
                self.assertEqual(result["status"], "no_answer")
                self.assertFalse(result["correct"])

    def test_invalid_ground_truth_is_not_model_incorrect(self):
        result = verdict("答案是 59", "五十九")
        self.assertEqual(result["status"], "invalid_gt")
        self.assertFalse(result["correct"])

    def test_result_contains_auditing_fields(self):
        result = verdict("<think>x</think>\n答案是 59", "59")
        self.assertTrue(result["marker_hit"])
        self.assertTrue(result["think_closed"])
        self.assertTrue(result["format_ok"])
        self.assertEqual(result["normalized_pred"], "59")
        self.assertEqual(result["normalized_gt"], "59")


class RewardTierTests(unittest.TestCase):
    def test_binary_rewards(self):
        self.assertEqual(reward_tier({"status": "correct"}), 1.0)
        self.assertEqual(reward_tier({"status": "incorrect"}), 0.0)
        self.assertEqual(reward_tier({"status": "no_answer"}), -0.5)

    def test_tiered_rewards(self):
        self.assertEqual(reward_tier({"status": "correct"}, "tiered"), 1.0)
        self.assertEqual(reward_tier({"status": "incorrect"}, "tiered"), -0.5)
        self.assertEqual(reward_tier({"status": "no_answer"}, "tiered"), -1.0)

    def test_invalid_reward_inputs_fail_loudly(self):
        with self.assertRaises(ValueError):
            reward_tier({"status": "invalid_gt"})
        with self.assertRaises(ValueError):
            reward_tier({"status": "correct"}, "unknown")


if __name__ == "__main__":
    unittest.main()
