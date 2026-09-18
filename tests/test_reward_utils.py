import unittest

from trainer.reward_utils import rep_penalty


class StubTokenizer:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        return self.token_ids


class RepetitionPenaltyTests(unittest.TestCase):
    def test_repeated_chinese_tokens_receive_penalty(self):
        tokenizer = StubTokenizer([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])

        self.assertEqual(rep_penalty("你好你好你好你好你好", tokenizer), 0.5)
        self.assertEqual(tokenizer.calls, [("你好你好你好你好你好", False)])

    def test_repeated_english_tokens_receive_penalty_case_insensitively(self):
        tokenizer = StubTokenizer([3, 3, 3, 3])

        self.assertEqual(rep_penalty("Hello HELLO hello hello", tokenizer), 0.5)
        self.assertEqual(tokenizer.calls, [("hello hello hello hello", False)])

    def test_unique_and_short_sequences_have_no_penalty(self):
        self.assertEqual(rep_penalty("unique", StubTokenizer([1, 2, 3, 4, 5])), 0.0)
        self.assertEqual(rep_penalty("short", StubTokenizer([1, 2])), 0.0)

    def test_invalid_configuration_is_rejected(self):
        tokenizer = StubTokenizer([])

        with self.assertRaisesRegex(ValueError, "n must be greater than zero"):
            rep_penalty("text", tokenizer, n=0)
        with self.assertRaisesRegex(ValueError, "cap must be non-negative"):
            rep_penalty("text", tokenizer, cap=-0.1)


if __name__ == "__main__":
    unittest.main()
