import unittest
from unittest.mock import patch
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

def tiny_model(vocab_size=64):
    torch.manual_seed(17)
    return MiniMindForCausalLM(MiniMindConfig(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, intermediate_size=32,
        vocab_size=vocab_size, max_position_embeddings=128)).eval()

class BooleanAttentionMaskTests(unittest.TestCase):
    def test_boolean_mask_matches_integer_mask(self):
        model = tiny_model()
        ids = torch.tensor([[0, 0, 3, 4], [0, 5, 6, 7]])
        mask = ids.ne(0)
        for cache in (False, True):
            with self.subTest(cache=cache), torch.no_grad():
                expected = model(ids, attention_mask=mask.long(), use_cache=cache)
                actual = model(ids, attention_mask=mask, use_cache=cache)
                torch.testing.assert_close(actual.logits, expected.logits)
                self.assertTrue(bool(actual.logits.isfinite().all()))
                if cache:
                    token = torch.tensor([[8], [9]])
                    extended = torch.cat((mask, torch.ones(2, 1, dtype=torch.bool)), 1)
                    actual_next = model(token, attention_mask=extended, past_key_values=actual.past_key_values)
                    expected_next = model(token, attention_mask=extended.long(), past_key_values=expected.past_key_values)
                    torch.testing.assert_close(actual_next.logits, expected_next.logits)
