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

class TopKSamplingTests(unittest.TestCase):
    def test_default_top_k_supports_small_vocabularies(self):
        result = tiny_model(16).generate(torch.tensor([[3, 4]]), max_new_tokens=2,
                                        eos_token_id=None)
        self.assertEqual(result.shape, (1, 4))
        self.assertTrue(bool((result < 16).all()))

    def test_oversized_top_k_matches_unfiltered_sampling(self):
        model = tiny_model(16)
        ids = torch.tensor([[3, 4]])
        torch.manual_seed(42)
        actual = model.generate(ids, max_new_tokens=3, top_k=100, top_p=1.0, eos_token_id=None)
        torch.manual_seed(42)
        expected = model.generate(ids, max_new_tokens=3, top_k=0, top_p=1.0, eos_token_id=None)
        self.assertTrue(torch.equal(actual, expected))
