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

class GenerationProjectionTests(unittest.TestCase):
    def test_projection_is_one_token_with_and_without_cache(self):
        for cache in (True, False):
            with self.subTest(use_cache=cache):
                model = tiny_model()
                ids = torch.tensor([[3, 4, 5, 6]])
                expected = ids.clone()
                for _ in range(3):
                    with torch.no_grad():
                        token = model(expected).logits[:, -1].argmax(-1, keepdim=True)
                    expected = torch.cat((expected, token), dim=1)
                lengths = []
                hook = model.lm_head.register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1]))
                actual = model.generate(ids, max_new_tokens=3, do_sample=False,
                                        eos_token_id=None, use_cache=cache)
                hook.remove()
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(lengths, [1, 1, 1])
