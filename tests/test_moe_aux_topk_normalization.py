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

from model.model_minimind import MOEFeedForward

class MoEAuxNormalizationTests(unittest.TestCase):
    def test_uniform_router_has_same_auxiliary_scale_for_each_k(self):
        for k in (1, 2, 4):
            with self.subTest(k=k):
                cfg = MiniMindConfig(hidden_size=16, num_hidden_layers=1, intermediate_size=32,
                                     use_moe=True, num_experts=4, num_experts_per_tok=k,
                                     router_aux_loss_coef=0.01)
                layer = MOEFeedForward(cfg).train()
                torch.nn.init.zeros_(layer.gate.weight)
                layer(torch.ones(2, 3, 16))
                self.assertAlmostEqual(layer.aux_loss.item(), 0.01, places=6)

    def test_auxiliary_loss_keeps_router_gradients(self):
        cfg = MiniMindConfig(hidden_size=16, intermediate_size=32, use_moe=True,
                             num_experts=4, num_experts_per_tok=2)
        layer = MOEFeedForward(cfg).train()
        layer(torch.randn(2, 3, 16))
        layer.aux_loss.backward()
        self.assertTrue(bool(layer.gate.weight.grad.isfinite().all()))
        self.assertGreater(layer.gate.weight.grad.abs().sum().item(), 0)
