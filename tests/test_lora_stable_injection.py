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

from model.model_lora import apply_lora

class LoRAInjectionTests(unittest.TestCase):
    def test_rank_equal_to_layer_width_does_not_recurse(self):
        model = tiny_model()
        apply_lora(model, rank=16)
        adapters = [name for name, module in model.named_modules() if hasattr(module, 'lora')]
        self.assertEqual(len(adapters), 2)
        self.assertFalse(any('.lora.' in name for name in adapters))
        self.assertTrue(bool(model(torch.tensor([[3, 4]])).logits.isfinite().all()))

    def test_reapplying_does_not_replace_or_double_existing_adapters(self):
        model = tiny_model()
        apply_lora(model, rank=4)
        adapters = {name: module.lora for name, module in model.named_modules() if hasattr(module, 'lora')}
        for adapter in adapters.values():
            torch.nn.init.constant_(adapter.B.weight, 0.05)
        ids = torch.tensor([[3, 4]])
        expected = model(ids).logits.detach()
        apply_lora(model, rank=4)
        for name, module in model.named_modules():
            if name in adapters:
                self.assertIs(module.lora, adapters[name])
        torch.testing.assert_close(model(ids).logits, expected)
