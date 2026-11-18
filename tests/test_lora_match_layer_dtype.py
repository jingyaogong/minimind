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

class LoRALayerDtypeTests(unittest.TestCase):
    def test_adapters_work_when_base_is_already_in_reduced_precision(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                model = tiny_model().to(dtype)
                ids = torch.tensor([[3, 4]])
                with torch.no_grad():
                    expected = model(ids).logits
                    apply_lora(model, rank=4)
                    actual = model(ids).logits
                torch.testing.assert_close(actual, expected)
                for module in model.modules():
                    if hasattr(module, 'lora'):
                        self.assertEqual(module.lora.A.weight.dtype, module.weight.dtype)
                        self.assertEqual(module.lora.B.weight.device, module.weight.device)

    def test_nonzero_adapters_receive_gradients_in_base_dtype(self):
        model = tiny_model().to(torch.bfloat16)
        apply_lora(model, rank=4)
        model(torch.tensor([[3, 4]])).logits.float().square().sum().backward()
        gradients = [module.lora.B.weight.grad for module in model.modules() if hasattr(module, 'lora')]
        self.assertTrue(all(grad is not None and bool(grad.isfinite().all()) for grad in gradients))
        self.assertTrue(any(grad.abs().sum().item() > 0 for grad in gradients))
