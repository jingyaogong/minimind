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

import tempfile
from pathlib import Path
from model.model_lora import apply_lora, load_lora, save_lora

class LoRACheckpointTests(unittest.TestCase):
    def checkpoint(self, model, path):
        save_lora(model, path)
        return torch.load(path, weights_only=True)

    def test_missing_adapter_matrix_is_not_silently_accepted(self):
        model = tiny_model()
        apply_lora(model, rank=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adapter.pth'
            state = self.checkpoint(model, path)
            del state[next(key for key in state if key.endswith('B.weight'))]
            torch.save(state, path)
            with self.assertRaisesRegex(RuntimeError, 'missing.*B.weight'):
                load_lora(model, path)

    def test_legacy_checkpoint_without_scaling_still_loads(self):
        model = tiny_model()
        apply_lora(model, rank=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pth'
            state = self.checkpoint(model, path)
            state = {key: value for key, value in state.items() if not key.endswith('scaling')}
            for key in state:
                if key.endswith('B.weight'):
                    state[key].fill_(0.125)
            torch.save(state, path)
            load_lora(model, path)
            for module in model.modules():
                if hasattr(module, 'lora'):
                    self.assertEqual(module.lora.scaling.item(), 1.0)
                    self.assertTrue(bool(module.lora.B.weight.eq(0.125).all()))
