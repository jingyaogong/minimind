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

from types import SimpleNamespace

class MultipleEOSTests(unittest.TestCase):
    def run_generation(self, eos):
        model = tiny_model()
        calls = []
        def forward(ids, *args, **kwargs):
            calls.append(ids)
            logits = torch.full((2, 1, 64), -10.0)
            logits[0, 0, 2 if len(calls) == 1 else 9] = 10
            logits[1, 0, 7 if len(calls) == 1 else 3] = 10
            return SimpleNamespace(logits=logits, past_key_values=None)
        with patch.object(model, 'forward', side_effect=forward):
            result = model.generate(torch.tensor([[4], [5]]), max_new_tokens=4,
                                    do_sample=False, eos_token_id=eos, use_cache=False)
        return result, len(calls)

    def test_each_sequence_stops_on_either_eos(self):
        for eos in ([2, 3], torch.tensor([2, 3])):
            with self.subTest(eos=eos):
                result, calls = self.run_generation(eos)
                self.assertEqual(calls, 2)
                self.assertEqual(result.tolist(), [[4, 2, 2], [5, 7, 3]])

    def test_empty_eos_list_disables_early_stopping(self):
        result, calls = self.run_generation([])
        self.assertEqual(calls, 4)
        self.assertEqual(result.shape, (2, 5))
