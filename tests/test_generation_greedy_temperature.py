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

class GreedyTemperatureTests(unittest.TestCase):
    def test_greedy_ignores_temperature_and_sampling_filters(self):
        model = tiny_model()
        logits = torch.zeros(1, 1, 64)
        logits[0, 0, 3] = 1.0
        logits[0, 0, 7] = 2.0
        for temperature in (0.0, -1.0, 0.7):
            with self.subTest(temperature=temperature), patch.object(model, 'forward', return_value=SimpleNamespace(logits=logits.clone(), past_key_values=None)):
                result = model.generate(torch.tensor([[4]]), max_new_tokens=1, do_sample=False,
                                        temperature=temperature, top_k=1000, top_p=0.0)
                self.assertEqual(result[0, -1].item(), 7)

    def test_repetition_penalty_still_applies_to_greedy(self):
        model = tiny_model()
        logits = torch.zeros(1, 1, 64)
        logits[0, 0, 3], logits[0, 0, 7] = 1.5, 2.0
        with patch.object(model, 'forward', return_value=SimpleNamespace(logits=logits, past_key_values=None)):
            result = model.generate(torch.tensor([[7]]), max_new_tokens=1, do_sample=False,
                                    temperature=0.0, repetition_penalty=2.0)
        self.assertEqual(result[0, -1].item(), 3)
