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
from transformers import StoppingCriteria, StoppingCriteriaList, MaxLengthCriteria

class StoppingCriteriaTests(unittest.TestCase):
    def fixed_forward(self, ids, *args, **kwargs):
        logits = torch.zeros(ids.size(0), 1, 64)
        logits[:, :, 7] = 1
        return SimpleNamespace(logits=logits, past_key_values=None)

    def test_standard_length_criterion_stops_and_finishes_stream(self):
        model = tiny_model()
        streamer = unittest.mock.Mock()
        with patch.object(model, 'forward', side_effect=self.fixed_forward):
            result = model.generate(torch.tensor([[3, 4]]), max_new_tokens=20, do_sample=False,
                                    eos_token_id=None, streamer=streamer,
                                    stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=4)]))
        self.assertEqual(result.tolist(), [[3, 4, 7, 7]])
        streamer.end.assert_called_once()

    def test_per_row_criteria_pad_finished_sequences(self):
        class RowStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                length = input_ids.size(1)
                return torch.tensor([length >= 2, length >= 3], device=input_ids.device)
        model = tiny_model()
        with patch.object(model, 'forward', side_effect=self.fixed_forward):
            result = model.generate(torch.tensor([[3], [4]]), max_new_tokens=10,
                                    do_sample=False, eos_token_id=None, pad_token_id=0,
                                    stopping_criteria=StoppingCriteriaList([RowStop()]))
        self.assertEqual(result.tolist(), [[3, 7, 0], [4, 7, 7]])
