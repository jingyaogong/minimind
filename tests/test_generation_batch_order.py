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

class GenerationBatchOrderTests(unittest.TestCase):
    def test_input_and_mask_are_expanded_in_prompt_order(self):
        model = tiny_model()
        ids = torch.tensor([[0, 3], [4, 5]])
        mask = torch.tensor([[0, 1], [1, 1]])
        with patch.object(model, 'forward', wraps=model.forward) as forward:
            output = model.generate(ids, attention_mask=mask, num_return_sequences=3,
                                    max_new_tokens=1, do_sample=False, eos_token_id=None)
        self.assertTrue(torch.equal(output[:, :2], ids.repeat_interleave(3, 0)))
        self.assertTrue(torch.equal(forward.call_args.args[1], mask.repeat_interleave(3, 0)))

    def test_single_return_keeps_original_batch(self):
        ids = torch.tensor([[3, 4], [5, 6]])
        self.assertTrue(torch.equal(tiny_model().generate(ids, max_new_tokens=0), ids))
