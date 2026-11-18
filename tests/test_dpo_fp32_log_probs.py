import unittest
import torch
import torch.nn.functional as F
from trainer.train_dpo import dpo_loss, logits_to_log_probs

class DPOPrecisionTests(unittest.TestCase):
    def test_half_precision_long_sequences_have_finite_loss_and_gradients(self):
        policy = torch.full((2, 4096), -30.0, dtype=torch.float16, requires_grad=True)
        reference = torch.full_like(policy, -30.0)
        mask = torch.ones_like(policy, dtype=torch.long)
        loss = dpo_loss(reference, policy, mask, beta=0.1)
        self.assertTrue(bool(loss.isfinite()))
        self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(2.0)).item(), places=6)
        loss.backward()
        self.assertTrue(bool(policy.grad.isfinite().all()))

    def test_log_softmax_is_computed_in_fp32_before_gather(self):
        logits = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 0.1, -0.1]]], dtype=torch.bfloat16)
        labels = torch.tensor([[2, 0]])
        actual = logits_to_log_probs(logits, labels)
        expected = F.log_softmax(logits.float(), dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected)
