import unittest
import torch
from trainer.train_distillation import distillation_loss

class EmptyDistillationTests(unittest.TestCase):
    def test_all_ignored_labels_produce_differentiable_zero(self):
        student = torch.randn(2, 4, 8, requires_grad=True)
        teacher = torch.randn_like(student)
        valid = torch.zeros(2, 4, dtype=torch.bool)
        loss = distillation_loss(student[valid], teacher[valid], temperature=1.5)
        self.assertTrue(bool(loss.isfinite()))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(student.grad, torch.zeros_like(student)))

    def test_nonempty_identical_logits_keep_zero_loss(self):
        student = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
        loss = distillation_loss(student, student.detach().clone(), temperature=1.5)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        loss.backward()
        self.assertTrue(bool(student.grad.isfinite().all()))
