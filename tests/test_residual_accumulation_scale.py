import unittest
from unittest.mock import patch
from types import SimpleNamespace
from contextlib import ExitStack, nullcontext
import torch
from trainer import train_pretrain, train_full_sft, train_lora

class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
    def forward(self, input_ids, labels=None):
        return SimpleNamespace(loss=self.weight * input_ids.float().mean(), aux_loss=torch.tensor(0.0))

class ResidualAccumulationTests(unittest.TestCase):
    def test_real_training_loops_match_mean_gradient_for_each_window(self):
        for trainer in (train_pretrain, train_full_sft, train_lora):
            for values in ([3, 5], [1, 3, 5, 7], [1, 3, 5, 7, 9]):
                with self.subTest(trainer=trainer.__name__, values=values):
                    model = ScalarModel()
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                    args = SimpleNamespace(device='cpu', epochs=1, learning_rate=0.1,
                                           accumulation_steps=4, grad_clip=100.0,
                                           log_interval=100, save_interval=100)
                    loader = [(torch.tensor([[value]]), torch.tensor([[value]])) for value in values]
                    replacements = {'model': model, 'optimizer': optimizer, 'args': args,
                                    'autocast_ctx': nullcontext(),
                                    'scaler': torch.amp.GradScaler('cpu', enabled=False),
                                    'get_lr': lambda *args: 0.1, 'Logger': lambda *args: None,
                                    'is_main_process': lambda: False}
                    with ExitStack() as stack:
                        for key, value in replacements.items():
                            stack.enter_context(patch.object(trainer, key, value, create=True))
                        if trainer is train_lora:
                            trainer.train_epoch(0, loader, len(loader), list(model.parameters()))
                        else:
                            trainer.train_epoch(0, loader, len(loader))
                    expected = 2.0
                    for index in range(0, len(values), 4):
                        window = values[index:index + 4]
                        expected -= 0.1 * sum(window) / len(window)
                    self.assertAlmostEqual(model.weight.item(), expected, places=6)
