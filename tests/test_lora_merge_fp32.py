import unittest
import tempfile
from pathlib import Path
import torch
from model.model_lora import apply_lora, save_lora, merge_lora

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4, bias=False)
    @property
    def device(self):
        return self.projection.weight.device

class LoRAMergePrecisionTests(unittest.TestCase):
    def test_merge_rounds_once_after_adding_delta(self):
        model = LinearModel()
        model.projection.weight.data.fill_(0.9997)
        apply_lora(model, rank=1)
        model.projection.lora.A.weight.data.fill_(1.0)
        model.projection.lora.B.weight.data.fill_(0.0002)
        with tempfile.TemporaryDirectory() as directory:
            adapter, merged = Path(directory) / 'adapter.pth', Path(directory) / 'merged.pth'
            save_lora(model, adapter)
            state = torch.load(adapter, weights_only=True)
            expected = (model.projection.weight.float() + state['projection.lora.B.weight'].float() @ state['projection.lora.A.weight'].float()).half()
            merge_lora(model, adapter, merged)
            actual = torch.load(merged, weights_only=True)
            torch.testing.assert_close(actual['projection.weight'], expected, rtol=0, atol=0)
            self.assertEqual(actual['projection.weight'].dtype, torch.float16)
            self.assertFalse(any('.lora.' in key for key in actual))
