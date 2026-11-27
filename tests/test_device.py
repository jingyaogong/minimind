from contextlib import nullcontext
from types import SimpleNamespace
from unittest import TestCase, mock

import torch

from model.device import inference_dtype, initialize_device, training_amp


class DeviceTests(TestCase):
    def test_cpu_and_cuda_do_not_import_torch_npu(self):
        with mock.patch("model.device.importlib.import_module") as import_module:
            self.assertEqual(initialize_device("cpu"), torch.device("cpu"))
            self.assertEqual(initialize_device("cuda:1"), torch.device("cuda:1"))
        import_module.assert_not_called()

    def test_missing_torchnpu_has_a_platform_error(self):
        with mock.patch("model.device.importlib.import_module", side_effect=ImportError):
            with self.assertRaisesRegex(RuntimeError, "requires torch_npu"):
                initialize_device("npu:0")

    def test_inference_defaults_and_explicit_dtype(self):
        self.assertIsNone(inference_dtype(torch.device("cpu"), "auto"))
        self.assertIs(torch.float16, inference_dtype(torch.device("cuda"), "auto"))
        self.assertIs(torch.float32, inference_dtype(torch.device("cuda"), "float32"))

    def test_cuda_training_keeps_autocast_and_fp16_scaler(self):
        with mock.patch.object(torch.cuda.amp, "autocast") as autocast, \
             mock.patch.object(torch.cuda.amp, "GradScaler") as scaler:
            self.assertEqual(
                training_amp(torch.device("cuda:0"), "float16"),
                (autocast.return_value, scaler.return_value),
            )
        autocast.assert_called_once_with(dtype=torch.float16)
        scaler.assert_called_once_with(enabled=True)

    def test_npu_training_uses_bf16_without_gradient_scaling(self):
        device = SimpleNamespace(type="npu")
        npu = mock.Mock()
        npu.is_bf16_supported.return_value = True
        with mock.patch.object(torch, "npu", npu, create=True), \
             mock.patch.object(torch, "autocast") as autocast, \
             mock.patch.object(torch.amp, "GradScaler") as scaler:
            self.assertEqual(
                training_amp(device, "bfloat16"),
                (autocast.return_value, scaler.return_value),
            )
        autocast.assert_called_once_with(device_type="npu", dtype=torch.bfloat16)
        scaler.assert_called_once_with("npu", enabled=False)

    def test_npu_training_precision_guards(self):
        device = SimpleNamespace(type="npu")
        npu = mock.Mock()
        npu.is_bf16_supported.return_value = False
        with mock.patch.object(torch, "npu", npu, create=True), \
             mock.patch.object(torch.amp, "GradScaler") as scaler:
            context, _ = training_amp(device, "float32")
            self.assertIsInstance(context, nullcontext)
            scaler.assert_called_once_with("npu", enabled=False)
            with self.assertRaisesRegex(ValueError, "does not support bfloat16"):
                training_amp(device, "bfloat16")
            with self.assertRaisesRegex(ValueError, "supports --dtype float32 or bfloat16"):
                training_amp(device, "float16")
