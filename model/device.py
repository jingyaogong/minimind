"""Device and precision setup shared by the MiniMind entry points."""

import importlib
from contextlib import nullcontext

import torch


def initialize_device(value):
    """Resolve a PyTorch device, loading TorchNPU for explicit NPU requests."""
    if str(value).partition(":")[0] != "npu":
        return torch.device(value)

    try:
        importlib.import_module("torch_npu")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "An NPU device requires torch_npu and a compatible CANN environment."
        ) from exc

    device = torch.device(value)
    index = 0 if device.index is None else device.index
    if not torch.npu.is_available() or index >= torch.npu.device_count():
        raise RuntimeError(f"NPU device npu:{index} is unavailable.")
    device = torch.device("npu", index)
    torch.npu.set_device(device)
    return device


def _require_bfloat16(device):
    if device.type == "npu" and not torch.npu.is_bf16_supported():
        raise ValueError("This NPU environment does not support bfloat16. Use --dtype float32.")


def inference_dtype(device, precision):
    if precision == "auto":
        return torch.float16 if device.type == "cuda" else None
    if precision == "bfloat16":
        _require_bfloat16(device)
    return getattr(torch, precision)


def training_amp(device, precision):
    """Return the autocast context and scaler for the requested training precision."""
    if device.type == "npu":
        if precision not in ("float32", "bfloat16"):
            raise ValueError("NPU training supports --dtype float32 or bfloat16.")
        if precision == "bfloat16":
            _require_bfloat16(device)
            context = torch.autocast(device_type="npu", dtype=torch.bfloat16)
        else:
            context = nullcontext()
        return context, torch.amp.GradScaler("npu", enabled=False)

    context = nullcontext()
    if device.type == "cuda" and precision != "float32":
        dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
        context = torch.cuda.amp.autocast(dtype=dtype)
    return context, torch.cuda.amp.GradScaler(enabled=(precision == "float16"))
