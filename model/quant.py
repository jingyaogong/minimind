"""
W8A8 QAT fake-quantization utilities (pure PyTorch, STE).

- WeightFakeQuant: per-output-channel symmetric int8 on Linear weights.
- ActFakeQuant:    per-tensor   symmetric int8 on activations, with EMA observer
                   and a warmup window (collect stats only, no quant in forward).
- QATLinear:       drop-in nn.Linear replacement that fake-quantizes its input
                   activation and its weight on every forward.
- prepare_qat:     in-place swap every nn.Linear in a model with QATLinear,
                   skipping any module whose qualified name matches `skip_patterns`
                   (default: lm_head, MoE router `mlp.gate`).

The STE used here zeros gradients for values that saturate (clip-aware STE),
which is more stable than plain pass-through for low-bit training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FakeQuantize(torch.autograd.Function):
    """round(x/scale + zp) -> clamp -> dequant. Backward: STE inside clip, 0 outside."""

    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        q = torch.round(x / scale + zero_point)
        q_clamped = q.clamp(qmin, qmax)
        ctx.save_for_backward((q >= qmin) & (q <= qmax))
        return (q_clamped - zero_point) * scale

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output * mask.to(grad_output.dtype), None, None, None, None


def fake_quantize(x, scale, zero_point, qmin, qmax):
    return _FakeQuantize.apply(x, scale, zero_point, qmin, qmax)


class WeightFakeQuant(nn.Module):
    """Per-output-channel symmetric int8 fake-quant for Linear weight [out, in]."""

    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1
        self.qmin = -(1 << (bits - 1))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        max_abs = w.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = (max_abs / self.qmax).to(w.dtype)
        zp = torch.zeros_like(scale)
        return fake_quantize(w, scale, zp, self.qmin, self.qmax)


class ActFakeQuant(nn.Module):
    """Per-tensor symmetric int8 fake-quant with EMA min/max observer + warmup."""

    def __init__(self, bits: int = 8, momentum: float = 0.99, observer_steps: int = 200):
        super().__init__()
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1
        self.qmin = -(1 << (bits - 1))
        self.momentum = momentum
        self.observer_steps = observer_steps
        self.register_buffer("running_max", torch.zeros(1))
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))
        self.enabled = True

    def extra_repr(self) -> str:
        return f"bits={self.bits}, observer_steps={self.observer_steps}, momentum={self.momentum}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if self.training:
            cur_max = x.detach().abs().amax().to(self.running_max.dtype)
            if self.step.item() == 0:
                self.running_max.copy_(cur_max.unsqueeze(0))
            else:
                self.running_max.mul_(self.momentum).add_(cur_max.unsqueeze(0) * (1.0 - self.momentum))
            self.step += 1
            # Warmup: collect stats only; passing x through unquantized lets the
            # observer stabilize before fake-quant starts perturbing gradients.
            if self.step.item() < self.observer_steps:
                return x
        max_abs = self.running_max.clamp_min(1e-8)
        scale = (max_abs / self.qmax).to(x.dtype)
        zp = torch.zeros_like(scale)
        return fake_quantize(x, scale, zp, self.qmin, self.qmax)


class QATLinear(nn.Linear):
    """nn.Linear with fake-quant on input activation and on weight."""

    def __init__(self, in_features, out_features, bias=True, *, w_bits=8, a_bits=8,
                 a_momentum=0.99, a_observer_steps=200):
        super().__init__(in_features, out_features, bias=bias)
        self.weight_fq = WeightFakeQuant(bits=w_bits)
        self.act_fq = ActFakeQuant(bits=a_bits, momentum=a_momentum, observer_steps=a_observer_steps)

    @classmethod
    def from_float(cls, linear: nn.Linear, **kw) -> "QATLinear":
        qm = cls(linear.in_features, linear.out_features, bias=linear.bias is not None, **kw)
        # Share existing tensors so optimizer state stays attached after swap.
        qm.weight = linear.weight
        if linear.bias is not None:
            qm.bias = linear.bias
        return qm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.act_fq(x)
        w_q = self.weight_fq(self.weight)
        return F.linear(x_q, w_q, self.bias)


DEFAULT_SKIP_PATTERNS = ("lm_head", "mlp.gate")


def _matches_skip(full_name: str, pattern: str) -> bool:
    """Pattern matches at a dot boundary, so 'mlp.gate' does NOT skip 'mlp.gate_proj'."""
    return full_name == pattern or full_name.endswith("." + pattern)


def prepare_qat(model: nn.Module, w_bits: int = 8, a_bits: int = 8,
                skip_patterns=DEFAULT_SKIP_PATTERNS,
                a_momentum: float = 0.99, a_observer_steps: int = 200) -> int:
    """Swap every nn.Linear in `model` with QATLinear in place.

    A Linear is skipped iff its qualified name equals a skip pattern or ends
    with `'.' + pattern` (segment-aware, not substring). Defaults skip the LM
    head (precision-sensitive, often tied to embeddings) and the MoE router
    `mlp.gate` (small, routing-sensitive) without touching the dense
    FeedForward's `mlp.gate_proj`.

    Returns the number of linears replaced.
    """
    replaced = 0
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear) or isinstance(child, QATLinear):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(_matches_skip(full_name, p) for p in skip_patterns):
                continue
            new_mod = QATLinear.from_float(
                child, w_bits=w_bits, a_bits=a_bits,
                a_momentum=a_momentum, a_observer_steps=a_observer_steps,
            )
            # Move buffers to weight's device. dtype on buffers stays fp32 on purpose.
            new_mod.act_fq.running_max = new_mod.act_fq.running_max.to(child.weight.device)
            new_mod.act_fq.step = new_mod.act_fq.step.to(child.weight.device)
            setattr(parent, child_name, new_mod)
            replaced += 1
    return replaced


def set_act_quant_enabled(model: nn.Module, enabled: bool) -> None:
    """Toggle activation fake-quant (e.g. disable for eval-on-fp comparison)."""
    for m in model.modules():
        if isinstance(m, ActFakeQuant):
            m.enabled = enabled


@torch.no_grad()
def bake_quantized_weights(model: nn.Module) -> None:
    """Replace each QATLinear's weight with its fake-quantized value, in place.

    After this, the model can run as plain QATLinears without further weight
    updates and the saved fp weights already reflect the int8 grid; useful for
    final eval / export.
    """
    for m in model.modules():
        if isinstance(m, QATLinear):
            m.weight.copy_(m.weight_fq(m.weight))
