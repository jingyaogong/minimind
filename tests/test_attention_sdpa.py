"""Regression tests for padded SDPA; no model weights or datasets are needed.

Run: python -m unittest discover -s tests -p test_attention_sdpa.py -v
"""
import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from model.model_minimind import Attention, MiniMindConfig, precompute_freqs_cis


class AttentionSDPATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(min(cls.old_threads, 2))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make_inputs(self, device="cpu", dtype=torch.float32, dropout=0.0):
        torch.manual_seed(42)
        config = MiniMindConfig(hidden_size=64, num_hidden_layers=1,
                               num_attention_heads=4, num_key_value_heads=2,
                               head_dim=16, dropout=dropout)
        attention = Attention(config).to(device=device, dtype=dtype).eval()
        inputs = torch.randn(2, 12, 64, device=device, dtype=dtype)
        positions = tuple(t.to(device) for t in precompute_freqs_cis(16, end=12))
        return attention, inputs, positions

    def check_parity(self, device, dtype):
        for causal in (True, False):
            for padding in ("none", "ones", "left", "right", "interior"):
                with self.subTest(device=device, causal=causal, padding=padding):
                    candidate, inputs, positions = self.make_inputs(device, dtype)
                    candidate.is_causal = causal
                    reference = copy.deepcopy(candidate)
                    candidate.flash, reference.flash = True, False
                    mask = torch.ones(2, 12, device=device, dtype=torch.long)
                    if padding == "left":
                        mask[0, :4], mask[1, :2] = 0, 0
                    elif padding == "right":
                        mask[0, 8:], mask[1, 10:] = 0, 0
                    elif padding == "interior":
                        mask[0, 3:5], mask[1, 6:8] = 0, 0
                    valid = mask.bool()
                    passed_mask = None if padding == "none" else mask
                    outputs, gradients = [], []
                    target = torch.randn_like(inputs).float()
                    for attention in (reference, candidate):
                        x = inputs.detach().clone().requires_grad_(True)
                        output, _ = attention(x, positions, attention_mask=passed_mask)
                        # Fully masked padded query rows are not equivalent to the
                        # old finite -1e9 mask, and must not contribute to the loss.
                        loss = (output[valid].float() - target[valid]).square().mean()
                        loss.backward()
                        self.assertTrue(torch.isfinite(output).all().item())
                        gradient = torch.cat([x.grad.float().flatten()] + [
                            p.grad.float().flatten() for p in attention.parameters()
                            if p.grad is not None
                        ])
                        self.assertTrue(torch.isfinite(gradient).all().item())
                        outputs.append(output.detach()[valid].float())
                        gradients.append(gradient)
                    bf16 = dtype == torch.bfloat16
                    torch.testing.assert_close(outputs[1], outputs[0],
                                               atol=1e-2 if bf16 else 1e-5,
                                               rtol=3e-2 if bf16 else 1e-5)
                    # Compare the gradient vectors, not just their norms.
                    relative_error = ((gradients[1] - gradients[0]).norm()
                                      / gradients[0].norm().clamp_min(1e-12)).item()
                    self.assertLess(relative_error, 3e-2 if bf16 else 1e-4)

    def test_cpu_outputs_and_full_gradients(self):
        self.check_parity("cpu", torch.float32)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                         "CUDA BF16 is optional")
    def test_cuda_bf16_outputs_and_full_gradients(self):
        self.check_parity("cuda", torch.bfloat16)

    def test_padded_inputs_use_combined_sdpa_mask(self):
        attention, inputs, positions = self.make_inputs()
        mask = torch.ones(2, 12, dtype=torch.long)
        mask[0, :3], mask[1, 9:] = 0, 0
        for causal in (True, False):
            attention.is_causal = causal
            with self.subTest(causal=causal), patch(
                "model.model_minimind.F.scaled_dot_product_attention",
                wraps=F.scaled_dot_product_attention,
            ) as sdpa:
                attention(inputs, positions, attention_mask=mask)
                sdpa.assert_called_once()
                kwargs = sdpa.call_args.kwargs
                expected = torch.ones(12, 12, dtype=torch.bool)
                if causal:
                    expected = expected.tril()
                expected = expected & mask.bool()[:, None, None, :]
                torch.testing.assert_close(kwargs["attn_mask"], expected)
                self.assertFalse(kwargs["is_causal"])

    def test_unpadded_inputs_keep_original_causal_sdpa(self):
        attention, inputs, positions = self.make_inputs()
        for mask in (None, torch.ones(2, 12, dtype=torch.long)):
            with self.subTest(mask_is_none=mask is None), patch(
                "model.model_minimind.F.scaled_dot_product_attention",
                wraps=F.scaled_dot_product_attention,
            ) as sdpa:
                attention(inputs, positions, attention_mask=mask)
                sdpa.assert_called_once()
                self.assertIsNone(sdpa.call_args.kwargs["attn_mask"])
                self.assertTrue(sdpa.call_args.kwargs["is_causal"])

    def test_manual_fallbacks_and_cached_continuation(self):
        attention, inputs, positions = self.make_inputs()
        mask = torch.ones(2, 12, dtype=torch.long)
        mask[0, :2] = 0
        attention.flash = False
        with torch.no_grad():
            _, cache = attention(inputs[:, :8], tuple(t[:8] for t in positions),
                                 use_cache=True, attention_mask=mask[:, :8])
        for length, past, enabled in ((12, None, False), (1, None, True),
                                      (1, cache, True), (4, cache, True)):
            with self.subTest(length=length, cached=past is not None, flash=enabled):
                start = 8 if past is not None else 0
                x = inputs[:, start:start + length]
                pos = tuple(t[start:start + length] for t in positions)
                passed_mask = mask[:, :start + length]
                attention.flash = enabled
                with patch("model.model_minimind.F.scaled_dot_product_attention",
                           wraps=F.scaled_dot_product_attention) as sdpa:
                    output, _ = attention(x, pos, past_key_value=past,
                                          attention_mask=passed_mask)
                    sdpa.assert_not_called()
                reference = copy.deepcopy(attention)
                reference.flash = False
                expected, _ = reference(x, pos, past_key_value=past,
                                        attention_mask=passed_mask)
                torch.testing.assert_close(output, expected, atol=0, rtol=0)

    def test_dropout_probability_tracks_training_mode(self):
        attention, inputs, positions = self.make_inputs(dropout=0.2)
        mask = torch.ones(2, 12, dtype=torch.long)
        mask[:, -2:] = 0
        for training, probability in ((True, 0.2), (False, 0.0)):
            attention.train(training)
            with self.subTest(training=training), patch(
                "model.model_minimind.F.scaled_dot_product_attention",
                wraps=F.scaled_dot_product_attention,
            ) as sdpa:
                attention(inputs, positions, attention_mask=mask)
                self.assertEqual(sdpa.call_args.kwargs["dropout_p"], probability)


if __name__ == "__main__":
    unittest.main()
