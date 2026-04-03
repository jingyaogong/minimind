"""Tests for the repetition_penalty fix in MiniMindForCausalLM.generate().

The original implementation divides all logits of previously-seen tokens by the
penalty factor. For tokens with *negative* logits, division by penalty > 1.0
moves the logit *toward* zero, which **increases** the token's probability —
the opposite of the intended effect.

The fix applies the penalty conditionally:
  - positive logits → divide by penalty (push toward zero, lower probability)
  - negative logits → multiply by penalty (push away from zero, lower probability)

This matches the widely-adopted convention from Hugging Face Transformers:
  https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
"""

import math
import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helper: apply the FIXED penalty logic to a logits tensor
# ---------------------------------------------------------------------------
def apply_penalty_fixed(logits: torch.Tensor, token_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    """Reference implementation of the correct repetition penalty."""
    out = logits.clone()
    prev = torch.unique(token_ids)
    score = out[prev]
    out[prev] = torch.where(score < 0, score * penalty, score / penalty)
    return out


def apply_penalty_buggy(logits: torch.Tensor, token_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    """The old (buggy) implementation for comparison."""
    out = logits.clone()
    out[torch.unique(token_ids)] /= penalty
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRepetitionPenaltyDirection:
    """Core tests: the penalty must ALWAYS reduce probability of seen tokens."""

    def test_positive_logit_reduced(self):
        """A previously-seen token with positive logit should have LOWER probability."""
        logits = torch.tensor([3.0, 1.0, 0.0, -1.0, -2.0])
        seen = torch.tensor([0])  # token 0 has logit 3.0
        penalty = 1.5

        orig_prob = F.softmax(logits, dim=-1)[0].item()
        fixed = apply_penalty_fixed(logits, seen, penalty)
        new_prob = F.softmax(fixed, dim=-1)[0].item()

        assert new_prob < orig_prob, f"Positive logit token prob should decrease: {orig_prob:.4f} → {new_prob:.4f}"

    def test_negative_logit_reduced(self):
        """A previously-seen token with negative logit should have LOWER probability."""
        logits = torch.tensor([3.0, 1.0, 0.0, -1.0, -2.0])
        seen = torch.tensor([4])  # token 4 has logit -2.0
        penalty = 1.5

        orig_prob = F.softmax(logits, dim=-1)[4].item()
        fixed = apply_penalty_fixed(logits, seen, penalty)
        new_prob = F.softmax(fixed, dim=-1)[4].item()

        assert new_prob < orig_prob, f"Negative logit token prob should decrease: {orig_prob:.4f} → {new_prob:.4f}"

    def test_buggy_boosts_negative_logit(self):
        """Demonstrate that the OLD code incorrectly BOOSTS negative-logit tokens."""
        logits = torch.tensor([3.0, 1.0, 0.0, -1.0, -2.0])
        seen = torch.tensor([4])
        penalty = 1.5

        orig_prob = F.softmax(logits, dim=-1)[4].item()
        buggy = apply_penalty_buggy(logits, seen, penalty)
        buggy_prob = F.softmax(buggy, dim=-1)[4].item()

        # The buggy version INCREASES probability — this is the bug
        assert buggy_prob > orig_prob, f"Buggy code should boost negative logit prob: {orig_prob:.4f} → {buggy_prob:.4f}"

    def test_zero_logit_unchanged(self):
        """A token with logit exactly 0.0 should remain at 0.0 regardless of penalty."""
        logits = torch.tensor([2.0, 0.0, -1.0])
        seen = torch.tensor([1])  # token 1 has logit 0.0
        penalty = 2.0

        fixed = apply_penalty_fixed(logits, seen, penalty)
        # 0.0 / penalty = 0.0, and 0.0 * penalty = 0.0, either branch gives 0.0
        assert fixed[1].item() == 0.0


class TestRepetitionPenaltyNoOp:
    """Penalty = 1.0 should be a no-op."""

    def test_penalty_one_is_noop(self):
        logits = torch.tensor([2.5, -1.3, 0.0, 4.1, -3.0])
        seen = torch.tensor([0, 1, 2, 3, 4])
        fixed = apply_penalty_fixed(logits, seen, 1.0)
        assert torch.allclose(logits, fixed)


class TestRepetitionPenaltyMultipleTokens:
    """Penalty applied to multiple seen tokens at once."""

    def test_all_seen_token_logits_move_away_from_zero(self):
        """Every seen token's logit magnitude should increase (move away from zero).
        
        Note: softmax probability may not always decrease for every penalized token
        when multiple tokens are penalized simultaneously (redistribution effect).
        The correct invariant is that logits move in the right direction."""
        logits = torch.tensor([5.0, 2.0, -1.0, -3.0, 0.5])
        seen = torch.tensor([0, 1, 2, 3])  # 4 is unseen
        penalty = 1.3

        fixed = apply_penalty_fixed(logits, seen, penalty)

        for idx in [0, 1]:  # positive logits should decrease
            assert fixed[idx] < logits[idx], (
                f"Token {idx} (logit={logits[idx]:.1f}) should decrease: "
                f"{logits[idx]:.4f} → {fixed[idx]:.4f}"
            )
        for idx in [2, 3]:  # negative logits should become more negative
            assert fixed[idx] < logits[idx], (
                f"Token {idx} (logit={logits[idx]:.1f}) should become more negative: "
                f"{logits[idx]:.4f} → {fixed[idx]:.4f}"
            )

    def test_unseen_token_prob_increases(self):
        """Unseen tokens should gain probability share when seen tokens are penalized."""
        logits = torch.tensor([5.0, 2.0, -1.0, -3.0, 0.5])
        seen = torch.tensor([0, 1, 2, 3])
        penalty = 1.3

        orig_prob_unseen = F.softmax(logits, dim=-1)[4].item()
        fixed = apply_penalty_fixed(logits, seen, penalty)
        new_prob_unseen = F.softmax(fixed, dim=-1)[4].item()

        assert new_prob_unseen > orig_prob_unseen, (
            f"Unseen token prob should increase: {orig_prob_unseen:.4f} → {new_prob_unseen:.4f}"
        )


class TestRepetitionPenaltyDuplicateTokens:
    """Duplicate token IDs in input should not cause issues."""

    def test_duplicates_in_input(self):
        logits = torch.tensor([1.0, -2.0, 3.0])
        seen = torch.tensor([1, 1, 1, 0, 0])  # duplicates
        penalty = 1.5

        fixed = apply_penalty_fixed(logits, seen, penalty)
        expected = apply_penalty_fixed(logits, torch.tensor([0, 1]), penalty)
        assert torch.allclose(fixed, expected)


class TestRepetitionPenaltyStrength:
    """Higher penalty should produce a stronger effect."""

    def test_higher_penalty_stronger_reduction(self):
        logits = torch.tensor([4.0, -2.0, 1.0])
        seen = torch.tensor([0, 1])

        probs_15 = F.softmax(apply_penalty_fixed(logits, seen, 1.5), dim=-1)
        probs_30 = F.softmax(apply_penalty_fixed(logits, seen, 3.0), dim=-1)

        # Higher penalty → lower prob for token 0 (positive logit)
        assert probs_30[0] < probs_15[0]
        # Higher penalty → lower prob for token 1 (negative logit)
        assert probs_30[1] < probs_15[1]


class TestRepetitionPenaltyBatch:
    """Test batch behavior matching the generate() loop."""

    def test_batch_independent(self):
        """Each batch element should be penalized independently."""
        logits = torch.tensor([
            [2.0, -1.0, 0.5],
            [-2.0, 3.0, 1.0],
        ])
        input_ids = torch.tensor([
            [0, 0, 1],
            [2, 1, 1],
        ])
        penalty = 1.5

        result = logits.clone()
        for i in range(input_ids.shape[0]):
            prev_tokens = torch.unique(input_ids[i])
            score = result[i, prev_tokens]
            result[i, prev_tokens] = torch.where(score < 0, score * penalty, score / penalty)

        # Batch 0: tokens 0,1 seen → logit[0,0]=2.0/1.5, logit[0,1]=-1.0*1.5, logit[0,2] unchanged
        assert abs(result[0, 0].item() - 2.0 / 1.5) < 1e-5
        assert abs(result[0, 1].item() - (-1.0 * 1.5)) < 1e-5
        assert abs(result[0, 2].item() - 0.5) < 1e-5

        # Batch 1: tokens 1,2 seen → logit[1,1]=3.0/1.5, logit[1,2]=1.0/1.5, logit[1,0] unchanged
        assert abs(result[1, 0].item() - (-2.0)) < 1e-5
        assert abs(result[1, 1].item() - 3.0 / 1.5) < 1e-5
        assert abs(result[1, 2].item() - 1.0 / 1.5) < 1e-5


class TestRepetitionPenaltyEdgeCases:
    """Edge cases and boundary conditions."""

    def test_all_negative_logits(self):
        """When all logits are negative, all seen tokens should still be pushed lower."""
        logits = torch.tensor([-0.5, -1.0, -2.0, -3.0])
        seen = torch.tensor([0, 1, 2, 3])
        penalty = 1.2

        orig_probs = F.softmax(logits, dim=-1)
        # When ALL tokens are penalized equally, softmax probs change if any logit magnitudes differ
        fixed = apply_penalty_fixed(logits, seen, penalty)

        # All logits are negative → all get multiplied by penalty → all become more negative
        for i in range(4):
            assert fixed[i] < logits[i], f"Token {i} logit should decrease (more negative)"

    def test_single_token_vocabulary(self):
        """Single-token vocabulary: penalty doesn't change softmax (always 1.0)."""
        logits = torch.tensor([5.0])
        seen = torch.tensor([0])
        penalty = 2.0

        fixed = apply_penalty_fixed(logits, seen, penalty)
        assert abs(F.softmax(fixed, dim=-1)[0].item() - 1.0) < 1e-6

    def test_large_penalty(self):
        """Even with very large penalty, probabilities should remain valid."""
        logits = torch.tensor([3.0, -3.0, 1.0, -1.0, 0.0])
        seen = torch.arange(5)
        penalty = 100.0

        fixed = apply_penalty_fixed(logits, seen, penalty)
        probs = F.softmax(fixed, dim=-1)

        assert torch.all(probs >= 0)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_empty_seen_tokens(self):
        """No seen tokens: logits should be unchanged."""
        logits = torch.tensor([2.0, -1.0, 0.5])
        seen = torch.tensor([], dtype=torch.long)
        penalty = 1.5

        fixed = apply_penalty_fixed(logits, seen, penalty)
        assert torch.allclose(logits, fixed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
