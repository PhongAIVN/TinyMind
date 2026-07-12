"""Tests for sampling strategies."""

import torch

from slm.inference.sampling import (
    apply_repetition_penalty,
    apply_top_k,
    apply_top_p,
    sample_next_token,
)


def test_greedy_picks_argmax() -> None:
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.5]])
    token = sample_next_token(logits, temperature=0.0)
    assert token.shape == (1, 1)
    assert token.item() == 1


def test_top_k_keeps_exactly_k() -> None:
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0, 0.0]])
    filtered = apply_top_k(logits, k=2)
    kept = (filtered > float("-inf")).sum().item()
    assert kept == 2
    assert filtered[0, 1] == 4.0 and filtered[0, 2] == 3.0


def test_top_k_zero_disables_filter() -> None:
    logits = torch.randn(1, 10)
    torch.testing.assert_close(apply_top_k(logits, k=0), logits)


def test_top_p_keeps_top_token_even_for_tiny_p() -> None:
    """The most likely token must always survive nucleus filtering."""
    logits = torch.tensor([[0.0, 10.0, 0.0]])
    filtered = apply_top_p(logits, p=0.01)
    assert filtered[0, 1] == 10.0
    assert (filtered > float("-inf")).sum().item() == 1


def test_top_p_masks_low_probability_tail() -> None:
    # Probabilities ~ [0.665, 0.245, 0.090]; p=0.9 keeps the first two.
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    filtered = apply_top_p(logits, p=0.9)
    assert filtered[0, 0] == 3.0
    assert filtered[0, 1] == 2.0
    assert filtered[0, 2] == float("-inf")


def test_repetition_penalty_reduces_seen_tokens() -> None:
    logits = torch.tensor([[2.0, 2.0, -1.0]])
    generated = torch.tensor([[0, 2]])  # tokens 0 and 2 already appeared
    penalized = apply_repetition_penalty(logits, generated, penalty=2.0)
    assert penalized[0, 0] == 1.0   # positive: divided by 2
    assert penalized[0, 1] == 2.0   # unseen: unchanged
    assert penalized[0, 2] == -2.0  # negative: multiplied by 2


def test_sampling_respects_top_k_one() -> None:
    """top_k=1 with any temperature must behave like greedy."""
    logits = torch.tensor([[0.0, 3.0, 1.0]])
    for _ in range(10):
        token = sample_next_token(logits, temperature=2.0, top_k=1)
        assert token.item() == 1


def test_batch_output_shape() -> None:
    logits = torch.randn(4, 100)
    token = sample_next_token(logits, temperature=1.0, top_k=10, top_p=0.9)
    assert token.shape == (4, 1)
    assert token.dtype == torch.int64
