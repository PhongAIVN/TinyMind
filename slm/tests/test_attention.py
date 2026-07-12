"""Tests for multi-head causal self-attention."""

import torch

from slm.model.attention import MultiHeadAttention
from slm.model.rope import precompute_rope_cos_sin
from slm.utils.config import ModelConfig


def _build(config: ModelConfig) -> tuple[MultiHeadAttention, torch.Tensor, torch.Tensor]:
    attention = MultiHeadAttention(config, layer_index=0)
    cos, sin = precompute_rope_cos_sin(config.head_dim, config.context_length)
    return attention, cos, sin


def test_output_shape(tiny_config: ModelConfig) -> None:
    attention, cos, sin = _build(tiny_config)
    x = torch.randn(2, 10, tiny_config.hidden_size)
    assert attention(x, cos, sin).shape == (2, 10, tiny_config.hidden_size)


def test_causality(tiny_config: ModelConfig) -> None:
    """Changing a future token must not change earlier outputs."""
    attention, cos, sin = _build(tiny_config)
    x = torch.randn(1, 8, tiny_config.hidden_size)

    baseline = attention(x, cos, sin)

    perturbed_input = x.clone()
    perturbed_input[:, 5:] += 10.0  # change tokens 5, 6, 7 drastically
    perturbed = attention(perturbed_input, cos, sin)

    # Positions 0-4 attend only to themselves and earlier — unchanged.
    torch.testing.assert_close(perturbed[:, :5], baseline[:, :5], rtol=1e-4, atol=1e-5)
    # Position 5 sees itself changed — must differ.
    assert not torch.allclose(perturbed[:, 5], baseline[:, 5])


def test_causal_mask_shape_and_values() -> None:
    mask = MultiHeadAttention._causal_mask(
        query_len=3, key_len=3, position_offset=0, device=torch.device("cpu")
    )
    assert mask.shape == (1, 1, 3, 3)
    expected = torch.tensor(
        [
            [0.0, float("-inf"), float("-inf")],
            [0.0, 0.0, float("-inf")],
            [0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(mask[0, 0], expected, equal_nan=True)


def test_causal_mask_with_offset() -> None:
    """A single query at absolute position 3 may see keys 0..3 but not 4."""
    mask = MultiHeadAttention._causal_mask(
        query_len=1, key_len=5, position_offset=3, device=torch.device("cpu")
    )
    row = mask[0, 0, 0]
    assert (row[:4] == 0).all()
    assert row[4] == float("-inf")
