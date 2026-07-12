"""Tests for the SwiGLU feed-forward network."""

import torch
import torch.nn.functional as F

from slm.model.mlp import SwiGLU
from slm.utils.config import ModelConfig


def test_output_shape(tiny_config: ModelConfig) -> None:
    mlp = SwiGLU(tiny_config)
    x = torch.randn(2, 7, tiny_config.hidden_size)
    assert mlp(x).shape == (2, 7, tiny_config.hidden_size)


def test_matches_manual_formula(tiny_config: ModelConfig) -> None:
    # Arrange
    mlp = SwiGLU(tiny_config)
    x = torch.randn(1, 3, tiny_config.hidden_size)

    # Act
    output = mlp(x)

    # Assert: down( silu(gate(x)) * up(x) )
    expected = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    torch.testing.assert_close(output, expected)


def test_no_bias_parameters(tiny_config: ModelConfig) -> None:
    """CLAUDE.md specifies bias=False everywhere."""
    mlp = SwiGLU(tiny_config)
    for name, _ in mlp.named_parameters():
        assert "bias" not in name
