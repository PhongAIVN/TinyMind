"""Tests for RMSNorm."""

import torch

from slm.model.rmsnorm import RMSNorm


def test_output_shape_matches_input() -> None:
    norm = RMSNorm(hidden_size=32)
    x = torch.randn(2, 5, 32)
    assert norm(x).shape == x.shape


def test_matches_manual_formula() -> None:
    # Arrange
    norm = RMSNorm(hidden_size=16, eps=1e-6)
    x = torch.randn(3, 16, dtype=torch.float64).float()

    # Act
    output = norm(x)

    # Assert: y = x / sqrt(mean(x^2) + eps) * weight, weight starts at 1
    expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_unit_rms_output() -> None:
    """After normalization (weight=1), the RMS of each row is ~1."""
    norm = RMSNorm(hidden_size=64)
    x = torch.randn(10, 64) * 5.0  # deliberately large scale
    output = norm(x)
    rms = output.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones(10), rtol=1e-3, atol=1e-3)


def test_zero_input_is_safe() -> None:
    """eps must prevent NaN on an all-zero row."""
    norm = RMSNorm(hidden_size=8)
    output = norm(torch.zeros(1, 8))
    assert not torch.isnan(output).any()


def test_preserves_half_precision_dtype() -> None:
    norm = RMSNorm(hidden_size=8)
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    assert norm(x).dtype == torch.bfloat16
