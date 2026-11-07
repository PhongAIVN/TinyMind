import math

import pytest
import torch

from core.rmsnorm import RMSNorm

DIM = 16
EPS = 1e-6


def test_output_shape_is_unchanged():
    norm = RMSNorm(DIM, EPS)
    x = torch.randn(2, 5, DIM)

    assert norm(x).shape == (2, 5, DIM)


def test_matches_hand_calculation():
    norm = RMSNorm(dim=2, eps=EPS)
    x = torch.tensor([[3.0, 4.0]])

    expected = x / math.sqrt(12.5)  # mean(9, 16) = 12.5

    torch.testing.assert_close(norm(x), expected)


def test_output_rms_is_one():
    norm = RMSNorm(DIM, EPS)
    x = torch.randn(2, 5, DIM) * 7.0

    rms = norm(x).pow(2).mean(dim=-1).sqrt()

    torch.testing.assert_close(rms, torch.ones(2, 5))


@pytest.mark.parametrize("scale", [0.5, 10.0, 1000.0])
def test_scaling_input_does_not_change_output(scale):
    norm = RMSNorm(DIM, EPS)
    x = torch.randn(2, 5, DIM)

    torch.testing.assert_close(norm(x * scale), norm(x))


def test_weight_scales_output():
    norm = RMSNorm(DIM, EPS)
    x = torch.randn(2, 5, DIM)
    base = norm(x)

    with torch.no_grad():
        norm.weight.fill_(2.0)

    torch.testing.assert_close(norm(x), 2.0 * base)


def test_each_token_is_normalized_independently():
    norm = RMSNorm(DIM, EPS)
    x = torch.randn(2, 5, DIM)
    x_changed = x.clone()
    x_changed[0, 0] += 100.0

    torch.testing.assert_close(norm(x_changed)[0, 1:], norm(x)[0, 1:])
    torch.testing.assert_close(norm(x_changed)[1], norm(x)[1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_output_keeps_input_dtype(dtype):
    norm = RMSNorm(DIM, EPS)
    x = torch.randn(2, 5, DIM).to(dtype)

    assert norm(x).dtype == dtype


def test_large_fp16_input_does_not_overflow():
    norm = RMSNorm(4, EPS)
    x = torch.full((1, 4), 1000.0, dtype=torch.float16)  # 1000² = 1e6 > giới hạn fp16 (65504)

    y = norm(x)

    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.float(), torch.ones(1, 4))


def test_weight_receives_gradient():
    norm = RMSNorm(DIM, EPS)
    norm(torch.randn(2, 5, DIM)).sum().backward()

    assert norm.weight.grad is not None