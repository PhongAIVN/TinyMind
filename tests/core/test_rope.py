import pytest
import torch

from core.rope import apply_rope, precompute_rope

HEAD_DIM = 8
MAX_LEN = 16
THETA = 10000.0


def test_table_shapes():
    cos, sin = precompute_rope(HEAD_DIM, MAX_LEN, THETA)

    assert cos.shape == (MAX_LEN, HEAD_DIM // 2)
    assert sin.shape == (MAX_LEN, HEAD_DIM // 2)


def test_odd_head_dim_raises():
    with pytest.raises(ValueError, match="head_dim"):
        precompute_rope(7, MAX_LEN, THETA)


def test_output_shape_is_unchanged():
    cos, sin = precompute_rope(HEAD_DIM, MAX_LEN, THETA)
    x = torch.randn(2, 4, MAX_LEN, HEAD_DIM)  # (B, h, T, hd)

    assert apply_rope(x, cos, sin).shape == x.shape


def test_position_zero_is_not_rotated():
    cos, sin = precompute_rope(HEAD_DIM, MAX_LEN, THETA)
    x = torch.randn(1, 1, 1, HEAD_DIM)

    torch.testing.assert_close(apply_rope(x, cos[:1], sin[:1]), x)


def test_rotation_preserves_vector_length():
    cos, sin = precompute_rope(HEAD_DIM, MAX_LEN, THETA)
    x = torch.randn(2, 4, MAX_LEN, HEAD_DIM)

    y = apply_rope(x, cos, sin)

    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))


def test_dot_product_depends_only_on_relative_position():
    cos, sin = precompute_rope(HEAD_DIM, MAX_LEN, THETA)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)

    def rotated_dot(m: int, n: int) -> torch.Tensor:
        q_m = apply_rope(q, cos[m], sin[m])
        k_n = apply_rope(k, cos[n], sin[n])
        return (q_m * k_n).sum()

    # Cùng khoảng cách 2 → bằng nhau
    torch.testing.assert_close(rotated_dot(1, 3), rotated_dot(5, 7))
    # Khoảng cách khác → khác nhau
    assert not torch.isclose(rotated_dot(1, 3), rotated_dot(1, 4))