"""Tests for Rotary Position Embedding."""

import torch

from slm.model.rope import apply_rope, precompute_rope_cos_sin


def test_table_shapes() -> None:
    cos, sin = precompute_rope_cos_sin(head_dim=16, max_positions=32)
    assert cos.shape == (32, 8)
    assert sin.shape == (32, 8)


def test_position_zero_is_identity() -> None:
    """At position 0 the rotation angle is 0, so nothing changes."""
    cos, sin = precompute_rope_cos_sin(head_dim=8, max_positions=4)
    x = torch.randn(1, 2, 1, 8)  # single token at position 0
    torch.testing.assert_close(apply_rope(x, cos, sin), x)


def test_rotation_preserves_norm() -> None:
    """Rotations are orthogonal: vector lengths must not change."""
    cos, sin = precompute_rope_cos_sin(head_dim=16, max_positions=10)
    x = torch.randn(2, 4, 10, 16)
    rotated = apply_rope(x, cos, sin)
    torch.testing.assert_close(
        rotated.norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-5
    )


def test_relative_position_property() -> None:
    """The core RoPE property: <rot(q, m), rot(k, n)> depends only on m - n.

    Rotate the same q, k pair at positions (2, 5) and (10, 13): both have
    offset 3, so the dot products must match.
    """
    cos, sin = precompute_rope_cos_sin(head_dim=32, max_positions=64)
    q = torch.randn(1, 1, 1, 32)
    k = torch.randn(1, 1, 1, 32)

    def dot_at(query_pos: int, key_pos: int) -> torch.Tensor:
        q_rot = apply_rope(q, cos, sin, position_offset=query_pos)
        k_rot = apply_rope(k, cos, sin, position_offset=key_pos)
        return (q_rot * k_rot).sum()

    torch.testing.assert_close(dot_at(2, 5), dot_at(10, 13), rtol=1e-4, atol=1e-4)


def test_position_offset_matches_full_sequence() -> None:
    """Rotating token t alone with offset t must equal rotating in-sequence."""
    cos, sin = precompute_rope_cos_sin(head_dim=8, max_positions=16)
    x = torch.randn(1, 2, 6, 8)

    full = apply_rope(x, cos, sin)
    single = apply_rope(x[:, :, 3:4], cos, sin, position_offset=3)
    torch.testing.assert_close(single, full[:, :, 3:4])
