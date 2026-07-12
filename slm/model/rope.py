"""Rotary Position Embedding (RoPE).

Purpose:
    Inject position information into attention by *rotating* each
    query/key vector by an angle proportional to its position. After
    rotation, the dot product between a query at position m and a key
    at position n depends only on the offset (m - n), giving the model
    relative position awareness without any learned position table.

Math:
    Split the head dimension into d/2 pairs. Pair i gets frequency

        theta_i = base^(-2i / d)            (base = 10000 by default)

    A token at position p rotates pair i by angle (p * theta_i):

        [x_even']   [cos(p*theta_i)  -sin(p*theta_i)] [x_even]
        [x_odd' ] = [sin(p*theta_i)   cos(p*theta_i)] [x_odd ]

    Low i → high frequency (encodes nearby offsets), high i → low
    frequency (encodes long-range offsets).

Input shape:  q or k of shape (batch, num_heads, seq_len, head_dim)
Output shape: same as input

Complexity:
    O(batch * num_heads * seq_len * head_dim) — elementwise.

Numerical stability:
    cos/sin tables are precomputed once in float32 and cached; rotation
    itself is elementwise multiply-add, which is safe in half precision.
"""

import torch


def precompute_rope_cos_sin(
    head_dim: int,
    max_positions: int,
    theta_base: float = 10000.0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin rotation tables for all positions.

    Args:
        head_dim: Dimension of a single attention head (must be even).
        max_positions: Number of positions to precompute (context length).
        theta_base: Frequency base; larger = slower-decaying frequencies.
        device: Device to build the tables on.

    Returns:
        Tuple ``(cos, sin)``, each of shape ``(max_positions, head_dim / 2)``
        in float32, where entry ``[p, i]`` is cos/sin of ``p * theta_i``.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

    # theta_i = base^(-2i/d) for i in [0, d/2)
    pair_indices = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    frequencies = theta_base ** (-pair_indices / head_dim)  # (head_dim/2,)

    positions = torch.arange(max_positions, dtype=torch.float32, device=device)
    # Outer product: angles[p, i] = p * theta_i
    angles = torch.outer(positions, frequencies)  # (max_positions, head_dim/2)

    return angles.cos(), angles.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_offset: int = 0,
) -> torch.Tensor:
    """Rotate query or key vectors by their position-dependent angles.

    Args:
        x: Queries or keys of shape ``(batch, num_heads, seq_len, head_dim)``.
        cos: Precomputed cosines of shape ``(max_positions, head_dim / 2)``.
        sin: Precomputed sines of shape ``(max_positions, head_dim / 2)``.
        position_offset: Absolute position of the first token in ``x``.
            Used during cached generation, where ``x`` holds only the new
            token but its true position is ``offset``.

    Returns:
        Rotated tensor with the same shape and dtype as ``x``.
    """
    seq_len = x.shape[2]
    # Slice the table rows for the actual positions of these tokens,
    # then reshape to (1, 1, seq_len, head_dim/2) for broadcasting.
    cos = cos[position_offset : position_offset + seq_len].view(1, 1, seq_len, -1)
    sin = sin[position_offset : position_offset + seq_len].view(1, 1, seq_len, -1)

    # Interpret consecutive (even, odd) channels as 2D coordinates.
    x_even = x[..., 0::2].to(torch.float32)
    x_odd = x[..., 1::2].to(torch.float32)

    # Standard 2D rotation applied to every pair.
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    # Re-interleave pairs back into the original channel order.
    rotated = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
    return rotated.to(x.dtype)
