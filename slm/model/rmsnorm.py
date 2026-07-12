"""Root Mean Square Layer Normalization (RMSNorm).

Purpose:
    Normalize activations so their scale stays stable across layers,
    which keeps gradients well-behaved during training. RMSNorm is a
    simplification of LayerNorm: it only rescales (no mean subtraction,
    no bias), which is cheaper and works just as well in Transformers.

Formula:
    RMS(x)     = sqrt(mean(x_i^2) + eps)
    RMSNorm(x) = (x / RMS(x)) * weight

    where the mean is taken over the last (feature) dimension and
    ``weight`` is a learned per-feature scale, initialized to 1.

Input shape:  (..., hidden_size)
Output shape: (..., hidden_size)  — unchanged

Complexity:
    O(N * hidden_size) for N tokens — one pass to compute the mean of
    squares, one pass to rescale.

Numerical stability:
    The mean of squares is computed in float32 even when the input is
    fp16/bf16, because squaring small half-precision values underflows
    easily. ``eps`` prevents division by zero for all-zero rows.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMS normalization over the last dimension with a learned scale."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` over its last dimension.

        Args:
            x: Activations of shape ``(..., hidden_size)``.

        Returns:
            Normalized tensor with the same shape and dtype as ``x``.
        """
        input_dtype = x.dtype
        x = x.to(torch.float32)

        # mean(x^2) over the feature dimension, kept for broadcasting.
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps)

        return (x * self.weight.to(torch.float32)).to(input_dtype)
