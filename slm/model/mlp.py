"""SwiGLU feed-forward network.

Purpose:
    The position-wise MLP is where the Transformer stores most of its
    learned knowledge. SwiGLU replaces the classic ReLU MLP with a
    *gated* variant that consistently trains better (used by LLaMA,
    PaLM, Mistral).

Formula:
    SwiGLU(x) = W_down( SiLU(W_gate x) ⊙ (W_up x) )

    where SiLU(z) = z * sigmoid(z) and ⊙ is elementwise multiply.
    The SiLU branch acts as a smooth, learned gate deciding how much of
    the W_up branch passes through, per feature.

Shapes (C=hidden_size, I=intermediate_size):
    input:   (batch, seq_len, C)
    gate/up: (batch, seq_len, I)
    output:  (batch, seq_len, C)

Complexity:
    O(batch * seq_len * C * I) — three matmuls of that size. With
    I = 3C, the MLP holds ~2/3 of each block's parameters.

Numerical stability:
    SiLU is smooth and bounded below (~ -0.278), so no special handling
    is needed beyond the usual mixed-precision care.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.utils.config import ModelConfig


class SwiGLU(nn.Module):
    """Gated feed-forward: down( silu(gate(x)) * up(x) )."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated MLP position-wise.

        Args:
            x: ``(batch, seq_len, hidden_size)``.

        Returns:
            ``(batch, seq_len, hidden_size)``.
        """
        gate = F.silu(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))
