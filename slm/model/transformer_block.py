"""A single pre-norm Transformer decoder block.

Purpose:
    Combine attention (token mixing) and the SwiGLU MLP (per-token
    computation) with residual connections.

Structure (pre-norm, as in LLaMA/GPT-NeoX):
    x = x + Attention(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))

    Pre-norm (normalizing *before* each sublayer) keeps the residual
    stream un-normalized, which makes deep stacks trainable without
    learning-rate warmup tricks that post-norm requires.

Input shape:  (batch, seq_len, hidden_size)
Output shape: (batch, seq_len, hidden_size)

Complexity:
    Attention O(T^2 * C) + MLP O(T * C * I) per sequence.
"""

import torch
import torch.nn as nn

from slm.model.attention import MultiHeadAttention
from slm.model.cache import KVCache
from slm.model.mlp import SwiGLU
from slm.model.rmsnorm import RMSNorm
from slm.utils.config import ModelConfig


class TransformerBlock(nn.Module):
    """Pre-norm decoder block: attention + SwiGLU, each with a residual."""

    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attention = MultiHeadAttention(config, layer_index)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Apply one decoder block.

        Args:
            x: ``(batch, seq_len, hidden_size)``.
            rope_cos: RoPE cosine table.
            rope_sin: RoPE sine table.
            kv_cache: Optional cache for generation.
            position_offset: Absolute position of ``x[:, 0]``.

        Returns:
            ``(batch, seq_len, hidden_size)``.
        """
        x = x + self.attention(
            self.attention_norm(x), rope_cos, rope_sin, kv_cache, position_offset
        )
        x = x + self.mlp(self.mlp_norm(x))
        return x
