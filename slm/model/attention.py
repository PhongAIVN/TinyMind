"""Multi-head causal self-attention, implemented from scratch.

Purpose:
    Let every token gather information from earlier tokens. Each of the
    ``num_heads`` heads learns its own projection subspace, so different
    heads can attend to different kinds of relationships.

Algorithm (per head):
    1. Project input x into queries Q, keys K, values V (no bias).
    2. Rotate Q and K with RoPE so scores encode relative positions.
    3. scores = Q @ K^T / sqrt(head_dim)
    4. Causal mask: position m may only attend to positions n <= m,
       enforced by setting future scores to -inf before softmax.
    5. weights = softmax(scores);  output = weights @ V
    6. Concatenate heads and apply the output projection.

Shapes (B=batch, T=seq_len, H=num_heads, D=head_dim, C=hidden_size=H*D):
    input x:        (B, T, C)
    Q, K, V:        (B, H, T, D)
    scores/weights: (B, H, T, T)   — (B, H, T, T_total) with KV cache
    output:         (B, T, C)

Complexity:
    Time  O(B * H * T^2 * D) — the T^2 term dominates for long sequences.
    Memory O(B * H * T^2) for the attention matrix (this is what
    FlashAttention later removes; see Phase 3).

Numerical stability:
    - Scaling by 1/sqrt(D) keeps score variance ~1 so softmax does not
      saturate at initialization.
    - Softmax runs in float32 (PyTorch's softmax subtracts the row max
      internally, preventing exp overflow).
    - The mask uses -inf, which softmax maps to exactly 0.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.model.cache import KVCache
from slm.model.rope import apply_rope
from slm.utils.config import ModelConfig


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention with RoPE and optional KV cache."""

    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.layer_index = layer_index

        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Run causal self-attention over ``x``.

        Args:
            x: Input activations of shape ``(batch, seq_len, hidden_size)``.
            rope_cos: RoPE cosine table ``(max_positions, head_dim / 2)``.
            rope_sin: RoPE sine table ``(max_positions, head_dim / 2)``.
            kv_cache: If given, K/V are appended to the cache and attention
                runs over the full cached prefix (generation mode).
            position_offset: Absolute position of ``x[:, 0]``; nonzero only
                during cached generation.

        Returns:
            Tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        batch_size, seq_len, hidden_size = x.shape

        # (B, T, C) -> (B, T, H, D) -> (B, H, T, D)
        queries = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Rotate Q and K by their absolute positions (V carries content
        # only, so it is not rotated).
        queries = apply_rope(queries, rope_cos, rope_sin, position_offset)
        keys = apply_rope(keys, rope_cos, rope_sin, position_offset)

        if kv_cache is not None:
            keys, values = kv_cache.update(self.layer_index, keys, values, position_offset)

        # (B, H, T, D) @ (B, H, D, T_total) -> (B, H, T, T_total)
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores + self._causal_mask(seq_len, keys.shape[2], position_offset, x.device)

        # Softmax in float32 for stability, then back to the working dtype.
        weights = F.softmax(scores.to(torch.float32), dim=-1).to(values.dtype)

        # (B, H, T, T_total) @ (B, H, T_total, D) -> (B, H, T, D)
        output = weights @ values

        # (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.out_proj(output)

    @staticmethod
    def _causal_mask(
        query_len: int,
        key_len: int,
        position_offset: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build an additive mask that blocks attention to future positions.

        Query i sits at absolute position ``position_offset + i`` and may
        attend to key positions ``<= position_offset + i``. Allowed entries
        are 0, blocked entries are -inf (softmax turns them into 0 weight).

        Returns:
            Mask of shape ``(1, 1, query_len, key_len)`` for broadcasting.
        """
        query_positions = torch.arange(query_len, device=device).unsqueeze(1) + position_offset
        key_positions = torch.arange(key_len, device=device).unsqueeze(0)
        allowed = key_positions <= query_positions  # (query_len, key_len)

        mask = torch.zeros(query_len, key_len, device=device)
        mask.masked_fill_(~allowed, float("-inf"))
        return mask.view(1, 1, query_len, key_len)
