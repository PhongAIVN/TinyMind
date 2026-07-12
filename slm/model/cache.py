"""KV cache for fast autoregressive generation.

Purpose:
    During generation, token t attends to keys/values of tokens 0..t.
    Without a cache, each new token recomputes K and V for the whole
    prefix — O(t) redundant work per step, O(T^2) total. The cache
    stores K and V once per layer, so each step only computes K/V for
    the single new token.

Shapes:
    Per layer, keys and values are stored as
    ``(batch, num_heads, max_seq_len, head_dim)``, preallocated once so
    generation never reallocates memory.

Complexity:
    ``update`` is O(batch * num_heads * new_len * head_dim) copies.
    Memory is O(num_layers * batch * num_heads * max_seq_len * head_dim).
"""

import torch


class KVCache:
    """Preallocated per-layer key/value storage for generation."""

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.keys = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.values = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.max_seq_len = max_seq_len
        # Number of positions already filled (same for every layer).
        self.current_length = 0

    def update(
        self,
        layer_index: int,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        start_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V for one layer and return the full valid prefix.

        Args:
            layer_index: Which Transformer layer this K/V belongs to.
            new_keys: ``(batch, num_heads, new_len, head_dim)``.
            new_values: ``(batch, num_heads, new_len, head_dim)``.
            start_position: Absolute position of the first new token.

        Returns:
            Tuple ``(keys, values)`` covering positions ``0 .. start + new_len``,
            each of shape ``(batch, num_heads, start + new_len, head_dim)``.
        """
        new_len = new_keys.shape[2]
        end_position = start_position + new_len
        if end_position > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: writing positions {start_position}:{end_position} "
                f"but max_seq_len is {self.max_seq_len}"
            )

        self.keys[layer_index][:, :, start_position:end_position] = new_keys
        self.values[layer_index][:, :, start_position:end_position] = new_values

        # The transformer updates every layer with the same positions per
        # forward pass; tracking length once (on the last layer) is enough.
        if layer_index == len(self.keys) - 1:
            self.current_length = end_position

        return (
            self.keys[layer_index][:, :, :end_position],
            self.values[layer_index][:, :, :end_position],
        )
