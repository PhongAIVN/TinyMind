"""The full decoder-only Transformer language model.

Purpose:
    Map a sequence of token ids to next-token logits.

Pipeline:
    token ids
      → TokenEmbedding                     (batch, T, C)
      → N × TransformerBlock (RoPE inside) (batch, T, C)
      → final RMSNorm                      (batch, T, C)
      → LM head (linear to vocab)          (batch, T, vocab_size)

Weight tying:
    When ``config.tie_embeddings`` is True the LM head shares the
    embedding table (logits = h @ E^T). For a small model the embedding
    is a large fraction of all parameters (32000 × 512 ≈ 16.4M), so
    tying saves that much and usually helps quality at this scale.

Complexity:
    O(num_layers * (T^2 * C + T * C * I)) per sequence, plus
    O(T * C * vocab_size) for the LM head.
"""

import torch
import torch.nn as nn

from slm.model.cache import KVCache
from slm.model.embedding import TokenEmbedding
from slm.model.rmsnorm import RMSNorm
from slm.model.rope import precompute_rope_cos_sin
from slm.model.transformer_block import TransformerBlock
from slm.utils.config import ModelConfig


class Transformer(nn.Module):
    """Decoder-only Transformer LM with RoPE, RMSNorm, and SwiGLU."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            TransformerBlock(config, layer_index) for layer_index in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_embeddings:
            # Share one weight tensor between input embedding and output head.
            self.lm_head.weight = self.embedding.embedding.weight

        # RoPE tables are deterministic functions of the config — computed
        # once and registered as (non-persistent) buffers so they follow
        # the model across devices but are not saved in checkpoints.
        rope_cos, rope_sin = precompute_rope_cos_sin(
            config.head_dim, config.context_length, config.rope_theta
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear/embedding weights with a small normal.

        std = 0.02 is the GPT-2 convention; small enough that residual
        streams start near zero and softmax attention starts near uniform.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Compute next-token logits for every position.

        Args:
            token_ids: ``(batch, seq_len)`` integer tensor.
            kv_cache: Optional KV cache for incremental generation.
            position_offset: Absolute position of ``token_ids[:, 0]``
                (nonzero only during cached generation).

        Returns:
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        seq_len = token_ids.shape[1]
        if position_offset + seq_len > self.config.context_length:
            raise ValueError(
                f"Sequence of length {seq_len} at offset {position_offset} exceeds "
                f"context_length {self.config.context_length}"
            )

        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin, kv_cache, position_offset)
        x = self.final_norm(x)
        return self.lm_head(x)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Count trainable parameters (tied weights counted once)."""
        seen: set[int] = set()
        total = 0
        for parameter in self.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            total += parameter.numel()
        if exclude_embeddings:
            total -= self.embedding.embedding.weight.numel()
        return total
