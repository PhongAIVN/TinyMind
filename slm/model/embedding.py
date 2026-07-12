"""Token embedding.

Purpose:
    Map discrete token ids to dense vectors the Transformer can process.
    This is a lookup table of shape (vocab_size, hidden_size). No position
    information is added here — positions are handled by RoPE inside
    attention.

Input shape:  (batch, seq_len) integer token ids
Output shape: (batch, seq_len, hidden_size)

Complexity:
    O(batch * seq_len) lookups; memory O(vocab_size * hidden_size).
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Learned token-id → vector lookup table."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for a batch of token id sequences.

        Args:
            token_ids: Integer tensor of shape ``(batch, seq_len)``.

        Returns:
            Float tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        return self.embedding(token_ids)

    @property
    def weight(self) -> torch.Tensor:
        """Expose the raw table so the LM head can tie to it."""
        return self.embedding.weight
