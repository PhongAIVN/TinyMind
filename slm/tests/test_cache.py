"""Tests for the KV cache — the key property: cached generation must
produce exactly the same logits as recomputing the full sequence."""

import pytest
import torch

from slm.model.cache import KVCache
from slm.model.transformer import Transformer
from slm.utils.config import ModelConfig


def _make_cache(config: ModelConfig, batch_size: int = 1) -> KVCache:
    return KVCache(
        num_layers=config.num_layers,
        batch_size=batch_size,
        num_heads=config.num_heads,
        max_seq_len=config.context_length,
        head_dim=config.head_dim,
        device="cpu",
    )


def test_cached_forward_matches_full_forward(tiny_config: ModelConfig) -> None:
    """Feed a sequence token by token through the cache; final-position
    logits must match a single full (uncached) forward pass."""
    model = Transformer(tiny_config)
    model.eval()
    tokens = torch.randint(0, tiny_config.vocab_size, (1, 10))

    with torch.no_grad():
        full_logits = model(tokens)

        cache = _make_cache(tiny_config)
        step_logits = []
        for position in range(tokens.shape[1]):
            logits = model(
                tokens[:, position : position + 1],
                kv_cache=cache,
                position_offset=position,
            )
            step_logits.append(logits)
        incremental_logits = torch.cat(step_logits, dim=1)

    torch.testing.assert_close(incremental_logits, full_logits, rtol=1e-4, atol=1e-4)


def test_prefill_then_decode_matches_full(tiny_config: ModelConfig) -> None:
    """Prefill 6 tokens at once, then decode 1 — the realistic pattern."""
    model = Transformer(tiny_config)
    model.eval()
    tokens = torch.randint(0, tiny_config.vocab_size, (2, 7))

    with torch.no_grad():
        full_logits = model(tokens)

        cache = _make_cache(tiny_config, batch_size=2)
        model(tokens[:, :6], kv_cache=cache, position_offset=0)
        last_logits = model(tokens[:, 6:7], kv_cache=cache, position_offset=6)

    torch.testing.assert_close(last_logits, full_logits[:, 6:7], rtol=1e-4, atol=1e-4)


def test_cache_overflow_raises(tiny_config: ModelConfig) -> None:
    cache = KVCache(
        num_layers=1, batch_size=1, num_heads=tiny_config.num_heads,
        max_seq_len=4, head_dim=tiny_config.head_dim, device="cpu",
    )
    shape = (1, tiny_config.num_heads, 5, tiny_config.head_dim)
    with pytest.raises(ValueError, match="overflow"):
        cache.update(0, torch.zeros(shape), torch.zeros(shape), start_position=0)


def test_current_length_tracking(tiny_config: ModelConfig) -> None:
    cache = _make_cache(tiny_config)
    shape = (1, tiny_config.num_heads, 3, tiny_config.head_dim)
    for layer_index in range(tiny_config.num_layers):
        cache.update(layer_index, torch.randn(shape), torch.randn(shape), start_position=0)
    assert cache.current_length == 3
