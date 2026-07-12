"""Tests for the full Transformer model."""

import pytest
import torch

from slm.model.transformer import Transformer
from slm.utils.config import ModelConfig


def test_forward_shape(tiny_config: ModelConfig) -> None:
    model = Transformer(tiny_config)
    tokens = torch.randint(0, tiny_config.vocab_size, (2, 12))
    logits = model(tokens)
    assert logits.shape == (2, 12, tiny_config.vocab_size)


def test_weight_tying(tiny_config: ModelConfig) -> None:
    model = Transformer(tiny_config)
    assert model.lm_head.weight is model.embedding.embedding.weight


def test_untied_weights() -> None:
    config = ModelConfig(
        vocab_size=128, num_layers=1, hidden_size=32, num_heads=2,
        head_dim=16, intermediate_size=64, context_length=32,
        tie_embeddings=False,
    )
    model = Transformer(config)
    assert model.lm_head.weight is not model.embedding.embedding.weight


def test_rejects_sequences_beyond_context(tiny_config: ModelConfig) -> None:
    model = Transformer(tiny_config)
    too_long = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.context_length + 1))
    with pytest.raises(ValueError, match="exceeds"):
        model(too_long)


def test_gradients_flow_to_all_parameters(tiny_config: ModelConfig) -> None:
    """One backward pass must touch every trainable parameter."""
    model = Transformer(tiny_config)
    tokens = torch.randint(0, tiny_config.vocab_size, (2, 8))
    model(tokens).sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"no gradient for {name}"


def test_parameter_count_tied_vs_untied(tiny_config: ModelConfig) -> None:
    tied = Transformer(tiny_config).num_parameters()
    untied_config = ModelConfig(**{**tiny_config.__dict__, "tie_embeddings": False})
    untied = Transformer(untied_config).num_parameters()
    embedding_size = tiny_config.vocab_size * tiny_config.hidden_size
    assert untied - tied == embedding_size


def test_invalid_head_geometry_raises() -> None:
    with pytest.raises(ValueError, match="must equal hidden_size"):
        ModelConfig(hidden_size=512, num_heads=8, head_dim=32)
