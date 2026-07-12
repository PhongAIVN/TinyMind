"""Shared test fixtures: a tiny model config that runs fast on CPU."""

import pytest
import torch

from slm.utils.config import ModelConfig


@pytest.fixture()
def tiny_config() -> ModelConfig:
    """A miniature model: 2 layers, 64-dim, 128-token vocab."""
    return ModelConfig(
        vocab_size=128,
        num_layers=2,
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        intermediate_size=128,
        context_length=64,
    )


@pytest.fixture(autouse=True)
def fixed_seed() -> None:
    """Make every test deterministic."""
    torch.manual_seed(0)
