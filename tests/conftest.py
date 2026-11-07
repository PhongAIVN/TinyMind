import pytest
import torch

from core.config import ModelConfig

@pytest.fixture(autouse=True)
def set_seed() -> None:
    """Thiết lập seed cho các phép thử."""
    torch.manual_seed(42)

@pytest.fixture
def tiny_config() -> ModelConfig:
    """Cấu hình mặc định cho các phép thử."""
    return ModelConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=4,
        ffn_hidden=128,
        max_seq_len=32,
        rope_theta=10000.0,
        norm_eps=1e-6,
        tie_embedding=True,
    )