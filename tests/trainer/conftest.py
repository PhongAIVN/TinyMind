"""Fixture dùng chung cho test phần trainer."""

import numpy as np
import pytest

from core.config import ModelConfig
from trainer.pretrain_dataset import DTYPE
from trainer.train_config import TrainConfig

VOCAB = 64
N_TOKENS = 2000


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB,
        d_model=32,
        n_layers=1,
        n_heads=2,
        ffn_hidden=64,
        max_seq_len=32,
        rope_theta=10000.0,
        norm_eps=1e-6,
        tie_embedding=True,
    )


@pytest.fixture
def token_bins(tmp_path):
    """train.bin và eval.bin chứa token ngẫu nhiên có seed cố định."""
    rng = np.random.default_rng(0)
    paths = {}
    for name in ("train", "eval"):
        path = tmp_path / f"{name}.bin"
        rng.integers(0, VOCAB, size=N_TOKENS, dtype=DTYPE).tofile(path)
        paths[name] = path
    return paths


@pytest.fixture
def make_train_config(tmp_path, token_bins):
    """Trả về hàm tạo TrainConfig, cho phép ghi đè từng trường."""

    def _make(**overrides) -> TrainConfig:
        defaults = dict(
            train_bin=str(token_bins["train"]),
            eval_bin=str(token_bins["eval"]),
            out_dir=str(tmp_path / "out"),
            seq_len=16,
            batch_size=2,
            accum_steps=2,
            max_steps=3,
            warmup_steps=1,
            lr=1e-3,
            min_lr=1e-4,
            weight_decay=0.1,
            grad_clip=1.0,
            log_interval=1000,
            eval_interval=1000,
            eval_batches=2,
            save_interval=1000,
            dtype="fp32",
            device="cpu",
            seed=0,
        )
        defaults.update(overrides)
        return TrainConfig(**defaults)

    return _make