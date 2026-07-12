"""End-to-end smoke test: a few real training steps on synthetic data.

Verifies the whole stack wires together (dataset → model → loss →
optimizer → scheduler → checkpoint) and that loss decreases on an
easily learnable corpus.
"""

from pathlib import Path

import numpy as np
import torch

from slm.train.trainer import Trainer
from slm.utils.config import Config, ModelConfig, TrainConfig


def _write_synthetic_corpus(path: Path, num_tokens: int, vocab_size: int) -> None:
    """A repeating 0,1,2,...,15 pattern — trivially learnable."""
    pattern_length = min(16, vocab_size)
    tokens = np.arange(num_tokens, dtype=np.uint16) % pattern_length
    tokens.tofile(path)


def test_short_training_run_decreases_loss(tmp_path: Path) -> None:
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    _write_synthetic_corpus(train_bin, num_tokens=20000, vocab_size=64)
    _write_synthetic_corpus(val_bin, num_tokens=2000, vocab_size=64)

    config = Config(
        model=ModelConfig(
            vocab_size=64, num_layers=2, hidden_size=32, num_heads=2,
            head_dim=16, intermediate_size=64, context_length=32,
        ),
        train=TrainConfig(
            train_data_path=str(train_bin),
            val_data_path=str(val_bin),
            sequence_length=32,
            batch_size=4,
            gradient_accumulation_steps=1,
            max_steps=30,
            learning_rate=1e-2,
            min_learning_rate=1e-3,
            warmup_steps=5,
            precision="fp32",
            log_interval=10,
            eval_interval=1000,   # skip mid-run eval
            checkpoint_interval=1000,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        ),
    )

    trainer = Trainer(config, device="cpu")
    val_loader = trainer._build_loader(str(val_bin), shuffle=False)
    loss_before = trainer.evaluate(val_loader)

    trainer.train()

    loss_after = trainer.evaluate(val_loader)
    assert loss_after < loss_before, (
        f"loss did not decrease: {loss_before:.3f} -> {loss_after:.3f}"
    )
    assert (tmp_path / "checkpoints" / "final.pt").is_file()


def test_resume_restores_step(tmp_path: Path) -> None:
    train_bin = tmp_path / "train.bin"
    _write_synthetic_corpus(train_bin, num_tokens=5000, vocab_size=64)

    config = Config(
        model=ModelConfig(
            vocab_size=64, num_layers=1, hidden_size=32, num_heads=2,
            head_dim=16, intermediate_size=64, context_length=16,
        ),
        train=TrainConfig(
            train_data_path=str(train_bin),
            val_data_path=str(train_bin),
            sequence_length=16,
            batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=5,
            warmup_steps=1,
            precision="fp32",
            eval_interval=1000,
            checkpoint_interval=1000,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        ),
    )

    trainer = Trainer(config, device="cpu")
    trainer.train()  # runs 5 steps, saves final.pt

    resumed = Trainer(config, device="cpu")
    resumed.resume(tmp_path / "checkpoints" / "final.pt")
    assert resumed.step == 5
    with torch.no_grad():
        tokens = torch.randint(0, 64, (1, 8))
        torch.testing.assert_close(trainer.model(tokens), resumed.model(tokens))
