"""Tests for config serialization and checkpoint save/load."""

from pathlib import Path

import torch

from slm.model.transformer import Transformer
from slm.utils.checkpoint import load_checkpoint, save_checkpoint
from slm.utils.config import Config, ModelConfig
from slm.utils.seed import set_seed


def test_config_json_roundtrip(tmp_path: Path) -> None:
    config = Config()
    config_path = tmp_path / "config.json"
    config.to_json(config_path)
    loaded = Config.from_json(config_path)
    assert loaded == config


def test_checkpoint_roundtrip(tmp_path: Path, tiny_config: ModelConfig) -> None:
    # Arrange: a model, optimizer, scheduler with some state
    set_seed(123)
    model = Transformer(tiny_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    tokens = torch.randint(0, tiny_config.vocab_size, (2, 8))
    model(tokens).sum().backward()
    optimizer.step()
    scheduler.step()

    # Act: save, then load into a fresh model
    checkpoint_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        checkpoint_path, model, optimizer, scheduler,
        step=7, epoch=2, config={"model": tiny_config.__dict__},
    )

    fresh_model = Transformer(tiny_config)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    fresh_scheduler = torch.optim.lr_scheduler.LambdaLR(fresh_optimizer, lambda step: 1.0)
    checkpoint = load_checkpoint(
        checkpoint_path, fresh_model, fresh_optimizer, fresh_scheduler
    )

    # Assert: counters and weights restored exactly
    assert checkpoint["step"] == 7
    assert checkpoint["epoch"] == 2
    for original, restored in zip(model.parameters(), fresh_model.parameters()):
        torch.testing.assert_close(original, restored)


def test_checkpoint_restores_forward_outputs(tmp_path: Path, tiny_config: ModelConfig) -> None:
    model = Transformer(tiny_config)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    checkpoint_path = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint_path, model, optimizer, scheduler, 0, 0, {})

    fresh_model = Transformer(tiny_config)
    load_checkpoint(checkpoint_path, fresh_model)

    tokens = torch.randint(0, tiny_config.vocab_size, (1, 6))
    with torch.no_grad():
        torch.testing.assert_close(model(tokens), fresh_model(tokens))
