"""Tests for loss, optimizer, and LR scheduler."""

import math

import torch

from slm.train.loss import IGNORE_INDEX, cross_entropy_loss
from slm.train.optimizer import build_optimizer
from slm.train.scheduler import build_scheduler, compute_lr_scale
from slm.utils.config import ModelConfig, TrainConfig


def test_loss_is_log_vocab_for_uniform_logits() -> None:
    """Uniform logits → loss = ln(vocab_size) exactly."""
    vocab_size = 50
    logits = torch.zeros(2, 4, vocab_size)
    targets = torch.randint(0, vocab_size, (2, 4))
    loss = cross_entropy_loss(logits, targets)
    assert abs(loss.item() - math.log(vocab_size)) < 1e-5


def test_loss_ignores_masked_positions() -> None:
    logits = torch.randn(1, 4, 10)
    targets = torch.tensor([[1, IGNORE_INDEX, IGNORE_INDEX, 3]])
    only_valid = torch.tensor([[1, 3]])
    loss_masked = cross_entropy_loss(logits, targets)
    loss_manual = cross_entropy_loss(logits[:, [0, 3]], only_valid)
    torch.testing.assert_close(loss_masked, loss_manual)


def test_optimizer_weight_decay_groups(tiny_config: ModelConfig) -> None:
    """Matrices get weight decay; norms (1-D) do not."""
    from slm.model.transformer import Transformer

    model = Transformer(tiny_config)
    optimizer = build_optimizer(model, TrainConfig(weight_decay=0.1))

    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay_group["params"])
    assert all(p.dim() < 2 for p in no_decay_group["params"])


def test_lr_warmup_is_linear() -> None:
    assert compute_lr_scale(0, warmup_steps=100, max_steps=1000, min_lr_ratio=0.1) == 0.0
    assert compute_lr_scale(50, warmup_steps=100, max_steps=1000, min_lr_ratio=0.1) == 0.5
    assert compute_lr_scale(100, warmup_steps=100, max_steps=1000, min_lr_ratio=0.1) == 1.0


def test_lr_decays_to_minimum() -> None:
    scale = compute_lr_scale(1000, warmup_steps=100, max_steps=1000, min_lr_ratio=0.1)
    assert abs(scale - 0.1) < 1e-9
    # Past max_steps the LR stays at the floor.
    assert compute_lr_scale(5000, 100, 1000, 0.1) == 0.1


def test_lr_midpoint_of_cosine() -> None:
    """Halfway through decay, cosine sits exactly between peak and floor."""
    scale = compute_lr_scale(550, warmup_steps=100, max_steps=1000, min_lr_ratio=0.1)
    assert abs(scale - 0.55) < 1e-9


def test_scheduler_integrates_with_optimizer() -> None:
    parameter = torch.nn.Parameter(torch.zeros(4, 4))
    config = TrainConfig(learning_rate=1.0, min_learning_rate=0.1,
                         warmup_steps=10, max_steps=100)
    optimizer = torch.optim.AdamW([parameter], lr=config.learning_rate)
    scheduler = build_scheduler(optimizer, config)

    # LambdaLR applies the step-0 scale immediately: warmup starts at 0.
    assert scheduler.get_last_lr()[0] == 0.0
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert abs(scheduler.get_last_lr()[0] - 1.0) < 1e-9
