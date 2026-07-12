"""Learning-rate schedule: linear warmup then cosine decay.

Purpose:
    - Warmup: Adam's moment estimates are garbage for the first steps;
      starting at full LR with random moments destabilizes training.
      LR ramps linearly 0 → peak over ``warmup_steps``.
    - Cosine decay: after warmup, LR follows half a cosine from peak
      down to ``min_lr``, finishing at ``max_steps``. The slow initial
      decline and gentle landing empirically beat step decays for LMs.

Formula (step s, warmup W, total S):
    s <  W:  lr = peak * s / W
    s >= W:  progress = (s - W) / (S - W)
             lr = min + 0.5 * (peak - min) * (1 + cos(pi * progress))
"""

import math

import torch

from slm.utils.config import TrainConfig


def compute_lr_scale(step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float) -> float:
    """Return the LR multiplier (relative to peak LR) for a given step.

    Args:
        step: Current optimizer step (0-based).
        warmup_steps: Steps of linear warmup.
        max_steps: Total training steps; LR reaches its floor here.
        min_lr_ratio: Floor LR as a fraction of peak LR.

    Returns:
        Multiplier in [0, 1] to apply to the peak learning rate.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps

    if step >= max_steps:
        return min_lr_ratio

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    """Wrap :func:`compute_lr_scale` in a PyTorch LambdaLR scheduler."""
    min_lr_ratio = config.min_learning_rate / config.learning_rate

    def lr_lambda(step: int) -> float:
        return compute_lr_scale(step, config.warmup_steps, config.max_steps, min_lr_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
