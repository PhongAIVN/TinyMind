"""Lịch learning rate: warmup tuyến tính rồi cosine."""

import math


def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Learning rate tại `step` (đếm từ 0).

    Công thức:
        step < warmup        : max_lr · (step + 1) / warmup
        warmup <= step < max : min_lr + 0.5·(max_lr − min_lr)·(1 + cos(π · tiến_độ))
        step >= max          : min_lr
        với tiến_độ = (step − warmup) / (max − warmup)
    """
    if step < 0:
        raise ValueError(f"step={step} phải >= 0")
    if max_lr < min_lr:
        raise ValueError(f"max_lr={max_lr} phải >= min_lr={min_lr}")
    if not 0 <= warmup_steps <= max_steps:
        raise ValueError(f"warmup_steps={warmup_steps} phải trong [0, max_steps={max_steps}]")

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))