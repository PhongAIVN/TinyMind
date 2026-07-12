"""Checkpoint saving and loading.

A checkpoint stores everything needed to resume training exactly:
model weights, optimizer state, scheduler state, step/epoch counters,
the full config, and all RNG states.
"""

from pathlib import Path

import torch

from slm.utils.seed import get_rng_state, set_rng_state


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    epoch: int,
    config: dict,
) -> None:
    """Save a full training checkpoint to ``path``.

    The parent directory is created if it does not exist. The file is
    written atomically (write to a temp file, then rename) so that a
    crash mid-write never corrupts an existing checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": config,
        "rng_state": get_rng_state(),
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: str = "cpu",
    restore_rng: bool = True,
) -> dict:
    """Load a checkpoint into the given model (and optionally optimizer state).

    Returns the raw checkpoint dict so the caller can read ``step``,
    ``epoch``, and ``config``.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if restore_rng and "rng_state" in checkpoint:
        set_rng_state(checkpoint["rng_state"])

    return checkpoint
