
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

def _rng_state() -> dict[str, Any]:
    """Trạng thái sinh số ngẫu nhiên của Python, NumPy, PyTorch."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_config: Any,
    train_config: Any,
    scaler: Any = None,
) -> None:
    """Lưu đủ mọi thứ để chạy tiếp đúng run này."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "model": {key: value.detach().cpu().float() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": _rng_state(),
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any = None,
    map_location: Any = "cpu",
) -> int:
    """Nạp lại toàn bộ trạng thái. Trả về step đã chạy xong."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    _restore_rng(payload["rng"])
    return int(payload["step"])


def load_weights(path: str | Path, model: nn.Module, map_location: Any = "cpu") -> None:
    """Chỉ nạp trọng số, bỏ optimizer và RNG — dùng khi bắt đầu giai đoạn mới."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])