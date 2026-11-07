"""Tham số cho một lần chạy pretrain (khác ModelConfig: kiến trúc)."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

ALLOWED_DTYPES = ("fp32", "fp16", "bf16")
POSITIVE_INT_FIELDS = (
    "seq_len", "batch_size", "accum_steps", "max_steps",
    "log_interval", "eval_interval", "eval_batches", "save_interval",
)
POSITIVE_FLOAT_FIELDS = ("lr", "grad_clip")
@dataclass(frozen=True)
class TrainConfig:
    train_bin: str
    eval_bin: str
    out_dir: str
    seq_len: int
    batch_size: int
    accum_steps: int
    max_steps: int
    warmup_steps: int
    lr: float
    min_lr: float
    weight_decay: float
    grad_clip: float
    log_interval: int
    eval_interval: int
    eval_batches: int
    save_interval: int
    dtype: str
    device: str
    seed: int

    def __post_init__(self) -> None:
        for name in POSITIVE_INT_FIELDS + ("seed",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name}={value!r} phải là int")
            if name != "seed" and value <= 0:
                raise ValueError(f"{name}={value} phải > 0")

        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int):
            raise ValueError(f"warmup_steps={self.warmup_steps!r} phải là int")
        if not 0 <= self.warmup_steps <= self.max_steps:
            raise ValueError(
                f"warmup_steps={self.warmup_steps} phải trong [0, max_steps={self.max_steps}]"
            )

        for name in POSITIVE_FLOAT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name}={value!r} phải là số thực")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}={value} phải hữu hạn và > 0")

        if not 0 <= self.min_lr <= self.lr:
            raise ValueError(f"min_lr={self.min_lr} phải trong [0, lr={self.lr}]")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay={self.weight_decay} phải >= 0")
        if self.dtype not in ALLOWED_DTYPES:
            raise ValueError(f"dtype={self.dtype!r} phải thuộc {ALLOWED_DTYPES}")

    @property
    def tokens_per_step(self) -> int:
        """Số token model học trong 1 bước cập nhật."""
        return self.batch_size * self.accum_steps * self.seq_len

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        if not isinstance(data, dict):
            raise ValueError(f"from_dict cần dict, nhận được {type(data).__name__}")

        valid_names = {field.name for field in fields(cls)}
        unknown = set(data) - valid_names
        missing = valid_names - set(data)

        errors = []
        if unknown:
            errors.append(f"khóa không hợp lệ: {sorted(unknown)}")
        if missing:
            errors.append(f"khóa bị thiếu: {sorted(missing)}")
        if errors:
            raise ValueError("; ".join(errors))

        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
