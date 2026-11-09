"""Tham số cho các chặng RL (GRPO, CISPO, PPO, Agentic RL). Khóa lạ hoặc thiếu khóa bắt buộc -> báo lỗi."""

import json
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any

ALGOS = ("grpo", "cispo", "ppo")
MODES = ("plain", "think", "no_think", "tools")
DTYPES = ("fp32", "fp16", "bf16")
ATTENTION = ("manual", "sdpa")


@dataclass(frozen=True)
class RLConfig:
    prompts: str  # jsonl {"messages": [{"role": "user", ...}, ...], "result": ...}
    eval_prompts: str
    init_from: str  # policy và ref cùng khởi đầu từ đây
    out_dir: str
    checkpoint_name: str
    algo: str
    mode: str  # plain / think / no_think / tools (agentic)
    steps: int
    prompts_per_step: int = 8
    group_size: int = 8  # số câu trả lời mỗi câu hỏi (GRPO); PPO thường để 1
    max_new_tokens: int = 64
    temperature: float = 1.0
    lr: float = 1e-6
    min_lr: float = 1e-7
    warmup_steps: int = 5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    beta_kl: float = 0.02
    clip_eps: float = 0.2
    clip_eps_high: float = 0.28  # CISPO: chặn trên của trọng số importance sampling
    inner_epochs: int = 1  # số lần cập nhật trên mỗi lô rollout
    gamma: float = 1.0  # PPO
    lam: float = 0.95  # PPO
    critic_lr: float = 1e-5  # PPO
    log_interval: int = 1
    eval_interval: int = 20
    eval_limit: int = 100
    save_interval: int = 50
    dtype: str = "bf16"
    attention: str = "sdpa"
    device: str = "cuda"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.algo not in ALGOS:
            raise ValueError(f"algo={self.algo!r} phải thuộc {ALGOS}")
        if self.mode not in MODES:
            raise ValueError(f"mode={self.mode!r} phải thuộc {MODES}")
        if self.dtype not in DTYPES or self.attention not in ATTENTION:
            raise ValueError(f"dtype/attention không hợp lệ: {self.dtype!r}, {self.attention!r}")
        for name in ("steps", "prompts_per_step", "group_size", "max_new_tokens", "inner_epochs",
                     "log_interval", "eval_interval", "save_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name}={value!r} phải là int > 0")
        if self.algo != "ppo" and self.group_size < 2:
            raise ValueError("GRPO/CISPO cần group_size >= 2 (advantage so trong nhóm)")
        if not 0 < self.min_lr <= self.lr or self.temperature <= 0:
            raise ValueError("cần 0 < min_lr <= lr và temperature > 0")

    @classmethod
    def from_json(cls, path: str | Path) -> "RLConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        names = {field.name for field in fields(cls)}
        required = {field.name for field in fields(cls) if field.default is MISSING}
        errors = []
        if unknown := set(data) - names:
            errors.append(f"khóa không hợp lệ: {sorted(unknown)}")
        if missing := required - set(data):
            errors.append(f"khóa bị thiếu: {sorted(missing)}")
        if errors:
            raise ValueError("; ".join(errors))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
