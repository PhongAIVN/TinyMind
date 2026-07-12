"""Model and training configuration.

Configuration is kept in plain dataclasses so that every hyperparameter is
explicit and discoverable. Configs can be loaded from / saved to JSON files
(see ``slm/configs/``).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Architecture hyperparameters for the decoder-only Transformer.

    The defaults match the initial model described in CLAUDE.md:
    a ~40M parameter model with 12 layers and a hidden size of 512.
    """

    vocab_size: int = 32000
    num_layers: int = 12
    hidden_size: int = 512
    num_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 1536
    context_length: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        """Validate that the head geometry is consistent."""
        if self.num_heads * self.head_dim != self.hidden_size:
            raise ValueError(
                f"num_heads * head_dim ({self.num_heads} * {self.head_dim}) "
                f"must equal hidden_size ({self.hidden_size})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be even for RoPE, "
                "because RoPE rotates pairs of dimensions"
            )


@dataclass
class TrainConfig:
    """Training loop hyperparameters."""

    # Data
    train_data_path: str = "data/train.bin"
    val_data_path: str = "data/val.bin"
    sequence_length: int = 2048

    # Optimization
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 10000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0

    # Precision: "fp32", "fp16", or "bf16"
    precision: str = "bf16"
    compile_model: bool = False

    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    eval_steps: int = 50
    checkpoint_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate precision choice early, before any training starts."""
        valid_precisions = ("fp32", "fp16", "bf16")
        if self.precision not in valid_precisions:
            raise ValueError(
                f"precision must be one of {valid_precisions}, got {self.precision!r}"
            )


@dataclass
class Config:
    """Top-level config bundling model and training settings."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        """Load a config from a JSON file with ``model`` and ``train`` keys."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def to_json(self, path: str | Path) -> None:
        """Save this config as a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    def to_dict(self) -> dict:
        """Return the config as a plain nested dict (for checkpoints)."""
        return asdict(self)
