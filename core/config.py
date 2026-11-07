import math
from dataclasses import asdict, dataclass, fields
from typing import Any

INT_FIELDS = ("vocab_size", "d_model", "n_layers", "n_heads", "ffn_hidden", "max_seq_len")
FLOAT_FIELDS = ("rope_theta", "norm_eps")
BOOL_FIELDS = ("tie_embedding",)


@dataclass(frozen=True)  # frozen: sau khi tạo xong thì không gán lại được thuộc tính
class ModelConfig:
    """Configuration for a Transformer language model.

    Attributes:
        vocab_size: Number of tokens in the vocabulary.
        d_model: Hidden/model dimension.
        n_layers: Number of Transformer layers.
        n_heads: Number of attention heads.
        ffn_hidden: Hidden dimension of the feed-forward network.
        max_seq_len: Maximum sequence length in tokens.
        rope_theta: Base frequency used by Rotary Positional Embeddings (RoPE).
        norm_eps: Small epsilon value used for numerical stability in normalization.
        tie_embedding: Whether to share the input embedding and output projection
            weights.
    """

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    ffn_hidden: int
    max_seq_len: int
    rope_theta: float
    norm_eps: float
    tie_embedding: bool

    def __post_init__(self) -> None:
        """Validate all fields right after the dataclass is created.

        Raises:
            ValueError: If any field has a wrong type or an invalid value.
        """
        # Kiểm tra int trước: phép % trong _validate_head_dim cần n_heads > 0.
        self._validate_int_fields()
        self._validate_head_dim()
        self._validate_float_fields()
        self._validate_bool_fields()

    def _validate_int_fields(self) -> None:
        """Each int field must be a real int (not bool) and > 0."""
        for name in INT_FIELDS:
            value = getattr(self, name)

            # bool là subclass của int (isinstance(True, int) == True), nên phải chặn riêng.
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name}={value!r} phải là int")

            if value <= 0:
                raise ValueError(f"{name}={value} phải > 0")

    def _validate_head_dim(self) -> None:
        """d_model must split evenly into heads, and head_dim must be even for RoPE."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} không chia hết cho n_heads={self.n_heads}"
            )

        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim={self.head_dim} phải là số chẵn vì RoPE xoay từng cặp chiều"
            )

    def _validate_float_fields(self) -> None:
        """Each float field must be a number (int or float, not bool), finite and > 0."""
        for name in FLOAT_FIELDS:
            value = getattr(self, name)

            # Cùng lý do như int: True/False cũng là số, nên phải chặn bool riêng.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name}={value!r} phải là số thực (int hoặc float)")

            # isfinite chặn nan, inf, -inf. Đặt trước "<= 0" vì nan <= 0 trả về False.
            if not math.isfinite(value):
                raise ValueError(f"{name}={value} phải là số hữu hạn")

            if value <= 0:
                raise ValueError(f"{name}={value} phải > 0")

    def _validate_bool_fields(self) -> None:
        """Each bool field must be exactly True or False (not 0/1, not "false")."""
        for name in BOOL_FIELDS:
            value = getattr(self, name)

            if not isinstance(value, bool):
                raise ValueError(f"{name}={value!r} phải là bool")

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.n_heads

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        """Create a ModelConfig from a dictionary.

        Args:
            d: Dictionary containing exactly the ModelConfig fields.

        Returns:
            A validated ModelConfig instance.

        Raises:
            ValueError: If `d` is not a dict, has unknown or missing keys,
                or any value is invalid.
        """
        if not isinstance(d, dict):
            raise ValueError(f"from_dict cần dict, nhận được {type(d).__name__}")

        valid_names = {field.name for field in fields(cls)}
        provided_names = set(d)

        unknown_names = provided_names - valid_names
        missing_names = valid_names - provided_names

        errors = []
        if unknown_names:
            errors.append(f"khóa không hợp lệ: {sorted(unknown_names)}")
        if missing_names:
            errors.append(f"khóa bị thiếu: {sorted(missing_names)}")
        if errors:
            raise ValueError("; ".join(errors))

        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        """Convert the ModelConfig to a dictionary (e.g. to save as JSON).

        Returns:
            A dictionary mapping field names to values.
        """
        return asdict(self)
