"""Tests for core.config.ModelConfig."""

from dataclasses import FrozenInstanceError

import pytest

from core.config import ModelConfig

# Bộ tham số hợp lệ nhỏ. Mỗi test tạo dict MỚI từ đây ({**VALID_KWARGS, ...}),
# không bao giờ sửa trực tiếp, để các test không ảnh hưởng lẫn nhau.
VALID_KWARGS = {
    "vocab_size": 100,
    "d_model": 64,
    "n_layers": 2,
    "n_heads": 4,
    "ffn_hidden": 128,
    "max_seq_len": 32,
    "rope_theta": 10000.0,
    "norm_eps": 1e-6,
    "tie_embedding": True,
}

# Danh sách viết lại ở đây, KHÔNG import từ core.config:
# nếu code quên một trường thì test vẫn bắt được.
INT_FIELDS = ["vocab_size", "d_model", "n_layers", "n_heads", "ffn_hidden", "max_seq_len"]
FLOAT_FIELDS = ["rope_theta", "norm_eps"]


# ---------------------------------------------------------------- đường đúng


def test_valid_config_is_created():
    config = ModelConfig(**VALID_KWARGS)

    assert config.d_model == 64
    assert config.head_dim == 16  # 64 / 4


def test_int_is_accepted_for_float_fields():
    """JSON hay ghi 10000 thay vì 10000.0, nên int phải được chấp nhận."""
    config = ModelConfig(**{**VALID_KWARGS, "rope_theta": 10000})

    assert config.rope_theta == 10000


def test_from_dict_creates_config():
    assert ModelConfig.from_dict(VALID_KWARGS) == ModelConfig(**VALID_KWARGS)


def test_to_dict_returns_all_fields():
    assert ModelConfig(**VALID_KWARGS).to_dict() == VALID_KWARGS


def test_to_dict_round_trip():
    config = ModelConfig(**VALID_KWARGS)

    assert ModelConfig.from_dict(config.to_dict()) == config


# ---------------------------------------------------------------- from_dict


@pytest.mark.parametrize("bad_input", ["abc", None, [("d_model", 64)]])
def test_from_dict_rejects_non_dict(bad_input):
    with pytest.raises(ValueError, match="dict"):
        ModelConfig.from_dict(bad_input)


def test_from_dict_unknown_key_raises():
    bad_dict = {**VALID_KWARGS, "foo": 1}

    with pytest.raises(ValueError, match="foo"):
        ModelConfig.from_dict(bad_dict)


def test_from_dict_missing_key_raises():
    bad_dict = dict(VALID_KWARGS)  # bản sao, để del không làm hỏng VALID_KWARGS
    del bad_dict["n_layers"]

    with pytest.raises(ValueError, match="n_layers"):
        ModelConfig.from_dict(bad_dict)


def test_from_dict_reports_unknown_and_missing_together():
    bad_dict = {**VALID_KWARGS, "foo": 1}
    del bad_dict["n_layers"]

    # "as exc_info" giữ lại lỗi để kiểm tra nhiều điều trên cùng một thông báo.
    with pytest.raises(ValueError) as exc_info:
        ModelConfig.from_dict(bad_dict)

    message = str(exc_info.value)
    assert "foo" in message
    assert "n_layers" in message


def test_from_dict_still_validates_values():
    """from_dict phải đi qua __post_init__, không chỉ kiểm tra tên khóa."""
    bad_dict = {**VALID_KWARGS, "d_model": 0}

    with pytest.raises(ValueError, match="d_model"):
        ModelConfig.from_dict(bad_dict)


# ---------------------------------------------------------------- kiểu và giá trị


@pytest.mark.parametrize("field_name", INT_FIELDS)
@pytest.mark.parametrize("bad_value", [0, -1, True, 2.0, "8"])
def test_invalid_int_field_raises(field_name, bad_value):
    """Mọi trường int phải là int thật (không bool/float/str) và > 0."""
    bad_kwargs = {**VALID_KWARGS, field_name: bad_value}

    with pytest.raises(ValueError, match=field_name):
        ModelConfig(**bad_kwargs)


@pytest.mark.parametrize("field_name", FLOAT_FIELDS)
@pytest.mark.parametrize(
    "bad_value", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, True, "1e4"]
)
def test_invalid_float_field_raises(field_name, bad_value):
    """Mọi trường float phải là số (không bool/str), hữu hạn và > 0."""
    bad_kwargs = {**VALID_KWARGS, field_name: bad_value}

    with pytest.raises(ValueError, match=field_name):
        ModelConfig(**bad_kwargs)


@pytest.mark.parametrize("bad_value", ["false", "true", 0, 1, None])
def test_tie_embedding_must_be_bool(bad_value):
    bad_kwargs = {**VALID_KWARGS, "tie_embedding": bad_value}

    with pytest.raises(ValueError, match="tie_embedding"):
        ModelConfig(**bad_kwargs)


# ---------------------------------------------------------------- ràng buộc kiến trúc


def test_d_model_not_divisible_by_n_heads_raises():
    bad_kwargs = {**VALID_KWARGS, "d_model": 100, "n_heads": 3}

    with pytest.raises(ValueError, match="n_heads"):
        ModelConfig(**bad_kwargs)


def test_odd_head_dim_raises():
    """head_dim = 6 / 2 = 3 là số lẻ, RoPE không xoay theo cặp được."""
    bad_kwargs = {**VALID_KWARGS, "d_model": 6, "n_heads": 2}

    with pytest.raises(ValueError, match="head_dim"):
        ModelConfig(**bad_kwargs)


# ---------------------------------------------------------------- frozen


def test_config_is_frozen():
    config = ModelConfig(**VALID_KWARGS)
    with pytest.raises(FrozenInstanceError):
        config.d_model = 128
