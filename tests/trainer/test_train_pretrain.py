import math

import numpy as np
import pytest
import torch

from core.config import ModelConfig
from trainer.pretrain_dataset import DTYPE
from trainer.train_pretrain import CHECKPOINT_NAME, train

PATTERN = [5, 9, 13, 17, 21, 25, 29, 33]


def test_train_runs_and_saves_checkpoint(tiny_model_config, make_train_config, tmp_path):
    config = make_train_config(max_steps=3)

    stats = train(tiny_model_config, config)

    assert stats["step"] == 3
    assert math.isfinite(stats["train_loss"])
    assert (tmp_path / "out" / CHECKPOINT_NAME).exists()


def test_initial_loss_is_close_to_log_vocab(tiny_model_config, make_train_config):
    """Bước đầu tiên: model chưa học gì nên loss ≈ ln V."""
    stats = train(
        tiny_model_config, make_train_config(max_steps=1, warmup_steps=1, lr=1e-8, min_lr=0.0)
    )

    assert abs(stats["train_loss"] - math.log(tiny_model_config.vocab_size)) < 0.3


def test_resume_continues_step_count(tiny_model_config, make_train_config, tmp_path):
    train(tiny_model_config, make_train_config(max_steps=2))

    stats = train(tiny_model_config, make_train_config(max_steps=5), resume=True)

    assert stats["step"] == 5


def test_resume_without_checkpoint_raises(tiny_model_config, make_train_config):
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        train(tiny_model_config, make_train_config(), resume=True)


def test_evaluation_runs(tiny_model_config, make_train_config):
    stats = train(tiny_model_config, make_train_config(max_steps=2, eval_interval=1))

    assert math.isfinite(stats["eval_loss"])


def test_loss_decreases_on_repeating_data(tmp_path, make_train_config):
    """Dữ liệu lặp lại một mẫu ngắn: model phải học thuộc, loss giảm mạnh."""
    tokens = np.array(PATTERN * 500, dtype=DTYPE)
    bin_path = tmp_path / "repeat.bin"
    tokens.tofile(bin_path)

    model_config = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        ffn_hidden=64,
        max_seq_len=32,
        rope_theta=10000.0,
        norm_eps=1e-6,
        tie_embedding=True,
    )
    common = dict(
        train_bin=str(bin_path),
        eval_bin=str(bin_path),
        seq_len=16,
        batch_size=4,
        accum_steps=1,
        lr=5e-3,
        min_lr=1e-4,
    )
    before = train(model_config, make_train_config(max_steps=1, warmup_steps=1, **common))
    after = train(model_config, make_train_config(max_steps=60, warmup_steps=5, **common))

    assert after["train_loss"] < 0.5 * before["train_loss"]


def test_batches_differ_between_steps(tiny_model_config, make_train_config):
    """Mỗi step phải lấy dữ liệu mới, không lặp lại đúng một batch."""
    config = make_train_config(max_steps=2, accum_steps=1, batch_size=2)
    seen = []

    from trainer import train_pretrain

    original = train_pretrain.next_token_loss

    def spy(logits, targets):
        seen.append(targets.clone())
        return original(logits, targets)

    train_pretrain.next_token_loss = spy
    try:
        train(tiny_model_config, config)
    finally:
        train_pretrain.next_token_loss = original

    assert not torch.equal(seen[0], seen[1])

def test_sdpa_training_matches_manual(tiny_model_config, make_train_config, tmp_path):
    """Đổi cách tính attention không được đổi kết quả train (cùng seed, fp32)."""
    manual = train(tiny_model_config, make_train_config(max_steps=3, eval_interval=3, out_dir=str(tmp_path / "a")))
    sdpa = train(tiny_model_config, make_train_config(max_steps=3, eval_interval=3, attention="sdpa", out_dir=str(tmp_path / "b")))

    assert sdpa["train_loss"] == pytest.approx(manual["train_loss"], rel=1e-4)
    assert sdpa["eval_loss"] == pytest.approx(manual["eval_loss"], rel=1e-4)


def test_unknown_attention_is_rejected(make_train_config):
    with pytest.raises(ValueError, match="attention"):
        make_train_config(attention="flash")
