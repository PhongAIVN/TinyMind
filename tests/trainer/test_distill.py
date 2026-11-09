"""Chưng cất: KL = 0 khi student = teacher; vị trí không học bị bỏ qua; train làm KL giảm."""

import pytest
import torch
import torch.nn.functional as F

from core.model import CausalLM
from trainer.losses import IGNORE_INDEX, distillation_loss, next_token_loss
from trainer.pretrain_dataset import PretrainDataset
from trainer.train_distill import make_distill_loss
from trainer.train_pretrain import run_training


def test_kl_is_zero_when_student_equals_teacher():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    targets = torch.randint(0, 11, (2, 5))
    loss = distillation_loss(logits, logits.clone(), targets, temperature=2.0, alpha=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_alpha_zero_is_plain_cross_entropy():
    torch.manual_seed(1)
    student, teacher = torch.randn(2, 5, 11), torch.randn(2, 5, 11)
    targets = torch.randint(0, 11, (2, 5))
    torch.testing.assert_close(distillation_loss(student, teacher, targets, alpha=0.0), next_token_loss(student, targets))


def test_kl_matches_manual_and_ignores_unlearned_positions():
    torch.manual_seed(2)
    student, teacher = torch.randn(1, 3, 7), torch.randn(1, 3, 7)
    targets = torch.tensor([[IGNORE_INDEX, 4, IGNORE_INDEX]])
    t = 3.0
    p = F.softmax(teacher[0, 1] / t, -1)
    manual = (p * (p.log() - F.log_softmax(student[0, 1] / t, -1))).sum() * t**2
    torch.testing.assert_close(distillation_loss(student, teacher, targets, temperature=t, alpha=1.0), manual)


def test_distillation_training_brings_student_closer(tiny_model_config, make_train_config, token_bins, tmp_path):
    torch.manual_seed(0)
    teacher = CausalLM(tiny_model_config).eval()
    loss_fn = make_distill_loss(teacher, temperature=1.0, alpha=1.0)
    datasets = [PretrainDataset(token_bins[name], 16) for name in ("train", "eval")]
    config = make_train_config(out_dir=str(tmp_path), max_steps=30, lr=1e-2, warmup_steps=0, min_lr=1e-2, eval_interval=1000)

    torch.manual_seed(5)
    batch = tuple(t.unsqueeze(0) for t in datasets[1][0])
    stats = run_training(tiny_model_config, config, *datasets, "student.pt", compute_loss=loss_fn, prepare_model=None)

    student = CausalLM(tiny_model_config)
    student.load_state_dict(torch.load(tmp_path / "student.pt", weights_only=False)["model"])
    fresh = CausalLM(tiny_model_config)
    with torch.no_grad():
        assert loss_fn(student, batch) < loss_fn(fresh, batch)
    assert stats["step"] == 30
