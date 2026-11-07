"""Tạo optimizer AdamW với 2 nhóm weight decay."""

import torch
import torch.nn as nn

DEFAULT_BETAS = (0.9, 0.95)
DEFAULT_EPS = 1e-8


def split_decay_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Tách tham số thành nhóm có và không có weight decay.

    Quy tắc: tensor từ 2 chiều trở lên (ma trận Linear, bảng embedding) thì decay;
    tensor 1 chiều (weight của RMSNorm, bias) thì không. Tham số đã đóng băng
    (requires_grad=False, ví dụ khi train LoRA) bị bỏ qua.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return decay, no_decay


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = DEFAULT_BETAS,
    eps: float = DEFAULT_EPS,
) -> torch.optim.AdamW:
    """AdamW với weight decay chỉ áp cho ma trận."""
    decay, no_decay = split_decay_params(model)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)