"""LoRA: học phần cập nhật hạng thấp ΔW = (alpha / r) · B·A, giữ nguyên trọng số gốc W."""

import math

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("q_proj", "v_proj")


class LoRALinear(nn.Module):
    """Bọc một nn.Linear đã có: y = x·Wᵀ + (alpha / r) · dropout(x)·Aᵀ·Bᵀ.

    Mục đích:
        Fine-tune với rất ít tham số: chỉ A (r × in) và B (out × r) được học, W đóng băng.
    Shape:
        x: (..., in)  ->  (..., out)
    Công thức:
        ΔW = scaling · B·A, scaling = alpha / r.
        B khởi tạo 0 -> ΔW = 0 -> lúc đầu model y hệt model gốc.
    Độ phức tạp:
        Thêm O(r · (in + out)) mỗi token; tham số học được r·(in + out) thay vì in·out.
    Ổn định số học:
        Chia cho r giữ độ lớn ΔW ổn định khi đổi r (không phải chỉnh lại learning rate).
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"r={r} phải > 0")
        if alpha <= 0:
            raise ValueError(f"alpha={alpha} phải > 0")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        weight = base.weight
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=weight.dtype, device=weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=weight.dtype, device=weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # như nn.Linear; B = 0 nên ΔW vẫn = 0
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return self.base(x) + self.scaling * update

    def delta_weight(self) -> torch.Tensor:
        """ΔW = scaling · B·A, shape (out, in) như base.weight."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def merged(self) -> nn.Linear:
        """nn.Linear mới với W + ΔW: sinh chữ không tốn thêm phép tính nào."""
        linear = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        linear = linear.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        with torch.no_grad():
            linear.weight.copy_(self.base.weight + self.delta_weight())
            if self.base.bias is not None:
                linear.bias.copy_(self.base.bias)
        return linear


def _replace_child(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, new_module)


def apply_lora(
    model: nn.Module,
    r: int,
    alpha: float,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    dropout: float = 0.0,
) -> list[str]:
    """Đóng băng cả model, bọc LoRA vào các nn.Linear có tên kết thúc bằng một phần tử của targets.

    Ví dụ targets=("q_proj", "v_proj") bắt "model.layers.3.attn.q_proj".
    Trả về danh sách tên lớp đã bọc; không bắt được lớp nào thì báo lỗi (tránh gõ sai tên mà không biết).
    """
    for param in model.parameters():
        param.requires_grad_(False)
    names = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets
    ]
    if not names:
        raise ValueError(f"không có nn.Linear nào tên thuộc {targets}")
    for name in names:
        _replace_child(model, name, LoRALinear(model.get_submodule(name), r, alpha, dropout))
    return names


def merge_lora(model: nn.Module) -> nn.Module:
    """Thay mọi LoRALinear bằng nn.Linear đã gộp. state_dict sau đó giống hệt model gốc."""
    names = [name for name, module in model.named_modules() if isinstance(module, LoRALinear)]
    for name in names:
        _replace_child(model, name, model.get_submodule(name).merged())
    return model


def count_trainable(model: nn.Module) -> tuple[int, int]:
    """(số tham số được học, tổng số tham số) — tham số dùng chung chỉ đếm 1 lần."""
    params = list(model.parameters())  # parameters() đã bỏ trùng tham số tied
    return sum(p.numel() for p in params if p.requires_grad), sum(p.numel() for p in params)
