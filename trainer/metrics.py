"""Đo và in số liệu khi train."""

import time
from collections.abc import Iterable

import torch

BYTES_PER_MB = 1e6

def global_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Chuẩn L2 của toàn bộ gradient, gộp mọi tham số.

    Công thức:
        sqrt( Σ_p ||grad_p||² )
    Lưu ý:
        Grad norm tăng vọt thường báo trước loss sắp nổ. Luôn log giá trị này.
    """
    total = torch.zeros((), dtype=torch.float32)
    for param in parameters:
        if param.grad is not None:
            total += param.grad.detach().float().pow(2).sum().cpu()
    return total.sqrt().item()


def max_vram_mb() -> float:
    """VRAM cao nhất đã dùng (MB). Trả 0 nếu chạy CPU."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / BYTES_PER_MB


class Throughput:
    """Đếm tốc độ xử lý token."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._start = time.perf_counter()
        self._tokens = 0

    def update(self, n_tokens: int) -> None:
        self._tokens += n_tokens

    def tokens_per_second(self) -> float:
        elapsed = time.perf_counter() - self._start
        return self._tokens / elapsed if elapsed > 0 else 0.0


def format_log(
    step: int,
    max_steps: int,
    loss: float,
    lr: float,
    grad_norm: float,
    tokens_per_second: float,
    vram_mb: float,
) -> str:
    """Một dòng log gọn cho mỗi lần in."""
    return (
        f"step {step:>6}/{max_steps} | loss {loss:6.3f} | lr {lr:.2e} | "
        f"gnorm {grad_norm:6.3f} | {tokens_per_second:9,.0f} tok/s | vram {vram_mb:6.0f} MB"
    )