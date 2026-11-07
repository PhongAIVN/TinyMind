import time

import torch

from trainer.metrics import Throughput, format_log, global_grad_norm, max_vram_mb


def test_grad_norm_matches_manual_computation():
    first = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    second = torch.nn.Parameter(torch.tensor([[12.0]]))
    (first.sum() + second.sum()).backward()  # grad toàn 1 → norm = sqrt(3)

    assert global_grad_norm([first, second]) == torch.tensor(3.0).sqrt().item()


def test_grad_norm_ignores_parameters_without_grad():
    with_grad = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    without_grad = torch.nn.Parameter(torch.tensor([10.0]))
    (with_grad * torch.tensor([1.0, 2.0])).sum().backward()  # grad = [1, 2]

    expected = torch.tensor([1.0, 2.0]).norm().item()

    assert global_grad_norm([with_grad, without_grad]) == expected
    assert without_grad.grad is None


def test_throughput_counts_tokens():
    meter = Throughput()
    meter.update(1000)
    time.sleep(0.01)

    assert meter.tokens_per_second() > 0


def test_throughput_reset_clears_counter():
    meter = Throughput()
    meter.update(1000)
    meter.reset()
    time.sleep(0.01)

    assert meter.tokens_per_second() == 0.0


def test_max_vram_is_zero_on_cpu():
    if not torch.cuda.is_available():
        assert max_vram_mb() == 0.0


def test_format_log_contains_every_number():
    line = format_log(
        step=7, max_steps=100, loss=1.234, lr=0.001, grad_norm=0.5, tokens_per_second=1234, vram_mb=42
    )

    for text in ("7", "100", "1.234", "loss", "lr", "gnorm", "tok/s", "vram"):
        assert text in line