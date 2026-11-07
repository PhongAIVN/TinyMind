"""Loss cho pretrain: đoán token kế tiếp."""

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy trung bình trên mọi vị trí.

    Mục đích:
        Đo model đoán token kế tiếp tốt đến đâu.
    Shape:
        logits: (B, T, V), targets: (B, T) long  ->  scalar
    Công thức:
        L = -(1/N) Σ log p(targets[b, t]); model chưa train: L ≈ ln V.
    Độ phức tạp:
        O(B·T·V)
    Ổn định số học:
        Ép logits sang float32 vì log-softmax trên V lớn dễ tràn ở fp16.
    """
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, vocab_size),
        targets.view(-1),
        ignore_index=IGNORE_INDEX,
    )