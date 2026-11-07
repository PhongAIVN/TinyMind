import math

import torch

from trainer.losses import IGNORE_INDEX, next_token_loss

VOCAB = 50
BATCH, SEQ = 2, 6


def test_uniform_logits_give_log_vocab():
    """Logits toàn 0 = đoán đều mọi token → loss = ln V."""
    logits = torch.zeros(BATCH, SEQ, VOCAB)
    targets = torch.randint(0, VOCAB, (BATCH, SEQ))

    loss = next_token_loss(logits, targets)

    assert abs(loss.item() - math.log(VOCAB)) < 1e-5


def test_confident_correct_prediction_gives_near_zero_loss():
    targets = torch.randint(0, VOCAB, (BATCH, SEQ))
    logits = torch.zeros(BATCH, SEQ, VOCAB)
    logits.scatter_(-1, targets.unsqueeze(-1), 50.0)

    assert next_token_loss(logits, targets).item() < 1e-5


def test_ignored_positions_do_not_count():
    """Vị trí có nhãn -100 bị bỏ qua, dùng cho SFT sau này."""
    logits = torch.zeros(BATCH, SEQ, VOCAB)
    targets = torch.randint(0, VOCAB, (BATCH, SEQ))
    logits.scatter_(-1, targets.unsqueeze(-1), 50.0)

    targets_with_ignore = targets.clone()
    targets_with_ignore[:, 0] = IGNORE_INDEX
    logits[:, 0] = 0.0  # vị trí này đoán rất tệ nhưng phải bị bỏ qua

    assert next_token_loss(logits, targets_with_ignore).item() < 1e-5


def test_fp16_logits_do_not_overflow():
    """Logits fp16 lớn: cross-entropy phải tính trong fp32 nên vẫn hữu hạn."""
    logits = torch.full((1, 2, VOCAB), 3000.0, dtype=torch.float16)
    targets = torch.zeros(1, 2, dtype=torch.long)

    loss = next_token_loss(logits, targets)

    assert torch.isfinite(loss)


def test_gradient_flows_to_logits():
    logits = torch.zeros(BATCH, SEQ, VOCAB, requires_grad=True)
    targets = torch.randint(0, VOCAB, (BATCH, SEQ))

    next_token_loss(logits, targets).backward()

    assert logits.grad is not None and logits.grad.abs().sum() > 0