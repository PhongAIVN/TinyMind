"""Next-token prediction loss.

Purpose:
    The training objective: at every position, predict the next token.
    Cross entropy between the model's logits and the shifted targets.

Formula:
    loss = -(1/N) * sum_t log softmax(logits_t)[target_t]

Shapes:
    logits:  (batch, seq_len, vocab_size)
    targets: (batch, seq_len) int64
    output:  scalar

Complexity:
    O(batch * seq_len * vocab_size).

Numerical stability:
    ``F.cross_entropy`` fuses log-softmax and NLL using the log-sum-exp
    trick, so large logits never overflow. Logits are upcast to float32
    first — computing log-softmax over 32000 classes in fp16/bf16 loses
    precision exactly where the loss signal lives.
"""

import torch
import torch.nn.functional as F

# Positions labeled with this id contribute nothing to the loss.
IGNORE_INDEX = -100


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute mean next-token cross entropy.

    Args:
        logits: ``(batch, seq_len, vocab_size)`` raw model outputs.
        targets: ``(batch, seq_len)`` int64 target token ids. Positions
            equal to ``IGNORE_INDEX`` are excluded from the mean.

    Returns:
        Scalar loss tensor.
    """
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, vocab_size),
        targets.view(-1),
        ignore_index=IGNORE_INDEX,
    )
