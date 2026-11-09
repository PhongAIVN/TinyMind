"""Loss: đoán token kế tiếp (pretrain, SFT) và DPO."""

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

def sequence_log_probs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Log-xác suất của cả câu trả lời: tổng log p(token) trên các vị trí được học.

    Shape:
        logits: (B, T, V), targets: (B, T) long  ->  (B,)
    Công thức:
        log π(y|x) = Σ_t log p(targets[t]), chỉ cộng các t có targets[t] != IGNORE_INDEX.
    Ổn định số học:
        log_softmax trên float32 (không lấy log của softmax: dễ ra log 0 = -inf).
    """
    log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)
    mask = targets != IGNORE_INDEX
    safe_targets = targets.clamp(min=0)  # IGNORE_INDEX âm không gather được; vị trí đó bị mask = 0
    token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct Preference Optimization.

    Mục đích:
        Đẩy xác suất câu chosen lên, câu rejected xuống, so với model tham chiếu (ref),
        không cần reward model hay RL.
    Shape:
        4 tensor (B,) log-prob cả câu  ->  (loss scalar, reward_chosen (B,), reward_rejected (B,))
    Công thức:
        r = β · (log π(y|x) - log π_ref(y|x))       (reward ngầm)
        L = -log σ(r_chosen - r_rejected)
        policy = ref  ->  r = 0  ->  L = -log σ(0) = ln 2.
    Ổn định số học:
        Dùng logsigmoid thay vì log(sigmoid(.)) để không ra log 0 khi chênh lệch lớn.
    """
    reward_chosen = beta * (policy_chosen - ref_chosen)
    reward_rejected = beta * (policy_rejected - ref_rejected)
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
    return loss, reward_chosen.detach(), reward_rejected.detach()


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Chưng cất: student bắt chước cả phân phối của teacher, không chỉ token đúng.

    Shape:
        student_logits, teacher_logits: (B, T, V), targets: (B, T)  ->  scalar
    Công thức:
        p = softmax(teacher / T), q = softmax(student / T)
        KL = Σ_v p·(log p - log q), trung bình trên các vị trí được học
        L = alpha · T² · KL + (1 - alpha) · CE(student, targets)
        Nhân T² để độ lớn gradient của KL không đổi khi đổi T.
        student = teacher -> KL = 0.
    Ổn định số học:
        log_softmax float32 cho cả hai; KL tính từ log-prob, không lấy log của xác suất.
    """
    mask = targets != IGNORE_INDEX
    teacher_log_probs = F.log_softmax(teacher_logits.to(torch.float32) / temperature, dim=-1)[mask]
    student_log_probs = F.log_softmax(student_logits.to(torch.float32) / temperature, dim=-1)[mask]
    kl = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean()
    return alpha * temperature**2 * kl + (1 - alpha) * next_token_loss(student_logits, targets)
