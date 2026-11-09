"""Nền RL dùng chung cho GRPO, CISPO, PPO, Agentic RL.

- Rollout: sinh câu trả lời và GHI log-prob ngay lúc lấy mẫu (old log-prob).
- token_log_probs: tính lại log-prob bằng một lần forward; phải khớp lúc lấy mẫu.
- Reward theo quy tắc (đúng/sai đáp số), KL tới ref, advantage nhóm (GRPO), GAE (PPO).

Căn chỉnh: với chuỗi seq dài L, mọi tensor theo token có shape (B, L - 1);
vị trí t ứng với token seq[t + 1] (token được dự đoán từ seq[: t + 1]).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.agent import final_answer, run_agent
from inference.eval_toy_math import MAX_TOOL_TURNS, prompt_for
from scripts.build_toy_math import parse_answer
from trainer.sft_dataset import ChatEncoder

STD_EPS = 1e-6


@dataclass(frozen=True)
class Episode:
    """Một lần trả lời: ids đầy đủ, cờ token do model sinh, log-prob lúc sinh (theo thứ tự token sinh)."""

    ids: list[int]
    generated: list[bool]
    log_probs: list[float]
    answer: str
    reward: float


@torch.no_grad()
def sample_with_log_probs(
    model: nn.Module, prompt_ids: list[int], n: int, max_new_tokens: int, temperature: float, eos_id: int
) -> tuple[list[list[int]], list[list[float]]]:
    """Lấy mẫu n câu từ cùng prompt (KV cache), trả token và log π(token) lúc lấy mẫu.

    Log-prob tính trên softmax(logits / temperature): đúng phân phối đã lấy mẫu.
    Mỗi câu dừng sau <eos> (eos được giữ lại, vì model cũng học lúc nào dừng).
    """
    model.eval()
    device = next(model.parameters()).device
    max_new_tokens = min(max_new_tokens, model.config.max_seq_len - len(prompt_ids))
    logits, past = model(torch.tensor([prompt_ids] * n, device=device), use_cache=True)
    tokens: list[list[int]] = [[] for _ in range(n)]
    log_probs: list[list[float]] = [[] for _ in range(n)]
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        step_log_probs = F.log_softmax(logits[:, -1].float() / temperature, dim=-1)
        next_token = torch.multinomial(step_log_probs.exp(), num_samples=1)
        chosen = step_log_probs.gather(-1, next_token).squeeze(-1)
        for index in (~finished).nonzero().flatten().tolist():
            tokens[index].append(int(next_token[index]))
            log_probs[index].append(float(chosen[index]))
        finished |= next_token.squeeze(-1) == eos_id
        if bool(finished.all()):
            break
        logits, past = model(next_token, past_kvs=past, use_cache=True)
    return tokens, log_probs


def toy_reward(answer: str, result: int) -> float:
    """Reward theo quy tắc: 1 nếu đáp số (số cuối trong câu trả lời) đúng, 0 nếu sai."""
    return 1.0 if parse_answer(answer) == result else 0.0


def rollout(
    model: nn.Module, encoder: ChatEncoder, question: str, result: int, mode: str,
    n: int, max_new_tokens: int, temperature: float,
) -> list[Episode]:
    """n câu trả lời cho một câu hỏi. mode="tools": được gọi máy tính (agentic), sinh từng câu một."""
    prompt = prompt_for(encoder, question, mode)
    episodes = []
    if mode != "tools":
        samples, sample_log_probs = sample_with_log_probs(model, prompt, n, max_new_tokens, temperature, encoder.im_end_id)
        for tokens, log_probs in zip(samples, sample_log_probs):
            ids, generated = prompt + tokens, [False] * len(prompt) + [True] * len(tokens)
            answer = final_answer(encoder, ids, generated)
            episodes.append(Episode(ids, generated, log_probs, answer, toy_reward(answer, result)))
        return episodes
    for _ in range(n):
        recorded: list[float] = []

        def sample_fn(ids: list[int]) -> list[int]:
            tokens, log_probs = sample_with_log_probs(model, ids, 1, max_new_tokens, temperature, encoder.im_end_id)
            recorded.extend(log_probs[0])
            return tokens[0]

        ids, generated, _ = run_agent(encoder, prompt, sample_fn, MAX_TOOL_TURNS)
        answer = final_answer(encoder, ids, generated)
        episodes.append(Episode(ids, generated, recorded, answer, toy_reward(answer, result)))
    return episodes


def collate_episodes(episodes: list[Episode], pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Đệm thành tensor. mask = 1 ở token model sinh (token tool / prompt / đệm = 0)."""
    length = max(len(episode.ids) for episode in episodes)
    batch = len(episodes)
    sequences = torch.full((batch, length), pad_id, dtype=torch.long)
    mask = torch.zeros(batch, length - 1)
    old_log_probs = torch.zeros(batch, length - 1)
    for row, episode in enumerate(episodes):
        sequences[row, : len(episode.ids)] = torch.tensor(episode.ids)
        positions = [index - 1 for index, flag in enumerate(episode.generated) if flag]
        if len(positions) != len(episode.log_probs):
            raise ValueError("số log-prob khác số token sinh ra")
        mask[row, positions] = 1.0
        old_log_probs[row, positions] = torch.tensor(episode.log_probs)
    rewards = torch.tensor([episode.reward for episode in episodes])
    return {key: value.to(device) for key, value in
            {"sequences": sequences, "mask": mask, "old_log_probs": old_log_probs, "rewards": rewards}.items()}


def token_log_probs(model: nn.Module, sequences: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """log π(seq[t+1] | seq[:t+1]) cho mọi t: (B, L) -> (B, L - 1), một lần forward."""
    logits = model(sequences[:, :-1])
    log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
    return log_probs.gather(-1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def kl_to_ref(log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
    """Ước lượng k3 của KL(π || π_ref) theo từng token: e^(r-p) - (r-p) - 1 >= 0, = 0 khi trùng."""
    delta = ref_log_probs - log_probs
    return delta.exp() - delta - 1


def group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """GRPO: A = (r - mean nhóm) / (std nhóm + eps). Nhóm toàn đúng hoặc toàn sai -> A = 0."""
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    return ((grouped - mean) / (std + STD_EPS)).view(-1)


def policy_loss(
    log_probs: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor, mask: torch.Tensor,
    algo: str, clip_eps: float, clip_eps_high: float,
) -> torch.Tensor:
    """Loss chính sách, trung bình theo token (mask).

    advantages: (B,) cho cả câu (GRPO) hoặc (B, L-1) theo token (PPO).
    grpo / ppo: -min(ρ·A, clip(ρ, 1-ε, 1+ε)·A), ρ = π / π_old.
    cispo:      -sg(min(ρ, 1+ε_high)) · A · log π   — cắt TRỌNG SỐ chứ không cắt gradient,
                nên token hiếm (ρ lớn) vẫn đóng góp gradient thay vì bị bỏ hẳn.
    """
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)
    ratio = (log_probs - old_log_probs).exp()
    if algo in ("grpo", "ppo"):
        per_token = -torch.min(ratio * advantages, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages)
    elif algo == "cispo":
        per_token = -ratio.clamp(max=1 + clip_eps_high).detach() * advantages * log_probs
    else:
        raise ValueError(f"algo={algo!r} phải là grpo, cispo hoặc ppo")
    return masked_mean(per_token, mask)


def gae(
    rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, gamma: float, lam: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation trên các token có mask = 1 (bỏ qua token tool / đệm).

    δ_t = r_t + γ·V_{t+1} - V_t,  A_t = δ_t + γλ·A_{t+1};  V sau token cuối = 0.
    λ = 0 -> A_t = δ_t (TD một bước); λ = 1, γ = 1 -> A_t = Σ_{k>=t} r_k - V_t.
    Trả về (advantages, returns = A + V), cùng shape (B, T), = 0 ngoài mask.
    """
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros(rewards.shape[0], device=rewards.device)
    next_value = torch.zeros_like(carry)
    for t in reversed(range(rewards.shape[1])):
        active = mask[:, t] > 0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        updated = delta + gamma * lam * carry
        carry = torch.where(active, updated, carry)
        next_value = torch.where(active, values[:, t], next_value)
        advantages[:, t] = torch.where(active, updated, torch.zeros_like(updated))
    return advantages, (advantages + values) * mask

