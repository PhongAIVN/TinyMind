"""Sampling strategies for next-token selection.

Given the logits for the next token, these functions transform the
distribution and pick one token id. All operate on a single step:

    logits shape: (batch, vocab_size) → chosen ids shape: (batch, 1)

Strategies (applied in this order in ``sample_next_token``):
    1. Repetition penalty — discourage tokens already generated.
    2. Temperature — flatten (>1) or sharpen (<1) the distribution.
    3. Top-k — keep only the k highest-probability tokens.
    4. Top-p (nucleus) — keep the smallest set whose mass >= p.
    5. Sample from what remains (or argmax if temperature == 0).

Numerical stability:
    All filtering happens in logit space using -inf for removed tokens;
    softmax afterwards renormalizes exactly.
"""

import torch


def apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Penalize tokens that already appeared in the sequence.

    Following the CTRL paper: positive logits are divided by ``penalty``,
    negative logits multiplied — both push scores down asymmetrically so
    already-seen tokens become less likely, whatever their sign.

    Args:
        logits: ``(batch, vocab_size)``.
        generated_ids: ``(batch, seq_len_so_far)`` previously seen ids.
        penalty: 1.0 = no penalty; typical values 1.05–1.3.

    Returns:
        New logits tensor of the same shape.
    """
    if penalty == 1.0:
        return logits

    logits = logits.clone()
    seen_scores = torch.gather(logits, dim=1, index=generated_ids)
    penalized = torch.where(seen_scores > 0, seen_scores / penalty, seen_scores * penalty)
    logits.scatter_(dim=1, index=generated_ids, src=penalized)
    return logits


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by 1/temperature (softmax sharpens as T → 0)."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Mask all but the k highest logits to -inf.

    Args:
        logits: ``(batch, vocab_size)``.
        k: Number of candidates to keep; 0 disables the filter.
    """
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth_value = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth_value, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest prefix of tokens (sorted by
    probability) whose cumulative mass is >= p; mask the rest to -inf.

    Args:
        logits: ``(batch, vocab_size)``.
        p: Cumulative probability threshold in (0, 1]; 1.0 disables.
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    # Remove tokens once the mass *before* them already reaches p —
    # shifting right by one guarantees the top token always survives.
    remove_sorted = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= p

    remove_mask = torch.zeros_like(remove_sorted).scatter_(
        dim=-1, index=sorted_indices, src=remove_sorted
    )
    return logits.masked_fill(remove_mask, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    generated_ids: torch.Tensor | None = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """Pick the next token id for each batch row.

    Args:
        logits: ``(batch, vocab_size)`` next-token logits.
        generated_ids: ``(batch, seq_len)`` context ids for the
            repetition penalty (ignored when penalty is 1.0).
        temperature: 0 = greedy argmax; otherwise softmax temperature.
        top_k: Keep only the top-k logits (0 = off).
        top_p: Nucleus threshold (1.0 = off).
        repetition_penalty: 1.0 = off.

    Returns:
        ``(batch, 1)`` int64 tensor of chosen token ids.
    """
    if repetition_penalty != 1.0 and generated_ids is not None:
        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # Greedy decoding: temperature 0 means "always take the best token".
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)

    probabilities = torch.softmax(logits.to(torch.float32), dim=-1)
    return torch.multinomial(probabilities, num_samples=1)
