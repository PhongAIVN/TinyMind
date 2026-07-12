"""Autoregressive text generation with KV cache and streaming.

Algorithm:
    1. Prefill: run the whole prompt through the model once, filling the
       KV cache for every layer. Cost O(T_prompt^2) but done once.
    2. Decode loop: feed only the last generated token; attention reads
       the cached K/V, so each step costs O(T_so_far) instead of
       O(T_so_far^2). Sample, append, repeat.
    3. Stop at ``max_new_tokens`` or when every sequence emits <eos>.

``generate_stream`` yields token ids one at a time (for live printing);
``generate`` collects them into a single tensor.

Usage:
    python -m slm.inference.generate \\
        --checkpoint checkpoints/final.pt \\
        --tokenizer tokenizer/slm.model \\
        --prompt "Xin chào" --max-new-tokens 100
"""

import argparse
from collections.abc import Iterator

import torch

from slm.inference.sampling import sample_next_token
from slm.model.cache import KVCache
from slm.model.transformer import Transformer
from slm.utils.config import ModelConfig


@torch.no_grad()
def generate_stream(
    model: Transformer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    eos_id: int | None = None,
) -> Iterator[torch.Tensor]:
    """Yield one generated token id per step, using a KV cache.

    Args:
        model: The Transformer in eval mode.
        prompt_ids: ``(batch, prompt_len)`` int64 prompt tokens.
        max_new_tokens: Maximum number of tokens to generate.
        temperature / top_k / top_p / repetition_penalty: See sampling.py.
        eos_id: If set, stop once every batch row has emitted it.

    Yields:
        ``(batch, 1)`` int64 tensors, one per generated position.
    """
    model.eval()
    device = prompt_ids.device
    batch_size, prompt_length = prompt_ids.shape
    config = model.config

    max_total = min(config.context_length, prompt_length + max_new_tokens)
    kv_cache = KVCache(
        num_layers=config.num_layers,
        batch_size=batch_size,
        num_heads=config.num_heads,
        max_seq_len=max_total,
        head_dim=config.head_dim,
        device=device,
        dtype=next(model.parameters()).dtype,
    )

    # Prefill: one pass over the whole prompt. Only the last position's
    # logits matter for choosing the first new token.
    logits = model(prompt_ids, kv_cache=kv_cache, position_offset=0)
    generated = prompt_ids
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for step_index in range(max_new_tokens):
        if generated.shape[1] >= max_total:
            break

        next_token = sample_next_token(
            logits[:, -1, :],
            generated_ids=generated,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        if eos_id is not None:
            # Rows that already finished keep emitting eos (ignored by caller).
            next_token = torch.where(
                finished.unsqueeze(1), torch.full_like(next_token, eos_id), next_token
            )
            finished |= next_token.squeeze(1) == eos_id

        yield next_token
        generated = torch.cat([generated, next_token], dim=1)

        if eos_id is not None and bool(finished.all()):
            break

        # Decode step: feed only the newest token; the cache supplies
        # K/V for everything before it.
        logits = model(
            next_token, kv_cache=kv_cache, position_offset=generated.shape[1] - 1
        )


@torch.no_grad()
def generate(
    model: Transformer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    **sampling_kwargs,
) -> torch.Tensor:
    """Generate tokens and return prompt + completion as one tensor.

    Args:
        model: The Transformer.
        prompt_ids: ``(batch, prompt_len)`` int64 prompt tokens.
        max_new_tokens: Maximum tokens to generate.
        **sampling_kwargs: Forwarded to :func:`generate_stream`.

    Returns:
        ``(batch, prompt_len + generated_len)`` int64 tensor.
    """
    pieces = [prompt_ids]
    for next_token in generate_stream(model, prompt_ids, max_new_tokens, **sampling_kwargs):
        pieces.append(next_token)
    return torch.cat(pieces, dim=1)


def main() -> None:
    """CLI: load a checkpoint and stream a completion to stdout."""
    from slm.tokenizer.tokenizer import Tokenizer  # local import: CLI-only dependency

    parser = argparse.ArgumentParser(description="Generate text with a trained SLM")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = ModelConfig(**checkpoint["config"]["model"])
    model = Transformer(model_config).to(device)
    model.load_state_dict(checkpoint["model"])

    tokenizer = Tokenizer(args.tokenizer)
    prompt_ids = torch.tensor(
        [tokenizer.encode(args.prompt, add_bos=True)], dtype=torch.long, device=device
    )

    print(args.prompt, end="", flush=True)
    generated_ids: list[int] = []
    for next_token in generate_stream(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_id=tokenizer.eos_id,
    ):
        token_id = int(next_token[0, 0])
        if token_id == tokenizer.eos_id:
            break
        generated_ids.append(token_id)
        # Decode the full tail each time so multi-token characters
        # (byte fallback) print correctly.
        print("\r" + args.prompt + tokenizer.decode(generated_ids), end="", flush=True)
    print()


if __name__ == "__main__":
    main()
