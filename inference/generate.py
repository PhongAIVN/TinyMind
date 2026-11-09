"""Sinh chữ: lặp lấy logits ở vị trí cuối, chọn token, nối vào chuỗi."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import ModelConfig
from core.model import CausalLM
from trainer.checkpoint import load_weights
from trainer.sft_dataset import ChatEncoder

GREEDY_TEMPERATURE = 0.0

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Chọn token kế tiếp từ logits của vị trí cuối.

    Shape:
        logits: (B, V)  ->  (B, 1) long
    Công thức:
        temperature = 0 : argmax (tham lam)
        temperature > 0 : lấy mẫu từ softmax(logits / temperature)
        top_k = k       : chỉ giữ k token điểm cao nhất
    Ổn định số học:
        Tính bằng float32; chia temperature trước khi softmax.
    """
    if temperature < 0:
        raise ValueError(f"temperature={temperature} phải >= 0")
    if temperature == GREEDY_TEMPERATURE:
        return logits.argmax(dim=-1, keepdim=True)

    scores = logits.float() / temperature
    if top_k is not None:
        if top_k < 1:
            raise ValueError(f"top_k={top_k} phải >= 1")
        k = min(top_k, scores.shape[-1])
        kth_value = scores.topk(k, dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < kth_value, float("-inf"))

    probs = F.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sinh tiếp `max_new_tokens` token sau `input_ids`.

    Shape:
        input_ids: (B, T)  ->  (B, T + n) với n <= max_new_tokens
    Độ phức tạp:
        use_cache=True : O(n · T) — mỗi bước chỉ tính cho 1 token mới.
        use_cache=False: O(n · T²) — mỗi bước tính lại cả chuỗi.
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens={max_new_tokens} phải >= 0")

    model.eval()
    output = input_ids
    step_input = input_ids
    past_kvs = None
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    for _ in range(max_new_tokens):
        if use_cache:
            logits, past_kvs = model(step_input, past_kvs=past_kvs, use_cache=True)
        else:
            logits = model(output)

        next_token = sample_next_token(logits[:, -1], temperature, top_k, generator)
        if eos_id is not None:
            next_token[finished] = eos_id
            finished |= next_token.squeeze(-1) == eos_id

        output = torch.cat([output, next_token], dim=1)
        step_input = next_token

        if eos_id is not None and bool(finished.all()):
            break

    return output


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> CausalLM:
    """Dựng lại model từ checkpoint (kiến trúc lấy từ chính checkpoint)."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CausalLM(ModelConfig.from_dict(payload["model_config"])).to(device)
    load_weights(checkpoint_path, model, map_location=device)
    return model


def main() -> None:
    """python -m inference.generate --prompt "Hà Nội là" """
    parser = argparse.ArgumentParser(description="Cho model viết tiếp một câu")
    parser.add_argument("--checkpoint", default="checkpoints/pretrain.pt")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--prompt", default="Hà Nội là")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true", help="tắt KV cache (chậm hơn)")
    parser.add_argument("--chat", action="store_true", help="bọc prompt theo khuôn hội thoại (model SFT)")
    args = parser.parse_args()

    from tokenizers import Tokenizer

    torch.manual_seed(args.seed)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = load_model(args.checkpoint)
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError(f"tokenizer vocab {tokenizer.get_vocab_size()} khác model vocab {model.config.vocab_size}")

    if args.chat:
        encoder = ChatEncoder(tokenizer)
        prompt_ids, stop_id = encoder.encode_prompt(args.prompt), encoder.im_end_id
    else:
        prompt_ids, stop_id = tokenizer.encode(args.prompt).ids, tokenizer.token_to_id("<|eos|>")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    output = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=stop_id,
        use_cache=not args.no_cache,
    )

    print(f"Prompt : {args.prompt}")
    if args.chat:
        print(f"Trả lời: {tokenizer.decode(output[0, len(prompt_ids):].tolist())}")
    else:
        print(f"Sinh ra: {tokenizer.decode(output[0].tolist())}")


if __name__ == "__main__":
    main()