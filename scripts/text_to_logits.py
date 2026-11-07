"""Chạy thử toàn bộ luồng: chữ → token ids → model → logits → loss.

Chạy từ thư mục gốc:
    python -m scripts.text_to_logits
"""

import math

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from core.config import ModelConfig
from core.model import CausalLM

TOKENIZER_PATH = "tokenizer/tokenizer.json"
TEXT = "Hoàng Sa và Trường Sa là của Việt Nam"
TOP_K = 5


def main() -> None:
    torch.manual_seed(0)

    # ---- 1. Chữ → token ids ----
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    ids = tokenizer.encode(TEXT).ids
    print(f"Chữ: {TEXT!r}")
    print(f"Số byte UTF-8: {len(TEXT.encode('utf-8'))}  →  số token: {len(ids)}")
    print(f"Token ids: {ids}")
    print(f"Từng token: {[tokenizer.decode([i]) for i in ids]}")

    # ---- 2. Tạo model (vocab_size PHẢI bằng số token của tokenizer) ----
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=64,
        n_layers=2,
        n_heads=4,
        ffn_hidden=128,
        max_seq_len=64,
        rope_theta=10000.0,
        norm_eps=1e-6,
        tie_embedding=True,
    )
    lm = CausalLM(config)
    lm.eval()  # chế độ suy luận

    # ---- 3. ids → tensor (B=1, T) ----
    input_ids = torch.tensor([ids], dtype=torch.long)
    print(f"\ninput_ids: {tuple(input_ids.shape)}")

    # ---- 4. Forward ----
    with torch.no_grad():  # không cần gradient khi chỉ chạy thử
        hidden = lm.model(input_ids)
        logits = lm.lm_head(hidden)
    print(f"hidden:    {tuple(hidden.shape)}")
    print(f"logits:    {tuple(logits.shape)}")

    # ---- 5. Loss: vị trí t đoán token t+1 ----
    vocab = config.vocab_size
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), input_ids[:, 1:].reshape(-1))
    print(f"\nloss = {loss.item():.3f}   (ln V = {math.log(vocab):.3f})")

    # ---- 6. Model đoán token tiếp theo sau câu ----
    top_ids = logits[0, -1].topk(TOP_K).indices.tolist()
    print(f"{TOP_K} token model đoán tiếp theo (chưa train → ngẫu nhiên): "
          f"{[tokenizer.decode([i]) for i in top_ids]}")


if __name__ == "__main__":
    main()