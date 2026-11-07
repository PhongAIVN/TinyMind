"""Kiểm tra sau khi chuẩn bị dữ liệu: tokenizer đúng chưa, có bao nhiêu token, nên train bao nhiêu bước."""

import argparse
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

from tokenizer.train_tokenizer import SPECIAL_TOKENS
from trainer.train_config import TrainConfig

SAMPLES = {
    "tiếng Việt": "Hà Nội là thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam, nằm ở đồng bằng sông Hồng.",
    "code": 'def fibonacci(n: int) -> list[int]:\n    """Return first n numbers."""\n    result = [0, 1]\n'
    "    for i in range(2, n):\n        result.append(result[i - 1] + result[i - 2])\n    return result[:n]\n",
}


def check_tokenizer(tokenizer: Tokenizer) -> list[str]:
    """Trả về danh sách lỗi (rỗng = ổn). In số byte/token của từng mẫu."""
    errors = [
        f"{token} có id {tokenizer.token_to_id(token)}, cần {expected_id}"
        for expected_id, token in enumerate(SPECIAL_TOKENS)
        if tokenizer.token_to_id(token) != expected_id
    ]
    for name, text in SAMPLES.items():
        ids = tokenizer.encode(text).ids
        if tokenizer.decode(ids) != text:
            errors.append(f"encode -> decode mẫu {name} không ra nguyên văn")
        print(f"{name:12s}: {len(text.encode('utf-8')) / len(ids):.2f} byte/token")
    return errors


def suggest_max_steps(n_tokens: int, config: TrainConfig) -> int:
    """Số bước để đi đúng 1 lượt dữ liệu (khớp cách PretrainDataset cắt cửa sổ)."""
    windows = (n_tokens - 1) // config.seq_len
    return windows // (config.batch_size * config.accum_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="kiểm tra tokenizer + dữ liệu pretrain")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--tokens-dir", default="dataset/pretrain/tokens")
    parser.add_argument("--train-config", default="configs/pretrain_50m.json")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"vocab_size = {tokenizer.get_vocab_size()}")
    errors = check_tokenizer(tokenizer)

    tokens_dir = Path(args.tokens_dir)
    meta = {split: json.loads((tokens_dir / f"{split}_meta.json").read_text(encoding="utf-8")) for split in ("train", "eval")}
    for split, info in meta.items():
        print(f"{split}: {info['n_tokens']:,} token")
        if info["vocab_size"] != tokenizer.get_vocab_size():
            errors.append(f"{split}.bin được tạo bằng tokenizer vocab {info['vocab_size']}, khác tokenizer hiện tại")

    config = TrainConfig.from_json(args.train_config)
    print(f"max_steps cho 1 lượt dữ liệu ({config.tokens_per_step:,} token/bước): {suggest_max_steps(meta['train']['n_tokens'], config)}")

    if errors:
        print("LỖI:\n  " + "\n  ".join(errors))
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
