"""SFT: dạy model đã pretrain trả lời theo khuôn hội thoại.

Dùng lại vòng lặp của pretrain (run_training); chỉ khác dataset và điểm khởi đầu (init_from).
Trong config SFT, train_bin / eval_bin trỏ tới file .jsonl hội thoại.
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from core.config import ModelConfig
from trainer.sft_dataset import SFTDataset
from trainer.train_config import TrainConfig
from trainer.train_pretrain import run_training

CHECKPOINT_NAME = "sft.pt"


def check_tokenizer_matches(tokenizer, model_config: ModelConfig, path: str) -> None:
    """Tokenizer khác vocab với model -> id lệch nhau mà không báo lỗi gì, model học lại từ đầu. Chặn sớm."""
    if tokenizer.get_vocab_size() != model_config.vocab_size:
        raise ValueError(
            f"{path}: vocab {tokenizer.get_vocab_size()} khác model vocab {model_config.vocab_size} "
            "-> sai tokenizer (phải dùng đúng file đã tokenize dữ liệu pretrain)"
        )


def main() -> None:
    """python -m trainer.train_sft --model-config configs/model_50m.json --train-config configs/sft_50m.json"""
    parser = argparse.ArgumentParser(description="SFT trên dữ liệu hội thoại")
    parser.add_argument("--model-config", default="configs/model_50m.json")
    parser.add_argument("--train-config", default="configs/sft_50m.json")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--resume", action="store_true", help="chạy tiếp từ sft.pt")
    args = parser.parse_args()

    model_config = ModelConfig.from_dict(json.loads(Path(args.model_config).read_text(encoding="utf-8")))
    train_config = TrainConfig.from_json(args.train_config)
    if not args.resume and not train_config.init_from:
        raise ValueError("SFT cần init_from (trọng số pretrain), nếu không sẽ train từ ngẫu nhiên")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    check_tokenizer_matches(tokenizer, model_config, args.tokenizer)
    train_dataset = SFTDataset(train_config.train_bin, tokenizer, train_config.seq_len)
    eval_dataset = SFTDataset(train_config.eval_bin, tokenizer, train_config.seq_len)
    examples_per_step = train_config.batch_size * train_config.accum_steps
    epochs = train_config.max_steps * examples_per_step / len(train_dataset)
    print(
        f"SFT: {len(train_dataset):,} hội thoại train (bỏ {train_dataset.dropped:,}), "
        f"{len(eval_dataset):,} eval | {train_config.max_steps} bước = {epochs:.2f} epoch"
    )

    stats = run_training(model_config, train_config, train_dataset, eval_dataset, CHECKPOINT_NAME, args.resume)
    print(f"Xong: {stats}")


if __name__ == "__main__":
    main()
