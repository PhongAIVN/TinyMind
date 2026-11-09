"""Chưng cất: teacher lớn (đóng băng) dạy student nhỏ, cùng tokenizer.

Kiến trúc student lấy từ --model-config (vd configs/model_student.json, 13,3M tham số);
kiến trúc teacher đọc từ chính checkpoint teacher. Student khởi tạo ngẫu nhiên (hoặc init_from).
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from core.attention import enable_sdpa
from core.config import ModelConfig
from inference.generate import load_model
from trainer.losses import distillation_loss
from trainer.sft_dataset import SFTDataset
from trainer.train_config import TrainConfig
from trainer.train_pretrain import LossFn, run_training
from trainer.train_sft import check_tokenizer_matches

CHECKPOINT_NAME = "student.pt"


def make_distill_loss(teacher: nn.Module, temperature: float, alpha: float) -> LossFn:
    def compute_loss(model: nn.Module, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        x, y = batch
        with torch.no_grad():
            teacher_logits = teacher(x)
        return distillation_loss(model(x), teacher_logits, y, temperature, alpha)

    return compute_loss


def load_teacher(path: str | Path, device: torch.device, use_sdpa: bool) -> nn.Module:
    teacher = enable_sdpa(load_model(path), use_sdpa).to(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def main() -> None:
    """python -m trainer.train_distill --teacher checkpoints/50m/think.pt --train-config configs/distill_50m.json"""
    parser = argparse.ArgumentParser(description="chưng cất teacher -> student")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--model-config", default="configs/model_student.json", help="kiến trúc student")
    parser.add_argument("--train-config", default="configs/distill_50m.json")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="trọng số KL (1 - alpha cho CE)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    student_config = ModelConfig.from_dict(json.loads(Path(args.model_config).read_text(encoding="utf-8")))
    train_config = TrainConfig.from_json(args.train_config)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    check_tokenizer_matches(tokenizer, student_config, args.tokenizer)
    teacher = load_teacher(args.teacher, torch.device(train_config.device), train_config.attention == "sdpa")
    if teacher.config.vocab_size != student_config.vocab_size:
        raise ValueError("teacher và student phải cùng vocab (cùng tokenizer)")

    train_dataset = SFTDataset(train_config.train_bin, tokenizer, train_config.seq_len)
    eval_dataset = SFTDataset(train_config.eval_bin, tokenizer, train_config.seq_len)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"Chưng cất: teacher {n_teacher:,} tham số -> student {args.model_config} | T={args.temperature} alpha={args.alpha}")
    stats = run_training(
        student_config, train_config, train_dataset, eval_dataset, CHECKPOINT_NAME, args.resume,
        compute_loss=make_distill_loss(teacher, args.temperature, args.alpha),
    )
    print(f"Xong: {stats}")


if __name__ == "__main__":
    main()
