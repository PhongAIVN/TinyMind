"""LoRA: fine-tune model SFT trên thế giới đồ chơi, chỉ học ma trận hạng thấp A, B.

Xong train: gộp ΔW vào W và lưu <out_dir>/lora_merged.pt (model thường, generate dùng được ngay).
"""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from core.config import ModelConfig
from core.lora import DEFAULT_TARGETS, apply_lora, count_trainable, merge_lora
from core.model import CausalLM
from trainer.checkpoint import load_weights
from trainer.sft_dataset import SFTDataset
from trainer.train_config import TrainConfig
from trainer.train_pretrain import run_training
from trainer.train_sft import check_tokenizer_matches

CHECKPOINT_NAME = "lora.pt"
MERGED_NAME = "lora_merged.pt"


def save_merged(model_config: ModelConfig, lora_checkpoint: Path, out_path: Path, lora_args: dict) -> None:
    """Dựng lại model + LoRA, nạp checkpoint, gộp, lưu dạng mà inference.load_model đọc được."""
    model = CausalLM(model_config)
    apply_lora(model, **lora_args)
    load_weights(lora_checkpoint, model)
    merge_lora(model)
    payload = {
        "model": {key: value.detach().cpu().float() for key, value in model.state_dict().items()},
        "model_config": model_config.to_dict(),
        "lora": lora_args,
    }
    torch.save(payload, out_path)


def main() -> None:
    """python -m trainer.train_lora --model-config configs/model_50m.json --train-config configs/lora_50m.json"""
    parser = argparse.ArgumentParser(description="LoRA trên thế giới đồ chơi")
    parser.add_argument("--model-config", default="configs/model_50m.json")
    parser.add_argument("--train-config", default="configs/lora_50m.json")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS), help="tên lớp Linear cần gắn LoRA")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model_config = ModelConfig.from_dict(json.loads(Path(args.model_config).read_text(encoding="utf-8")))
    train_config = TrainConfig.from_json(args.train_config)
    if not args.resume and not train_config.init_from:
        raise ValueError("LoRA cần init_from (model SFT để gắn LoRA lên)")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    check_tokenizer_matches(tokenizer, model_config, args.tokenizer)

    lora_args = {"r": args.rank, "alpha": args.alpha, "targets": tuple(args.targets), "dropout": args.dropout}

    def attach_lora(model):
        names = apply_lora(model, **lora_args)
        trainable, total = count_trainable(model)
        print(f"LoRA r={args.rank} alpha={args.alpha} trên {len(names)} lớp | học {trainable:,}/{total:,} tham số ({trainable / total:.2%})")
        return model

    train_dataset = SFTDataset(train_config.train_bin, tokenizer, train_config.seq_len)
    eval_dataset = SFTDataset(train_config.eval_bin, tokenizer, train_config.seq_len)
    stats = run_training(
        model_config, train_config, train_dataset, eval_dataset, CHECKPOINT_NAME, args.resume, prepare_model=attach_lora
    )
    print(f"Xong: {stats}")

    out_dir = Path(train_config.out_dir)
    save_merged(model_config, out_dir / CHECKPOINT_NAME, out_dir / MERGED_NAME, lora_args)
    print(f"Đã gộp LoRA -> {out_dir / MERGED_NAME}")


if __name__ == "__main__":
    main()
