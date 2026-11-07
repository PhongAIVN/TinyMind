"""Vòng lặp pretrain: ghép dataset, model, optimizer, scheduler, checkpoint."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from core.config import ModelConfig
from core.model import CausalLM
from trainer.checkpoint import load_checkpoint, save_checkpoint
from trainer.losses import next_token_loss
from trainer.lr_scheduler import get_lr
from trainer.metrics import Throughput, format_log, global_grad_norm, max_vram_mb
from trainer.optim import build_optimizer
from trainer.pretrain_dataset import PretrainDataset
from trainer.train_config import TrainConfig

TORCH_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
CHECKPOINT_NAME = "pretrain.pt"

def build_loader(dataset: PretrainDataset, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    """DataLoader có seed cố định, bỏ batch lẻ cuối để mọi batch cùng kích thước."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        generator=generator if shuffle else None,
    )


def endless(loader: DataLoader):
    """Lặp vô hạn qua loader: pretrain đếm theo step, không theo epoch."""
    while True:
        yield from loader


@torch.no_grad()
def evaluate(model: CausalLM, loader: DataLoader, n_batches: int, device: torch.device, autocast) -> float:
    """Val loss trung bình trên n_batches batch đầu tiên."""
    model.eval()
    total, count = 0.0, 0
    for batch_index, (x, y) in enumerate(loader):
        if batch_index >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast():
            total += next_token_loss(model(x), y).item()
        count += 1
    model.train()
    return total / max(count, 1)


def train(model_config: ModelConfig, train_config: TrainConfig, resume: bool = False) -> dict[str, Any]:
    """Chạy pretrain. Trả về thống kê cuối cùng."""
    torch.manual_seed(train_config.seed)
    device = torch.device(train_config.device)
    dtype = TORCH_DTYPES[train_config.dtype]
    use_amp = dtype is not torch.float32 and device.type == "cuda"

    def autocast():
        return torch.autocast(device_type=device.type, dtype=dtype) if use_amp else nullcontext()

    scaler = torch.amp.GradScaler(device.type, enabled=(use_amp and dtype is torch.float16))

    train_loader = build_loader(
        PretrainDataset(train_config.train_bin, train_config.seq_len),
        train_config.batch_size,
        train_config.seed,
        shuffle=True,
    )
    eval_loader = build_loader(
        PretrainDataset(train_config.eval_bin, train_config.seq_len),
        train_config.batch_size,
        train_config.seed,
        shuffle=False,
    )

    model = CausalLM(model_config).to(device)
    optimizer = build_optimizer(model, lr=train_config.lr, weight_decay=train_config.weight_decay)

    checkpoint_path = Path(train_config.out_dir) / CHECKPOINT_NAME
    start_step = 0
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"không có checkpoint để resume: {checkpoint_path}")
        start_step = load_checkpoint(checkpoint_path, model, optimizer, scaler, map_location=device)

    batches = endless(train_loader)
    # Resume: bỏ qua đúng số batch đã dùng, để chạy tiếp trùng với một lần chạy liền mạch.
    for _ in range(start_step * train_config.accum_steps):
        next(batches)

    throughput = Throughput()
    tokens_per_micro_batch = train_config.batch_size * train_config.seq_len
    model.train()

    train_loss, eval_loss, step = float("nan"), float("nan"), start_step
    for step in range(start_step, train_config.max_steps):
        lr = get_lr(
            step,
            max_lr=train_config.lr,
            min_lr=train_config.min_lr,
            warmup_steps=train_config.warmup_steps,
            max_steps=train_config.max_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(train_config.accum_steps):
            x, y = next(batches)
            x, y = x.to(device), y.to(device)
            with autocast():
                loss = next_token_loss(model(x), y) / train_config.accum_steps
            scaler.scale(loss).backward()
            train_loss += loss.item()
            throughput.update(tokens_per_micro_batch)

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip).item()
        scaler.step(optimizer)
        scaler.update()

        if (step + 1) % train_config.log_interval == 0:
            print(
                format_log(
                    step + 1,
                    train_config.max_steps,
                    train_loss,
                    lr,
                    grad_norm,
                    throughput.tokens_per_second(),
                    max_vram_mb(),
                )
            )
            throughput.reset()

        if (step + 1) % train_config.eval_interval == 0:
            eval_loss = evaluate(model, eval_loader, train_config.eval_batches, device, autocast)
            print(f"step {step + 1:>6} | val loss {eval_loss:6.3f}")

        if (step + 1) % train_config.save_interval == 0:
            save_checkpoint(
                checkpoint_path, model, optimizer, step + 1, model_config, train_config, scaler
            )

    save_checkpoint(checkpoint_path, model, optimizer, step + 1, model_config, train_config, scaler)
    return {"step": step + 1, "train_loss": train_loss, "eval_loss": eval_loss}


def main() -> None:
    """Chạy từ dòng lệnh: python -m trainer.train_pretrain --model-config ... --train-config ..."""
    parser = argparse.ArgumentParser(description="Pretrain một SLM tiếng Việt")
    parser.add_argument("--model-config", default="configs/model_tiny.json")
    parser.add_argument("--train-config", default="configs/pretrain_tiny.json")
    parser.add_argument("--resume", action="store_true", help="chạy tiếp từ checkpoint")
    args = parser.parse_args()

    model_config = ModelConfig.from_dict(
        json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    )
    train_config = TrainConfig.from_json(args.train_config)

    stats = train(model_config, train_config, resume=args.resume)
    print(f"Xong: {stats}")


if __name__ == "__main__":
    main()