"""Training loop for the SLM.

Features:
    - Gradient accumulation (large effective batch on small GPUs)
    - Mixed precision (fp32 / fp16 with loss scaling / bf16)
    - Gradient clipping
    - Warmup + cosine LR schedule
    - Periodic validation and checkpointing with full resume
    - Logging: loss, LR, grad norm, tokens/sec, GPU memory

Usage:
    python -m slm.train.trainer --config slm/configs/slm_base.json
    python -m slm.train.trainer --config ... --resume checkpoints/latest.pt
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from slm.dataset.dataset import PackedTokenDataset
from slm.model.transformer import Transformer
from slm.train.loss import cross_entropy_loss
from slm.train.optimizer import build_optimizer
from slm.train.scheduler import build_scheduler
from slm.utils.checkpoint import load_checkpoint, save_checkpoint
from slm.utils.config import Config
from slm.utils.seed import set_seed

PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class Trainer:
    """Owns the model, data, optimizer, and the step loop."""

    def __init__(self, config: Config, device: str | None = None) -> None:
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(config.train.seed)

        self.model = Transformer(config.model).to(self.device)
        if config.train.compile_model:
            self.model = torch.compile(self.model)

        self.optimizer = build_optimizer(self.model, config.train)
        self.scheduler = build_scheduler(self.optimizer, config.train)

        self.amp_dtype = PRECISION_DTYPES[config.train.precision]
        self.use_amp = self.amp_dtype != torch.float32 and self.device == "cuda"
        # fp16 gradients underflow without loss scaling; bf16 does not need it.
        self.grad_scaler = torch.amp.GradScaler(
            self.device, enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        self.step = 0
        self.epoch = 0

        num_params = self.model.num_parameters()
        print(f"Model parameters: {num_params / 1e6:.1f}M | device: {self.device} "
              f"| precision: {config.train.precision}")

    def _build_loader(self, data_path: str, shuffle: bool) -> DataLoader:
        """Create a DataLoader over a packed .bin token file."""
        dataset = PackedTokenDataset(data_path, self.config.train.sequence_length)
        return DataLoader(
            dataset,
            batch_size=self.config.train.batch_size,
            shuffle=shuffle,
            drop_last=True,
            pin_memory=(self.device == "cuda"),
        )

    def _forward_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """One forward pass under autocast, returning the loss."""
        with torch.autocast(self.device, dtype=self.amp_dtype, enabled=self.use_amp):
            logits = self.model(inputs)
        # Loss in fp32 outside autocast (see loss.py for why).
        return cross_entropy_loss(logits, targets)

    def _optimizer_step(self) -> float:
        """Clip gradients and apply one optimizer + scheduler step.

        Returns:
            The pre-clip gradient norm (for logging).
        """
        train_cfg = self.config.train
        self.grad_scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), train_cfg.grad_clip_norm
        )
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return float(grad_norm)

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """Average loss over up to ``eval_steps`` validation batches."""
        self.model.eval()
        losses: list[float] = []
        for batch_index, (inputs, targets) in enumerate(val_loader):
            if batch_index >= self.config.train.eval_steps:
                break
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            losses.append(self._forward_loss(inputs, targets).item())
        self.model.train()
        return sum(losses) / max(1, len(losses))

    def _log(self, loss: float, grad_norm: float, tokens_per_second: float) -> None:
        """Print one training log line."""
        lr = self.scheduler.get_last_lr()[0]
        memory = ""
        if self.device == "cuda":
            memory = f" | mem {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB"
        print(
            f"step {self.step:6d} | loss {loss:.4f} | lr {lr:.2e} "
            f"| grad_norm {grad_norm:.2f} | {tokens_per_second:,.0f} tok/s{memory}"
        )

    def train(self) -> None:
        """Run the full training loop until ``max_steps``."""
        train_cfg = self.config.train
        train_loader = self._build_loader(train_cfg.train_data_path, shuffle=True)
        val_loader = self._build_loader(train_cfg.val_data_path, shuffle=False)

        tokens_per_micro_batch = train_cfg.batch_size * train_cfg.sequence_length
        self.model.train()

        accumulated_loss = 0.0
        micro_step = 0
        window_start_time = time.perf_counter()
        window_tokens = 0

        while self.step < train_cfg.max_steps:
            self.epoch += 1
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                loss = self._forward_loss(inputs, targets)
                # Divide so accumulated gradients average over micro-batches.
                self.grad_scaler.scale(loss / train_cfg.gradient_accumulation_steps).backward()
                accumulated_loss += loss.item() / train_cfg.gradient_accumulation_steps
                window_tokens += tokens_per_micro_batch
                micro_step += 1

                if micro_step % train_cfg.gradient_accumulation_steps != 0:
                    continue

                grad_norm = self._optimizer_step()
                self.step += 1

                if self.step % train_cfg.log_interval == 0:
                    elapsed = time.perf_counter() - window_start_time
                    self._log(accumulated_loss, grad_norm, window_tokens / elapsed)
                    window_start_time = time.perf_counter()
                    window_tokens = 0
                accumulated_loss = 0.0

                if self.step % train_cfg.eval_interval == 0:
                    val_loss = self.evaluate(val_loader)
                    print(f"step {self.step:6d} | val_loss {val_loss:.4f}")

                if self.step % train_cfg.checkpoint_interval == 0:
                    self.save(Path(train_cfg.checkpoint_dir) / f"step_{self.step}.pt")

                if self.step >= train_cfg.max_steps:
                    break

        self.save(Path(train_cfg.checkpoint_dir) / "final.pt")
        print("Training complete.")

    def save(self, path: str | Path) -> None:
        """Save a full resume-able checkpoint."""
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
            epoch=self.epoch,
            config=self.config.to_dict(),
        )
        print(f"Checkpoint saved: {path}")

    def resume(self, path: str | Path) -> None:
        """Restore model/optimizer/scheduler/RNG state from a checkpoint."""
        checkpoint = load_checkpoint(
            path, self.model, self.optimizer, self.scheduler, device=self.device
        )
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        print(f"Resumed from {path} at step {self.step}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the SLM")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    config = Config.from_json(args.config)
    trainer = Trainer(config)
    if args.resume:
        trainer.resume(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
