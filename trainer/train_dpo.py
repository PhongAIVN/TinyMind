"""DPO: dạy model thích câu trả lời chosen hơn rejected, so với một bản sao đóng băng (ref).

Policy và ref cùng khởi đầu từ init_from; ref không bao giờ được cập nhật.
Tiêu chí "xong" (CLAUDE.md): loss = ln 2 khi policy = ref; tỉ lệ chosen thắng tăng.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from core.attention import enable_sdpa
from core.config import ModelConfig
from core.model import CausalLM
from trainer.checkpoint import load_weights
from trainer.dpo_dataset import DPODataset
from trainer.losses import IGNORE_INDEX, dpo_loss, sequence_log_probs
from trainer.train_config import TrainConfig
from trainer.train_pretrain import LossFn, run_training
from trainer.train_sft import check_tokenizer_matches

CHECKPOINT_NAME = "dpo.pt"
DEFAULT_BETA = 0.1


def pair_log_probs(model: nn.Module, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-prob cả câu của chosen và rejected, ghép làm một batch để chỉ forward 1 lần."""
    x_chosen, y_chosen, x_rejected, y_rejected = batch
    log_probs = sequence_log_probs(model(torch.cat([x_chosen, x_rejected])), torch.cat([y_chosen, y_rejected]))
    n_pairs = x_chosen.shape[0]
    return log_probs[:n_pairs], log_probs[n_pairs:]


def make_dpo_loss(ref_model: nn.Module, beta: float, sft_weight: float = 0.0) -> LossFn:
    """Hàm loss cho run_training; ref chạy không gradient.

    sft_weight > 0 (kiểu RPO): cộng w · NLL trung bình mỗi token của câu chosen. DPO chỉ đòi
    khoảng cách chosen - rejected tăng, nên có thể kéo CẢ HAI xuống; hạng này giữ chosen không tụt.
    """

    def compute_loss(model: nn.Module, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        policy_chosen, policy_rejected = pair_log_probs(model, batch)
        with torch.no_grad():
            ref_chosen, ref_rejected = pair_log_probs(ref_model, batch)
        loss, _, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
        if sft_weight > 0:
            chosen_tokens = (batch[1] != IGNORE_INDEX).sum(dim=-1).clamp(min=1)
            loss = loss + sft_weight * (-(policy_chosen / chosen_tokens)).mean()
        return loss

    return compute_loss


def load_frozen(model_config: ModelConfig, path: str | Path, device: torch.device, use_sdpa: bool) -> CausalLM:
    model = enable_sdpa(CausalLM(model_config), use_sdpa)
    load_weights(path, model, map_location="cpu")
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def preference_stats(policy: nn.Module, ref: nn.Module, dataset: DPODataset, beta: float, batch_size: int) -> dict:
    """Tỉ lệ cặp có reward chosen > reward rejected, và độ chênh trung bình."""
    device = next(policy.parameters()).device
    wins, margins = 0, []
    for batch in DataLoader(dataset, batch_size=batch_size):
        batch = tuple(tensor.to(device) for tensor in batch)
        _, reward_chosen, reward_rejected = dpo_loss(*pair_log_probs(policy, batch), *pair_log_probs(ref, batch), beta)
        margin = reward_chosen - reward_rejected
        wins += int((margin > 0).sum())
        margins.append(margin)
    return {"win_rate": wins / len(dataset), "margin": float(torch.cat(margins).mean())}


def main() -> None:
    """python -m trainer.train_dpo --model-config configs/model_50m.json --train-config configs/dpo_50m.json"""
    parser = argparse.ArgumentParser(description="DPO trên cặp chosen/rejected")
    parser.add_argument("--model-config", default="configs/model_50m.json")
    parser.add_argument("--train-config", default="configs/dpo_50m.json")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="β càng lớn càng bám sát ref")
    parser.add_argument("--sft-weight", type=float, default=1.0, help="hệ số loss SFT trên chosen (0 = DPO thuần)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model_config = ModelConfig.from_dict(json.loads(Path(args.model_config).read_text(encoding="utf-8")))
    train_config = TrainConfig.from_json(args.train_config)
    if not train_config.init_from:
        raise ValueError("DPO cần init_from: điểm xuất phát của policy và cũng là model tham chiếu (ref)")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    check_tokenizer_matches(tokenizer, model_config, args.tokenizer)

    device = torch.device(train_config.device)
    use_sdpa = train_config.attention == "sdpa"
    ref = load_frozen(model_config, train_config.init_from, device, use_sdpa)
    train_dataset = DPODataset(train_config.train_bin, tokenizer, train_config.seq_len)
    eval_dataset = DPODataset(train_config.eval_bin, tokenizer, train_config.seq_len)
    print(f"DPO β={args.beta} sft_weight={args.sft_weight}: {len(train_dataset):,} cặp train, {len(eval_dataset):,} cặp eval (loss ban đầu phải ≈ ln 2 = 0.693)")

    stats = run_training(
        model_config, train_config, train_dataset, eval_dataset, CHECKPOINT_NAME, args.resume,
        compute_loss=make_dpo_loss(ref, args.beta, args.sft_weight),
    )
    print(f"Xong: {stats}")

    policy = load_frozen(model_config, Path(train_config.out_dir) / CHECKPOINT_NAME, device, use_sdpa)
    preference = preference_stats(policy, ref, eval_dataset, args.beta, train_config.batch_size)
    print(f"Eval: chosen thắng {preference['win_rate']:.1%} cặp, reward chênh trung bình {preference['margin']:.3f}")


if __name__ == "__main__":
    main()
