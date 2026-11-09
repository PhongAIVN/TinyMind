"""GRPO / CISPO / Agentic RL trên thế giới đồ chơi, reward theo quy tắc.

Mỗi bước: lấy prompts_per_step câu hỏi, mỗi câu sinh group_size câu trả lời (lưu log-prob lúc sinh),
chấm đúng/sai, advantage = chuẩn hóa trong nhóm, cập nhật chính sách + phạt KL tới ref.
mode = "tools": model được gọi máy tính giữa chừng; token kết quả tool có mask 0 (không học).
"""

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from tokenizers import Tokenizer

from core.attention import enable_sdpa
from core.config import ModelConfig
from core.model import CausalLM
from inference.eval_toy_math import evaluate_toy_math
from trainer.checkpoint import load_weights, save_checkpoint
from trainer.lr_scheduler import get_lr
from trainer.optim import build_optimizer
from trainer.rl import (
    collate_episodes, group_advantages, kl_to_ref, masked_mean, policy_loss, rollout, token_log_probs,
)
from trainer.rl_config import RLConfig
from trainer.sft_dataset import ChatEncoder
from trainer.train_sft import check_tokenizer_matches

TORCH_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def read_prompts(path: str | Path) -> list[tuple[str, int]]:
    """(câu hỏi, đáp số) từ jsonl thế giới đồ chơi."""
    examples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        examples.append((record["messages"][0]["content"], record["result"]))
    return examples


def autocast_for(config: RLConfig, device: torch.device):
    dtype = TORCH_DTYPES[config.dtype]
    if device.type == "cuda" and dtype is not torch.float32:
        return lambda: torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext


def load_policy(model_config: ModelConfig, config: RLConfig, device: torch.device, trainable: bool) -> CausalLM:
    model = enable_sdpa(CausalLM(model_config), config.attention == "sdpa")
    load_weights(config.init_from, model, map_location="cpu")
    model.to(device)
    if not trainable:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    return model


def collect_rollouts(model, encoder, examples, config: RLConfig, autocast) -> list:
    episodes = []
    with autocast():
        for question, result in examples:
            episodes += rollout(model, encoder, question, result, config.mode, config.group_size,
                                config.max_new_tokens, config.temperature)
    return episodes


def update_policy(model, ref, optimizer, batch, advantages, config: RLConfig, autocast) -> dict[str, float]:
    with torch.no_grad(), autocast():
        ref_log_probs = token_log_probs(ref, batch["sequences"], config.temperature)
    model.train()
    for _ in range(config.inner_epochs):
        with autocast():
            log_probs = token_log_probs(model, batch["sequences"], config.temperature)
        loss_policy = policy_loss(log_probs, batch["old_log_probs"], advantages, batch["mask"],
                                  config.algo, config.clip_eps, config.clip_eps_high)
        kl = masked_mean(kl_to_ref(log_probs, ref_log_probs), batch["mask"])
        loss = loss_policy + config.beta_kl * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
    return {"loss": float(loss.detach()), "kl": float(kl.detach()), "grad_norm": float(grad_norm)}


def evaluate_policy(model, tokenizer, config: RLConfig, autocast) -> float:
    model.eval()
    with torch.no_grad(), autocast():
        return evaluate_toy_math(model, tokenizer, config.eval_prompts, config.eval_limit, config.mode)["accuracy"]


def train_grpo(model_config: ModelConfig, config: RLConfig, tokenizer) -> dict[str, float]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    autocast = autocast_for(config, device)
    encoder = ChatEncoder(tokenizer)
    policy = load_policy(model_config, config, device, trainable=True)
    ref = load_policy(model_config, config, device, trainable=False)
    optimizer = build_optimizer(policy, lr=config.lr, weight_decay=config.weight_decay)
    examples = read_prompts(config.prompts)
    rng = random.Random(config.seed)

    accuracy = evaluate_policy(policy, tokenizer, config, autocast)
    print(f"[{config.algo}/{config.mode}] trước khi train: eval {accuracy:.1%}")
    history = {"eval_start": accuracy}
    for step in range(config.steps):
        lr = get_lr(step, max_lr=config.lr, min_lr=config.min_lr, warmup_steps=config.warmup_steps, max_steps=config.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        episodes = collect_rollouts(policy, encoder, rng.sample(examples, config.prompts_per_step), config, autocast)
        batch = collate_episodes(episodes, encoder.pad_id, device)
        advantages = group_advantages(batch["rewards"], config.group_size)
        stats = update_policy(policy, ref, optimizer, batch, advantages, config, autocast)
        if (step + 1) % config.log_interval == 0:
            length = batch["mask"].sum(dim=1).mean()
            print(f"step {step + 1:>5}/{config.steps} | reward {batch['rewards'].mean():.3f} | loss {stats['loss']:+.4f} "
                  f"| kl {stats['kl']:.4f} | gnorm {stats['grad_norm']:.3f} | len {length:.1f} | lr {lr:.2e}")
        if (step + 1) % config.eval_interval == 0 or step + 1 == config.steps:
            accuracy = evaluate_policy(policy, tokenizer, config, autocast)
            print(f"step {step + 1:>5} | eval {accuracy:.1%}")
        if (step + 1) % config.save_interval == 0 or step + 1 == config.steps:
            save_checkpoint(Path(config.out_dir) / config.checkpoint_name, policy, optimizer, step + 1, model_config, config)
    history["eval_end"] = accuracy
    return history


def main() -> None:
    """python -m trainer.train_grpo --rl-config configs/grpo_50m.json"""
    parser = argparse.ArgumentParser(description="GRPO / CISPO / Agentic RL")
    parser.add_argument("--rl-config", required=True)
    parser.add_argument("--model-config", default="configs/model_50m.json")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    args = parser.parse_args()
    config = RLConfig.from_json(args.rl_config)
    if config.algo == "ppo":
        raise ValueError("algo=ppo: dùng python -m trainer.train_ppo")
    model_config = ModelConfig.from_dict(json.loads(Path(args.model_config).read_text(encoding="utf-8")))
    tokenizer = Tokenizer.from_file(args.tokenizer)
    check_tokenizer_matches(tokenizer, model_config, args.tokenizer)
    history = train_grpo(model_config, config, tokenizer)
    print(f"Xong: eval {history['eval_start']:.1%} -> {history['eval_end']:.1%}")


if __name__ == "__main__":
    main()
