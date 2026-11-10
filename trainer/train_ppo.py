"""PPO: actor (policy) + critic (ValueModel), advantage theo từng token bằng GAE.

Reward theo token: r_t = -β·(log π_old - log π_ref) ở mọi token sinh ra, cộng reward đúng/sai
ở token cuối. Khác GRPO: không cần nhiều câu cho một câu hỏi, nhưng tốn thêm một critic.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

from core.attention import enable_sdpa
from core.config import ModelConfig
from core.value_model import ValueModel
from trainer.checkpoint import save_checkpoint
from trainer.lr_scheduler import get_lr
from trainer.optim import build_optimizer
from trainer.rl import collate_episodes, gae, masked_mean, policy_loss, token_log_probs
from trainer.rl_config import RLConfig
from trainer.sft_dataset import ChatEncoder
from trainer.train_grpo import autocast_for, collect_rollouts, evaluate_policy, load_policy, read_prompts
from trainer.train_sft import check_tokenizer_matches

VALUE_CLIP = 0.2
ADVANTAGE_EPS = 1e-8


def token_rewards(batch: dict, ref_log_probs: torch.Tensor, beta_kl: float) -> torch.Tensor:
    """Phạt KL theo từng token + reward cuối câu đặt ở token sinh ra cuối cùng."""
    mask = batch["mask"]
    rewards = -beta_kl * (batch["old_log_probs"] - ref_log_probs) * mask
    last = mask.shape[1] - 1 - mask.flip(dims=[1]).argmax(dim=1)
    rewards[torch.arange(mask.shape[0], device=mask.device), last] += batch["rewards"]
    return rewards


def whiten(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = masked_mean(values, mask)
    variance = masked_mean((values - mean) ** 2, mask)
    return (values - mean) / (variance.sqrt() + ADVANTAGE_EPS) * mask


def build_critic(model_config: ModelConfig, policy, config: RLConfig, device: torch.device) -> ValueModel:
    critic = enable_sdpa(ValueModel(model_config), config.attention == "sdpa")
    critic.load_body({key: value.detach().cpu() for key, value in policy.state_dict().items()})
    return critic.to(device)


def ppo_update(policy, critic, optimizers, batch, advantages, returns, old_values, config, autocast) -> dict:
    policy.train()
    critic.train()
    for _ in range(config.inner_epochs):
        with autocast():
            log_probs = token_log_probs(policy, batch["sequences"], config.temperature)
            values = critic(batch["sequences"][:, :-1]).float()
        loss_policy = policy_loss(log_probs, batch["old_log_probs"], advantages, batch["mask"],
                                  "ppo", config.clip_eps, config.clip_eps_high)
        clipped = old_values + (values - old_values).clamp(-VALUE_CLIP, VALUE_CLIP)
        loss_value = 0.5 * masked_mean(torch.max((values - returns) ** 2, (clipped - returns) ** 2), batch["mask"])
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        (loss_policy + loss_value).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config.grad_clip)
        for optimizer in optimizers:
            optimizer.step()
    return {"policy": float(loss_policy.detach()), "value": float(loss_value.detach())}


def train_ppo(model_config: ModelConfig, config: RLConfig, tokenizer) -> dict[str, float]:
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    autocast = autocast_for(config, device)
    encoder = ChatEncoder(tokenizer)
    policy = load_policy(model_config, config, device, trainable=True)
    ref = load_policy(model_config, config, device, trainable=False)
    critic = build_critic(model_config, policy, config, device)
    actor_optimizer = build_optimizer(policy, lr=config.lr, weight_decay=config.weight_decay)
    critic_optimizer = build_optimizer(critic, lr=config.critic_lr, weight_decay=0.0)
    examples = read_prompts(config.prompts)
    rng = random.Random(config.seed)

    accuracy = evaluate_policy(policy, tokenizer, config, autocast)
    print(f"[ppo/{config.mode}] trước khi train: eval {accuracy:.1%}")
    history = {"eval_start": accuracy}
    for step in range(config.steps):
        lr = get_lr(step, max_lr=config.lr, min_lr=config.min_lr, warmup_steps=config.warmup_steps, max_steps=config.steps)
        for group in actor_optimizer.param_groups:
            group["lr"] = lr
        episodes = collect_rollouts(policy, encoder, rng.sample(examples, config.prompts_per_step), config, autocast)
        batch = collate_episodes(episodes, encoder.pad_id, device)
        with torch.no_grad(), autocast():
            ref_log_probs = token_log_probs(ref, batch["sequences"], config.temperature)
            old_values = critic(batch["sequences"][:, :-1]).float() * batch["mask"]
        rewards = token_rewards(batch, ref_log_probs, config.beta_kl)
        advantages, returns = gae(rewards, old_values, batch["mask"], config.gamma, config.lam)
        stats = ppo_update(policy, critic, (actor_optimizer, critic_optimizer), batch,
                           whiten(advantages, batch["mask"]), returns, old_values, config, autocast)
        if (step + 1) % config.log_interval == 0:
            print(f"step {step + 1:>5}/{config.steps} | reward {batch['rewards'].mean():.3f} | policy {stats['policy']:+.4f} "
                  f"| value {stats['value']:.4f} | lr {lr:.2e}")
        if (step + 1) % config.eval_interval == 0 or step + 1 == config.steps:
            accuracy = evaluate_policy(policy, tokenizer, config, autocast)
            print(f"step {step + 1:>5} | eval {accuracy:.1%}")
        if (step + 1) % config.save_interval == 0 or step + 1 == config.steps:
            save_checkpoint(Path(config.out_dir) / config.checkpoint_name, policy, actor_optimizer, step + 1, model_config, config)
    history["eval_end"] = accuracy
    return history


def main() -> None:
    """python -m trainer.train_ppo --rl-config configs/ppo_50m.json"""
    parser = argparse.ArgumentParser(description="PPO với critic và GAE")
    parser.add_argument("--rl-config", required=True)
    parser.add_argument("--model-config", default="configs/model_50m.json")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    args = parser.parse_args()
    config = RLConfig.from_json(args.rl_config)
    if config.algo != "ppo":
        raise ValueError("train_ppo cần algo=ppo")
    model_config = ModelConfig.from_dict(json.loads(Path(args.model_config).read_text(encoding="utf-8")))
    tokenizer = Tokenizer.from_file(args.tokenizer)
    check_tokenizer_matches(tokenizer, model_config, args.tokenizer)
    history = train_ppo(model_config, config, tokenizer)
    print(f"Xong: eval {history['eval_start']:.1%} -> {history['eval_end']:.1%}")


if __name__ == "__main__":
    main()
