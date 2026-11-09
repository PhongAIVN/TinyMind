"""RL: log-prob lúc sinh = tính lại; advantage nhóm trung bình 0; GAE λ=0 = TD một bước;
token kết quả tool có mask 0; GRPO/CISPO/PPO chạy trọn một vòng nhỏ."""

import json
from pathlib import Path

import pytest
import torch

from core.config import ModelConfig
from core.model import CausalLM
from core.value_model import ValueModel
from trainer.rl import (
    Episode, collate_episodes, gae, group_advantages, kl_to_ref, policy_loss, sample_with_log_probs, token_log_probs,
)
from trainer.rl_config import RLConfig
from trainer.train_ppo import token_rewards

REAL_TOKENIZER = Path("tokenizer/tokenizer.json")
needs_tokenizer = pytest.mark.skipif(not REAL_TOKENIZER.exists(), reason="cần tokenizer thật")


def test_sampled_log_probs_equal_recomputed(tiny_model_config):
    torch.manual_seed(0)
    model = CausalLM(tiny_model_config)
    prompt = [1, 5, 9, 2]
    tokens, log_probs = sample_with_log_probs(model, prompt, n=3, max_new_tokens=6, temperature=0.8, eos_id=7)
    episodes = [Episode(prompt + t, [False] * len(prompt) + [True] * len(t), lp, "", 0.0) for t, lp in zip(tokens, log_probs)]
    batch = collate_episodes(episodes, pad_id=0, device=torch.device("cpu"))

    recomputed = token_log_probs(model, batch["sequences"], temperature=0.8)

    mask = batch["mask"].bool()
    torch.testing.assert_close(recomputed[mask], batch["old_log_probs"][mask], atol=1e-5, rtol=1e-5)


def test_group_advantages_have_zero_mean_per_group():
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    advantages = group_advantages(rewards, group_size=4)
    torch.testing.assert_close(advantages.view(2, 4).mean(dim=1), torch.zeros(2), atol=1e-6, rtol=0)
    assert torch.all(advantages[4:] == 0)  # nhóm toàn đúng: không có gì để học
    assert advantages[0] > 0 > advantages[1]


def test_gae_lambda_zero_is_one_step_td():
    torch.manual_seed(1)
    rewards, values = torch.randn(2, 5), torch.randn(2, 5)
    mask = torch.tensor([[0, 1, 1, 1, 0], [0, 0, 1, 1, 1]], dtype=torch.float)
    advantages, _ = gae(rewards, values, mask, gamma=0.9, lam=0.0)
    for row in range(2):
        positions = mask[row].nonzero().flatten().tolist()
        for i, t in enumerate(positions):
            next_value = values[row, positions[i + 1]] if i + 1 < len(positions) else 0.0
            expected = rewards[row, t] + 0.9 * next_value - values[row, t]
            assert advantages[row, t].item() == pytest.approx(float(expected), abs=1e-6)
    assert torch.all(advantages[mask == 0] == 0)


def test_gae_lambda_one_is_return_minus_value_and_skips_tool_tokens():
    rewards = torch.tensor([[0.0, 0.5, 9.0, 0.0, 1.0]])
    values = torch.tensor([[0.0, 0.2, 5.0, 0.3, 0.4]])
    mask = torch.tensor([[0.0, 1.0, 0.0, 1.0, 1.0]])  # vị trí 2 = token tool: bỏ qua cả reward lẫn value
    advantages, returns = gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert advantages[0, 1].item() == pytest.approx(0.5 + 0.0 + 1.0 - 0.2)
    assert advantages[0, 3].item() == pytest.approx(1.0 - 0.3)
    assert returns[0, 1].item() == pytest.approx(1.5)


def test_tool_result_tokens_have_mask_zero():
    ids = [1, 2, 3, 40, 41, 42, 50, 51, 60, 61]
    generated = [False, False, False, True, True, True, False, False, True, True]  # 50, 51 = kết quả tool
    episode = Episode(ids, generated, [-0.1, -0.2, -0.3, -0.4, -0.5], "", 1.0)
    batch = collate_episodes([episode], pad_id=0, device=torch.device("cpu"))
    assert batch["mask"][0].tolist() == [0, 0, 1, 1, 1, 0, 0, 1, 1]
    assert batch["old_log_probs"][0, 7].item() == pytest.approx(-0.4)


def test_kl_estimator_is_zero_at_ref_and_non_negative():
    log_probs = torch.randn(4, 6)
    assert torch.all(kl_to_ref(log_probs, log_probs) == 0)
    assert torch.all(kl_to_ref(log_probs, torch.randn(4, 6)) >= 0)


def test_grpo_and_cispo_gradients_at_ratio_one():
    """ρ = 1: GRPO và CISPO cùng ra gradient -A·∇log π (policy gradient thuần)."""
    advantages = torch.tensor([1.0, -2.0])
    mask = torch.ones(2, 3)
    grads = []
    for algo in ("grpo", "cispo"):
        log_probs = torch.full((2, 3), -1.0, requires_grad=True)
        policy_loss(log_probs, log_probs.detach(), advantages, mask, algo, 0.2, 0.28).backward()
        grads.append(log_probs.grad)
    torch.testing.assert_close(grads[0], grads[1])
    torch.testing.assert_close(grads[0][:, 0], -advantages / 6)


def test_grpo_clips_but_cispo_keeps_gradient_when_ratio_is_large():
    log_probs = torch.tensor([[0.0]], requires_grad=True)
    old = torch.tensor([[-1.0]])  # ρ = e ≈ 2.7 > 1 + ε
    for algo, has_gradient in (("grpo", False), ("cispo", True)):
        log_probs.grad = None
        policy_loss(log_probs, old, torch.tensor([1.0]), torch.ones(1, 1), algo, 0.2, 0.28).backward()
        assert (log_probs.grad.abs().item() > 0) == has_gradient, algo


def test_token_rewards_put_final_reward_on_last_generated_token():
    batch = {"mask": torch.tensor([[0.0, 1.0, 1.0, 0.0]]), "old_log_probs": torch.zeros(1, 4), "rewards": torch.tensor([1.0])}
    rewards = token_rewards(batch, torch.zeros(1, 4), beta_kl=0.1)
    assert rewards.tolist() == [[0.0, 0.0, 1.0, 0.0]]


def test_value_model_starts_at_zero_and_loads_policy_body(tiny_model_config):
    policy = CausalLM(tiny_model_config)
    critic = ValueModel(tiny_model_config)
    critic.load_body(policy.state_dict())
    assert torch.all(critic(torch.randint(0, tiny_model_config.vocab_size, (2, 5))) == 0)
    for key, value in critic.model.state_dict().items():
        torch.testing.assert_close(value, policy.model.state_dict()[key])


def test_rl_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "rl.json"
    path.write_text(json.dumps({"prompts": "a", "eval_prompts": "b", "init_from": "c", "out_dir": "d",
                                "checkpoint_name": "e.pt", "algo": "grpo", "mode": "plain", "steps": 1, "lr_typo": 1}))
    with pytest.raises(ValueError, match="lr_typo"):
        RLConfig.from_json(path)


@needs_tokenizer
@pytest.mark.parametrize("algo, mode", [("grpo", "plain"), ("cispo", "think"), ("grpo", "tools"), ("ppo", "plain")])
def test_rl_training_runs_end_to_end(tmp_path, algo, mode):
    from tokenizers import Tokenizer

    from scripts.build_toy_math import build_toy_math
    from trainer.train_grpo import train_grpo
    from trainer.train_ppo import train_ppo

    tokenizer = Tokenizer.from_file(str(REAL_TOKENIZER))
    model_config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(), d_model=16, n_layers=1, n_heads=2,
        ffn_hidden=32, max_seq_len=128, rope_theta=10000.0, norm_eps=1e-6, tie_embedding=True,
    )
    torch.manual_seed(0)
    torch.save({"model": CausalLM(model_config).state_dict()}, tmp_path / "start.pt")
    build_toy_math(tmp_path / "math", n_train=20, n_eval=4)
    config = RLConfig(
        prompts=str(tmp_path / "math" / "train.jsonl"), eval_prompts=str(tmp_path / "math" / "eval.jsonl"),
        init_from=str(tmp_path / "start.pt"), out_dir=str(tmp_path / "out"), checkpoint_name="rl.pt",
        algo=algo, mode=mode, steps=2, prompts_per_step=2, group_size=1 if algo == "ppo" else 2,
        max_new_tokens=6, lr=1e-4, min_lr=1e-5, warmup_steps=0, eval_interval=10, eval_limit=2,
        dtype="fp32", attention="manual", device="cpu",
    )
    history = (train_ppo if algo == "ppo" else train_grpo)(model_config, config, tokenizer)
    assert 0.0 <= history["eval_end"] <= 1.0
    assert torch.load(tmp_path / "out" / "rl.pt", weights_only=False)["step"] == 2
