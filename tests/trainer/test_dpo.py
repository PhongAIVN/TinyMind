"""DPO: loss = ln 2 khi policy = ref, log-prob chỉ tính câu trả lời, train làm chosen thắng."""

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from core.config import ModelConfig
from core.model import CausalLM
from scripts.build_toy_dpo import build_toy_dpo, wrong_result
from scripts.build_toy_math import build_toy_math, parse_answer, parse_problem, solve
from trainer.dpo_dataset import DPODataset
from trainer.losses import IGNORE_INDEX, dpo_loss, sequence_log_probs
from trainer.train_dpo import CHECKPOINT_NAME, load_frozen, make_dpo_loss, pair_log_probs, preference_stats
from trainer.train_pretrain import run_training

REAL_TOKENIZER = Path("tokenizer/tokenizer.json")
needs_tokenizer = pytest.mark.skipif(not REAL_TOKENIZER.exists(), reason="cần tokenizer thật")
PAIR = {"prompt": [{"role": "user", "content": "Tính 2 + 3."}], "chosen": "2 + 3 = 5.", "rejected": "2 + 3 = 3."}


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(REAL_TOKENIZER))


def test_dpo_loss_is_ln2_when_policy_equals_ref():
    log_probs = torch.randn(5)
    other = torch.randn(5)
    loss, reward_chosen, reward_rejected = dpo_loss(log_probs, other, log_probs, other, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2))
    assert torch.all(reward_chosen == 0) and torch.all(reward_rejected == 0)


def test_dpo_gradient_raises_chosen_and_lowers_rejected():
    chosen = torch.zeros(3, requires_grad=True)
    rejected = torch.zeros(3, requires_grad=True)
    loss, _, _ = dpo_loss(chosen, rejected, torch.zeros(3), torch.zeros(3), beta=0.5)
    loss.backward()
    assert torch.all(chosen.grad < 0) and torch.all(rejected.grad > 0)  # gradient descent: chosen ↑, rejected ↓


def test_sequence_log_probs_sums_only_learned_positions():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 7)
    targets = torch.tensor([[1, IGNORE_INDEX, 3, IGNORE_INDEX], [IGNORE_INDEX] * 3 + [6]])
    expected = torch.stack([
        F.log_softmax(logits[0, 0], -1)[1] + F.log_softmax(logits[0, 2], -1)[3],
        F.log_softmax(logits[1, 3], -1)[6],
    ])
    torch.testing.assert_close(sequence_log_probs(logits, targets), expected)


def test_wrong_result_never_equals_answer():
    import random

    rng = random.Random(0)
    for problem in [("+", 0, 0), ("-", 5, 5), ("×", 0, 7), ("+", 99, 99)]:
        assert all(wrong_result(problem, rng) != solve(problem) for _ in range(50))


def test_parse_problem_roundtrip():
    assert parse_problem("57 - 18 = 39.") == ("-", 57, 18)
    with pytest.raises(ValueError):
        parse_problem("không phải phép tính")


def test_toy_dpo_uses_eval_problems_and_disjoint_train(tmp_path):
    build_toy_math(tmp_path / "math", n_train=10, n_eval=40)
    counts = build_toy_dpo(tmp_path / "dpo", tmp_path / "math" / "eval.jsonl", n_train=300)

    def read(name):
        return [json.loads(line) for line in (tmp_path / "dpo" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]

    assert counts == {"train": 300, "eval": 40}
    eval_problems = {parse_problem(pair["chosen"]) for pair in read("eval")}
    for pair in read("train") + read("eval"):
        assert parse_answer(pair["chosen"]) == pair["result"] != parse_answer(pair["rejected"])
    assert not eval_problems & {parse_problem(pair["chosen"]) for pair in read("train")}


@needs_tokenizer
def test_dpo_dataset_shares_prompt_and_labels_only_answer(tokenizer, tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(PAIR, ensure_ascii=False) + "\n", encoding="utf-8")
    x_chosen, y_chosen, x_rejected, y_rejected = DPODataset(path, tokenizer, seq_len=48)[0]

    first_answer = (y_chosen != IGNORE_INDEX).nonzero()[0].item()
    assert torch.equal(x_chosen[: first_answer + 1], x_rejected[: first_answer + 1])  # cùng prompt
    learned = y_chosen[y_chosen != IGNORE_INDEX].tolist()
    assert tokenizer.decode(learned) == "2 + 3 = 5."  # chỉ câu trả lời (+ <|im_end|> bị decode bỏ qua)


@needs_tokenizer
def test_dpo_training_starts_at_ln2_and_makes_chosen_win(tokenizer, tmp_path, make_train_config):
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(), d_model=16, n_layers=1, n_heads=2,
        ffn_hidden=32, max_seq_len=48, rope_theta=10000.0, norm_eps=1e-6, tie_embedding=True,
    )
    torch.manual_seed(0)
    start_path = tmp_path / "start.pt"
    torch.save({"model": CausalLM(config).state_dict()}, start_path)
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps(PAIR, ensure_ascii=False) + "\n", encoding="utf-8")
    dataset = DPODataset(pairs, tokenizer, seq_len=48)
    ref = load_frozen(config, start_path, torch.device("cpu"), use_sdpa=False)
    compute_loss = make_dpo_loss(ref, beta=0.1)

    policy = load_frozen(config, start_path, torch.device("cpu"), use_sdpa=False)
    batch = tuple(t.unsqueeze(0) for t in dataset[0])
    assert compute_loss(policy, batch).item() == pytest.approx(math.log(2), abs=1e-6)

    train_config = make_train_config(
        out_dir=str(tmp_path / "out"), init_from=str(start_path), seq_len=48, batch_size=1,
        accum_steps=1, max_steps=20, warmup_steps=0, lr=1e-2, min_lr=1e-2, weight_decay=0.0,
    )
    run_training(config, train_config, dataset, dataset, CHECKPOINT_NAME, compute_loss=compute_loss)

    trained = load_frozen(config, tmp_path / "out" / CHECKPOINT_NAME, torch.device("cpu"), use_sdpa=False)
    stats = preference_stats(trained, ref, dataset, beta=0.1, batch_size=1)
    assert stats["win_rate"] == 1.0 and stats["margin"] > 0
    chosen, rejected = pair_log_probs(trained, batch)
    ref_chosen, ref_rejected = pair_log_probs(ref, batch)
    assert chosen - rejected > ref_chosen - ref_rejected


@needs_tokenizer
def test_sft_weight_adds_chosen_nll(tokenizer, tmp_path):
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(), d_model=16, n_layers=1, n_heads=2,
        ffn_hidden=32, max_seq_len=48, rope_theta=10000.0, norm_eps=1e-6, tie_embedding=True,
    )
    torch.manual_seed(0)
    model = CausalLM(config)
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps(PAIR, ensure_ascii=False) + "\n", encoding="utf-8")
    batch = tuple(t.unsqueeze(0) for t in DPODataset(pairs, tokenizer, seq_len=48)[0])

    plain = make_dpo_loss(model, beta=0.1)(model, batch)
    with_sft = make_dpo_loss(model, beta=0.1, sft_weight=2.0)(model, batch)
    chosen, _ = pair_log_probs(model, batch)
    n_tokens = (batch[1] != IGNORE_INDEX).sum()
    torch.testing.assert_close(with_sft - plain, 2.0 * (-chosen.sum() / n_tokens))
