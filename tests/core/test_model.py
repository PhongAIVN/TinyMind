import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F

from core.model import CausalLM, Model


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())  # tham số dùng chung chỉ đếm 1 lần


def test_model_returns_hidden_states(tiny_config):
    model = Model(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 6))

    assert model(ids).shape == (2, 6, tiny_config.d_model)


def test_causal_lm_returns_logits(tiny_config):
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 6))

    assert lm(ids).shape == (2, 6, tiny_config.vocab_size)


def test_embedding_and_lm_head_share_weights(tiny_config):
    lm = CausalLM(tiny_config)

    assert lm.lm_head.weight is lm.model.embed.weight


def test_parameter_count_matches_formula(tiny_config):
    lm = CausalLM(tiny_config)
    V, d, L, ffn = tiny_config.vocab_size, tiny_config.d_model, tiny_config.n_layers, tiny_config.ffn_hidden

    expected = V * d + L * (4 * d * d + 3 * d * ffn + 2 * d) + d

    assert count_params(lm) == expected


def test_untied_model_has_extra_lm_head(tiny_config):
    untied_config = dataclasses.replace(tiny_config, tie_embedding=False)

    tied = count_params(CausalLM(tiny_config))
    untied = count_params(CausalLM(untied_config))

    assert untied - tied == tiny_config.vocab_size * tiny_config.d_model


def test_initial_loss_is_close_to_log_vocab_size(tiny_config):
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (4, 16))

    logits = lm(ids)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, tiny_config.vocab_size),
        ids[:, 1:].reshape(-1),
    )

    assert abs(loss.item() - math.log(tiny_config.vocab_size)) < 0.3


def test_model_is_causal(tiny_config):
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
    ids_changed = ids.clone()
    ids_changed[:, -1] = (ids[:, -1] + 1) % tiny_config.vocab_size

    torch.testing.assert_close(lm(ids_changed)[:, :-1], lm(ids)[:, :-1])


def test_prefix_gives_same_logits(tiny_config):
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 10))

    torch.testing.assert_close(lm(ids[:, :4]), lm(ids)[:, :4])


def test_sequence_longer_than_max_raises(tiny_config):
    lm = CausalLM(tiny_config)
    ids = torch.zeros(1, tiny_config.max_seq_len + 1, dtype=torch.long)

    with pytest.raises(ValueError, match="max_seq_len"):
        lm(ids)


def test_can_overfit_one_batch(tiny_config):
    lm = CausalLM(tiny_config)
    optimizer = torch.optim.AdamW(lm.parameters(), lr=3e-3)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 16))

    for _ in range(100):
        logits = lm(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, tiny_config.vocab_size), ids[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.5