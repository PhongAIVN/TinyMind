"""SDPA (F.scaled_dot_product_attention) phải cho cùng kết quả với attention tự viết."""

import copy

import pytest
import torch

from core.attention import CausalSelfAttention, enable_sdpa
from core.model import CausalLM
from core.rope import precompute_rope

SEQ_LEN = 8


def rope_for(config, seq_len):
    cos, sin = precompute_rope(config.head_dim, config.max_seq_len, config.rope_theta)
    return cos[:seq_len], sin[:seq_len]


@pytest.fixture
def lm_pair(tiny_config):
    """Hai model cùng trọng số: một dùng attention tự viết, một dùng SDPA."""
    torch.manual_seed(0)
    manual = CausalLM(tiny_config)
    sdpa = enable_sdpa(copy.deepcopy(manual))
    return manual, sdpa


def test_attention_sdpa_matches_manual(tiny_config):
    attn = CausalSelfAttention(tiny_config)
    cos, sin = rope_for(tiny_config, SEQ_LEN)
    x = torch.randn(2, SEQ_LEN, tiny_config.d_model)

    manual_out = attn(x, cos, sin)
    attn.use_sdpa = True
    sdpa_out = attn(x, cos, sin)

    torch.testing.assert_close(sdpa_out, manual_out)


def test_sdpa_blocks_future(tiny_config):
    attn = CausalSelfAttention(tiny_config)
    attn.use_sdpa = True
    cos, sin = rope_for(tiny_config, SEQ_LEN)
    x = torch.randn(2, SEQ_LEN, tiny_config.d_model)
    x_changed = x.clone()
    x_changed[:, -1] = torch.randn(2, tiny_config.d_model)

    torch.testing.assert_close(attn(x_changed, cos, sin)[:, :-1], attn(x, cos, sin)[:, :-1])


def test_enable_sdpa_toggles_every_layer_without_touching_weights(lm_pair):
    manual, sdpa = lm_pair
    layers = [m for m in sdpa.modules() if isinstance(m, CausalSelfAttention)]

    assert layers and all(layer.use_sdpa for layer in layers)
    for (name, a), (_, b) in zip(manual.state_dict().items(), sdpa.state_dict().items()):
        assert torch.equal(a, b), name

    enable_sdpa(sdpa, enabled=False)
    assert not any(layer.use_sdpa for layer in layers)


def test_model_logits_and_gradients_match(lm_pair, tiny_config):
    """Train bằng SDPA phải ra cùng loss và gradient như bản tự viết."""
    manual, sdpa = lm_pair
    ids = torch.randint(0, tiny_config.vocab_size, (2, SEQ_LEN))

    manual_logits, sdpa_logits = manual(ids), sdpa(ids)
    torch.testing.assert_close(sdpa_logits, manual_logits)

    manual_logits.sum().backward()
    sdpa_logits.sum().backward()
    for (name, a), (_, b) in zip(manual.named_parameters(), sdpa.named_parameters()):
        torch.testing.assert_close(b.grad, a.grad, msg=name)


def test_sdpa_step_by_step_cache_matches_full_forward(lm_pair, tiny_config):
    manual, sdpa = lm_pair
    ids = torch.randint(0, tiny_config.vocab_size, (2, SEQ_LEN))
    full_logits = manual(ids)

    step_logits, past_kvs = [], None
    for position in range(SEQ_LEN):
        logits, past_kvs = sdpa(ids[:, position : position + 1], past_kvs=past_kvs, use_cache=True)
        step_logits.append(logits)

    torch.testing.assert_close(torch.cat(step_logits, dim=1), full_logits)


def test_sdpa_multi_token_chunk_after_cache(lm_pair, tiny_config):
    """Có cache VÀ nhiều query cùng lúc: nhánh phải tự tạo mask lệch past_len."""
    manual, sdpa = lm_pair
    ids = torch.randint(0, tiny_config.vocab_size, (2, SEQ_LEN))
    full_logits = manual(ids)

    _, past_kvs = sdpa(ids[:, :5], use_cache=True)
    chunk_logits, _ = sdpa(ids[:, 5:], past_kvs=past_kvs, use_cache=True)

    torch.testing.assert_close(chunk_logits, full_logits[:, 5:])
