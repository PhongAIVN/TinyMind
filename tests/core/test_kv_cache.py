import torch

from core.model import CausalLM, Model

SEQ_LEN = 6


def test_forward_without_cache_returns_tensor(tiny_config):
    """API cũ không đổi: không bật cache thì vẫn trả về đúng một tensor."""
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, SEQ_LEN))

    assert isinstance(lm(ids), torch.Tensor)


def test_cache_shapes(tiny_config):
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, SEQ_LEN))

    _, past_kvs = lm(ids, use_cache=True)

    assert len(past_kvs) == tiny_config.n_layers
    for key, value in past_kvs:
        assert key.shape == (2, tiny_config.n_heads, SEQ_LEN, tiny_config.head_dim)
        assert value.shape == key.shape


def test_step_by_step_with_cache_matches_full_forward(tiny_config):
    """Tiêu chí xong chặng 3: đưa từng token một kèm cache = đưa cả chuỗi một lần."""
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, SEQ_LEN))

    full_logits = lm(ids)

    step_logits = []
    past_kvs = None
    for position in range(SEQ_LEN):
        logits, past_kvs = lm(ids[:, position : position + 1], past_kvs=past_kvs, use_cache=True)
        step_logits.append(logits)

    torch.testing.assert_close(torch.cat(step_logits, dim=1), full_logits)


def test_prefill_then_one_token_matches_full_forward(tiny_config):
    """Cách dùng thật: nạp cả prompt một lần (prefill), rồi sinh từng token."""
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, SEQ_LEN))

    full_logits = lm(ids)

    _, past_kvs = lm(ids[:, :-1], use_cache=True)
    last_logits, _ = lm(ids[:, -1:], past_kvs=past_kvs, use_cache=True)

    torch.testing.assert_close(last_logits[:, -1], full_logits[:, -1])


def test_cache_grows_by_one_each_step(tiny_config):
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (1, 3))

    _, past_kvs = lm(ids, use_cache=True)
    assert past_kvs[0][0].shape[-2] == 3

    _, past_kvs = lm(ids[:, :1], past_kvs=past_kvs, use_cache=True)
    assert past_kvs[0][0].shape[-2] == 4


def test_model_body_also_supports_cache(tiny_config):
    """Thân model (dùng cho Critic ở PPO) cũng phải trả cache."""
    model = Model(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (1, 3))

    hidden, past_kvs = model(ids, use_cache=True)

    assert hidden.shape == (1, 3, tiny_config.d_model)
    assert len(past_kvs) == tiny_config.n_layers


def test_chunked_prefill_matches_full_forward(tiny_config):
    """Đưa 3 token, rồi đưa tiếp 3 token CÙNG LÚC kèm cache = đưa cả 6 token một lần."""
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 6))

    full_logits = lm(ids)

    _, past_kvs = lm(ids[:, :3], use_cache=True)
    chunk_logits, _ = lm(ids[:, 3:], past_kvs=past_kvs, use_cache=True)

    torch.testing.assert_close(chunk_logits, full_logits[:, 3:])
