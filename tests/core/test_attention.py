import torch

from core.attention import CausalSelfAttention
from core.rope import precompute_rope


def rope_for(config, seq_len):
    cos, sin = precompute_rope(config.head_dim, config.max_seq_len, config.rope_theta)
    return cos[:seq_len], sin[:seq_len]


def test_output_shape_is_unchanged(tiny_config):
    attn = CausalSelfAttention(tiny_config)
    cos, sin = rope_for(tiny_config, 6)
    x = torch.randn(2, 6, tiny_config.d_model)

    assert attn(x, cos, sin).shape == (2, 6, tiny_config.d_model)


def test_parameter_count(tiny_config):
    attn = CausalSelfAttention(tiny_config)
    d = tiny_config.d_model

    assert sum(p.numel() for p in attn.parameters()) == 4 * d * d  # q, k, v, o


def test_future_token_does_not_change_past_outputs(tiny_config):
    attn = CausalSelfAttention(tiny_config)
    cos, sin = rope_for(tiny_config, 6)
    x = torch.randn(2, 6, tiny_config.d_model)
    x_changed = x.clone()
    x_changed[:, -1] = torch.randn(2, tiny_config.d_model)  # chỉ đổi token cuối

    out = attn(x, cos, sin)
    out_changed = attn(x_changed, cos, sin)

    torch.testing.assert_close(out_changed[:, :-1], out[:, :-1])  # quá khứ không đổi
    assert not torch.allclose(out_changed[:, -1], out[:, -1])     # token cuối thì đổi


def test_single_token_sequence_works(tiny_config):
    attn = CausalSelfAttention(tiny_config)
    cos, sin = rope_for(tiny_config, 1)
    x = torch.randn(2, 1, tiny_config.d_model)

    assert attn(x, cos, sin).shape == (2, 1, tiny_config.d_model)