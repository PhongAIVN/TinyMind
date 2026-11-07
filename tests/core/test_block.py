import torch

from core.block import Block
from core.rope import precompute_rope


def rope_for(config, seq_len):
    cos, sin = precompute_rope(config.head_dim, config.max_seq_len, config.rope_theta)
    return cos[:seq_len], sin[:seq_len]


def test_output_shape_is_unchanged(tiny_config):
    block = Block(tiny_config)
    cos, sin = rope_for(tiny_config, 6)
    x = torch.randn(2, 6, tiny_config.d_model)

    assert block(x, cos, sin).shape == (2, 6, tiny_config.d_model)


def test_parameter_count(tiny_config):
    block = Block(tiny_config)
    d, ffn = tiny_config.d_model, tiny_config.ffn_hidden

    expected = 4 * d * d + 3 * d * ffn + 2 * d  # attention + ffn + 2 RMSNorm

    assert sum(p.numel() for p in block.parameters()) == expected


def test_block_is_causal(tiny_config):
    block = Block(tiny_config)
    cos, sin = rope_for(tiny_config, 6)
    x = torch.randn(2, 6, tiny_config.d_model)
    x_changed = x.clone()
    x_changed[:, -1] = torch.randn(2, tiny_config.d_model)

    torch.testing.assert_close(block(x_changed, cos, sin)[:, :-1], block(x, cos, sin)[:, :-1])