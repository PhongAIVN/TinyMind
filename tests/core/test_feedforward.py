import torch

from core.feedforward import FeedForward

D_MODEL = 64
FFN_HIDDEN = 128


def test_output_shape_is_unchanged():
    ffn = FeedForward(D_MODEL, FFN_HIDDEN)
    x = torch.randn(2, 5, D_MODEL)

    assert ffn(x).shape == (2, 5, D_MODEL)


def test_parameter_count():
    ffn = FeedForward(D_MODEL, FFN_HIDDEN)

    n_params = sum(p.numel() for p in ffn.parameters())

    assert n_params == 3 * D_MODEL * FFN_HIDDEN  # gate + up + down, không bias


def test_each_token_is_processed_independently():
    ffn = FeedForward(D_MODEL, FFN_HIDDEN)
    x = torch.randn(2, 5, D_MODEL)
    x_changed = x.clone()
    x_changed[:, 0] += 1.0

    torch.testing.assert_close(ffn(x_changed)[:, 1:], ffn(x)[:, 1:])