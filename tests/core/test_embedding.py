import pytest
import torch

from core.embedding import TokenEmbedding

VOCAB = 100
D_MODEL = 16
INIT_STD = 0.02


def test_output_shape():
    emb = TokenEmbedding(VOCAB, D_MODEL, INIT_STD)
    ids = torch.randint(0, VOCAB, (2, 5))

    assert emb(ids).shape == (2, 5, D_MODEL)


def test_parameter_count():
    emb = TokenEmbedding(VOCAB, D_MODEL, INIT_STD)

    assert sum(p.numel() for p in emb.parameters()) == VOCAB * D_MODEL


def test_output_is_row_of_weight():
    emb = TokenEmbedding(VOCAB, D_MODEL, INIT_STD)
    ids = torch.tensor([[5, 42]])

    out = emb(ids)

    torch.testing.assert_close(out[0, 0], emb.weight[5])
    torch.testing.assert_close(out[0, 1], emb.weight[42])


def test_same_token_gives_same_vector():
    emb = TokenEmbedding(VOCAB, D_MODEL, INIT_STD)
    out = emb(torch.tensor([[7, 3, 7]]))

    torch.testing.assert_close(out[0, 0], out[0, 2])


def test_init_std_is_small():
    emb = TokenEmbedding(1000, 64, INIT_STD)

    assert abs(emb.weight.std().item() - INIT_STD) < 0.002


def test_gradient_only_reaches_used_rows():
    emb = TokenEmbedding(VOCAB, D_MODEL, INIT_STD)
    emb(torch.tensor([[1, 2]])).sum().backward()

    used = emb.weight.grad[[1, 2]]
    unused = emb.weight.grad[3:]

    assert used.abs().sum() > 0
    assert unused.abs().sum() == 0


@pytest.mark.parametrize("bad_ids", [torch.tensor([[VOCAB]]), torch.tensor([[-1]])])
def test_out_of_range_ids_raise(bad_ids):
    emb = TokenEmbedding(VOCAB, D_MODEL, INIT_STD)

    with pytest.raises(ValueError, match="input_ids"):
        emb(bad_ids)


def test_float_ids_raise():
    emb = TokenEmbedding(VOCAB, D_MODEL, INIT_STD)

    with pytest.raises(ValueError, match="long"):
        emb(torch.tensor([[1.0, 2.0]]))