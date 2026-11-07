import numpy as np
import pytest
import torch

from trainer.pretrain_dataset import DTYPE, PretrainDataset

N_TOKENS = 101
SEQ_LEN = 10


@pytest.fixture
def bin_path(tmp_path):
    """File .bin chứa 0, 1, 2, ... để dễ kiểm tra vị trí."""
    path = tmp_path / "tokens.bin"
    np.arange(N_TOKENS, dtype=DTYPE).tofile(path)
    return path


def test_length_counts_full_windows(bin_path):
    dataset = PretrainDataset(bin_path, SEQ_LEN)

    assert len(dataset) == (N_TOKENS - 1) // SEQ_LEN


def test_x_and_y_are_shifted_by_one(bin_path):
    dataset = PretrainDataset(bin_path, SEQ_LEN)

    x, y = dataset[0]

    torch.testing.assert_close(y[:-1], x[1:])
    assert x.tolist() == list(range(SEQ_LEN))
    assert y.tolist() == list(range(1, SEQ_LEN + 1))


def test_windows_do_not_overlap(bin_path):
    dataset = PretrainDataset(bin_path, SEQ_LEN)

    x0, _ = dataset[0]
    x1, _ = dataset[1]

    assert x1[0].item() == x0[-1].item() + 1


def test_last_window_stays_inside_file(bin_path):
    dataset = PretrainDataset(bin_path, SEQ_LEN)

    _, y = dataset[len(dataset) - 1]

    assert y[-1].item() <= N_TOKENS - 1


def test_output_is_long_tensor(bin_path):
    dataset = PretrainDataset(bin_path, SEQ_LEN)

    x, y = dataset[0]

    assert x.dtype == torch.long and y.dtype == torch.long
    assert x.shape == (SEQ_LEN,) and y.shape == (SEQ_LEN,)


def test_same_index_gives_same_sample(bin_path):
    dataset = PretrainDataset(bin_path, SEQ_LEN)

    torch.testing.assert_close(dataset[3][0], dataset[3][0])


def test_negative_index_counts_from_end(bin_path):
    dataset = PretrainDataset(bin_path, SEQ_LEN)

    torch.testing.assert_close(dataset[-1][0], dataset[len(dataset) - 1][0])


@pytest.mark.parametrize("index", [-100, 1000])
def test_index_out_of_range_raises(bin_path, index):
    dataset = PretrainDataset(bin_path, SEQ_LEN)

    with pytest.raises(IndexError):
        dataset[index]


def test_file_shorter_than_window_raises(tmp_path):
    path = tmp_path / "tiny.bin"
    np.arange(5, dtype=DTYPE).tofile(path)

    with pytest.raises(ValueError, match="token"):
        PretrainDataset(path, SEQ_LEN)


def test_invalid_seq_len_raises(bin_path):
    with pytest.raises(ValueError, match="seq_len"):
        PretrainDataset(bin_path, 0)