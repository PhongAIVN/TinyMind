"""Tests for the packed token dataset and text cleaning."""

from pathlib import Path

import numpy as np
import pytest
import torch

from slm.dataset.dataset import PackedTokenDataset
from slm.dataset.preprocess import clean_text


def _write_tokens(path: Path, tokens: list[int]) -> None:
    np.array(tokens, dtype=np.uint16).tofile(path)


def test_window_count(tmp_path: Path) -> None:
    # 21 tokens, window length 5 → (21 - 1) // 5 = 4 windows
    data_file = tmp_path / "tokens.bin"
    _write_tokens(data_file, list(range(21)))
    dataset = PackedTokenDataset(data_file, sequence_length=5)
    assert len(dataset) == 4


def test_target_is_input_shifted_by_one(tmp_path: Path) -> None:
    data_file = tmp_path / "tokens.bin"
    _write_tokens(data_file, list(range(11)))
    dataset = PackedTokenDataset(data_file, sequence_length=5)

    inputs, targets = dataset[0]
    torch.testing.assert_close(inputs, torch.arange(0, 5))
    torch.testing.assert_close(targets, torch.arange(1, 6))

    inputs, targets = dataset[1]
    torch.testing.assert_close(inputs, torch.arange(5, 10))
    torch.testing.assert_close(targets, torch.arange(6, 11))


def test_dtype_is_int64(tmp_path: Path) -> None:
    data_file = tmp_path / "tokens.bin"
    _write_tokens(data_file, list(range(10)))
    inputs, targets = PackedTokenDataset(data_file, sequence_length=4)[0]
    assert inputs.dtype == torch.int64
    assert targets.dtype == torch.int64


def test_too_small_corpus_raises(tmp_path: Path) -> None:
    data_file = tmp_path / "tokens.bin"
    _write_tokens(data_file, [1, 2, 3])
    with pytest.raises(ValueError, match="too few"):
        PackedTokenDataset(data_file, sequence_length=10)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PackedTokenDataset(tmp_path / "missing.bin", sequence_length=4)


def test_clean_text_normalizes_vietnamese_nfc() -> None:
    # "ế" written as decomposed e + circumflex + acute must become one NFC char.
    decomposed = "tiếng Việt"
    cleaned = clean_text(decomposed)
    assert cleaned == "tiếng Việt"  # "tiếng Việt" in NFC


def test_clean_text_collapses_blank_lines_and_crlf() -> None:
    raw = "a\r\n\r\n\r\n\r\n\r\nb   \r\n"
    cleaned = clean_text(raw)
    assert cleaned == "a\n\n\nb"  # at most 2 blank lines survive, trailing spaces gone
