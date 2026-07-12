"""Tests for tokenizer training and the Tokenizer wrapper.

Trains a tiny real SentencePiece model on a synthetic bilingual corpus
(fast: < 1 s) so the whole pipeline is exercised end to end.
"""

from pathlib import Path

import pytest

from slm.tokenizer.tokenizer import Tokenizer
from slm.tokenizer.train_tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID, train_tokenizer

SAMPLE_SENTENCES = [
    "Xin chào thế giới, hôm nay trời đẹp quá.",
    "Tiếng Việt là ngôn ngữ chính thức của Việt Nam.",
    "Hello world, the weather is nice today.",
    "English and Vietnamese share this small corpus.",
    "Học máy và trí tuệ nhân tạo đang phát triển nhanh.",
    "Machine learning models need lots of text data.",
] * 30  # repeat so sentencepiece has enough tokens to fit a vocab


@pytest.fixture(scope="module")
def trained_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Tokenizer:
    tmp_path = tmp_path_factory.mktemp("tokenizer")
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text("\n".join(SAMPLE_SENTENCES), encoding="utf-8")

    model_path = train_tokenizer(
        input_files=[str(corpus_file)],
        output_prefix=str(tmp_path / "test_spm"),
        vocab_size=330,
        model_type="unigram",
        character_coverage=1.0,
    )
    return Tokenizer(model_path)


def test_special_token_ids(trained_tokenizer: Tokenizer) -> None:
    assert trained_tokenizer.unk_id == UNK_ID
    assert trained_tokenizer.bos_id == BOS_ID
    assert trained_tokenizer.eos_id == EOS_ID
    assert trained_tokenizer.pad_id == PAD_ID


def test_encode_decode_roundtrip_vietnamese(trained_tokenizer: Tokenizer) -> None:
    text = "Xin chào thế giới"
    assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text


def test_encode_decode_roundtrip_english(trained_tokenizer: Tokenizer) -> None:
    text = "Hello world today"
    assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text


def test_bos_eos_flags(trained_tokenizer: Tokenizer) -> None:
    plain = trained_tokenizer.encode("Xin chào")
    wrapped = trained_tokenizer.encode("Xin chào", add_bos=True, add_eos=True)
    assert wrapped == [trained_tokenizer.bos_id] + plain + [trained_tokenizer.eos_id]


def test_byte_fallback_handles_unseen_characters(trained_tokenizer: Tokenizer) -> None:
    """Characters absent from training (e.g. CJK) must roundtrip via bytes."""
    text = "日本語"
    decoded = trained_tokenizer.decode(trained_tokenizer.encode(text))
    assert decoded == text


def test_vocab_size(trained_tokenizer: Tokenizer) -> None:
    assert trained_tokenizer.vocab_size == 330


def test_missing_model_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Tokenizer(tmp_path / "nope.model")


def test_invalid_model_type_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_type"):
        train_tokenizer(["x.txt"], str(tmp_path / "y"), model_type="wordpiece")
