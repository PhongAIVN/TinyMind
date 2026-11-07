from pathlib import Path

import pytest

from tokenizer.train_tokenizer import SPECIAL_TOKENS, build_tokenizer

SAMPLE_TEXT = (
    "Hà Nội là thủ đô của Việt Nam.\n\n"
    "Người Việt nói tiếng Việt. Tiếng Việt có dấu thanh.\n\n"
) * 50
VOCAB_SIZE = 400
REAL_TOKENIZER = Path("tokenizer/tokenizer.json")


@pytest.fixture
def small_tokenizer(tmp_path):
    """Train tokenizer tí hon trên file tạm. tmp_path là thư mục tạm pytest tự tạo và tự xoá."""
    data_file = tmp_path / "sample.txt"
    data_file.write_text(SAMPLE_TEXT, encoding="utf-8")
    return build_tokenizer([str(data_file)], VOCAB_SIZE)


def test_vocab_size_is_respected(small_tokenizer):
    assert small_tokenizer.get_vocab_size() <= VOCAB_SIZE


def test_special_tokens_get_first_ids(small_tokenizer):
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        assert small_tokenizer.token_to_id(token) == expected_id


@pytest.mark.parametrize("token", SPECIAL_TOKENS)
def test_special_token_is_not_split(small_tokenizer, token):
    ids = small_tokenizer.encode(f"xin chào {token} tạm biệt").ids

    assert small_tokenizer.token_to_id(token) in ids


@pytest.mark.parametrize("text", ["Hà Nội là thủ đô của Việt Nam.", "Chữ lạ chưa gặp: ỹ ữ 😀"])
def test_decode_encode_round_trip(small_tokenizer, text):
    ids = small_tokenizer.encode(text).ids

    assert small_tokenizer.decode(ids) == text


@pytest.mark.skipif(not REAL_TOKENIZER.exists(), reason="chưa train tokenizer thật")
def test_real_tokenizer_round_trip():
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(REAL_TOKENIZER))
    text = "Hà Nội là thủ đô của Việt Nam."

    assert tokenizer.decode(tokenizer.encode(text).ids) == text
    assert tokenizer.token_to_id("<think>") == SPECIAL_TOKENS.index("<think>")

def _write_jsonl(path, texts):
    import json

    path.write_text("".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts), encoding="utf-8")
    return str(path)


def test_iter_texts_reads_jsonl_documents_whole(tmp_path):
    from tokenizer.train_tokenizer import iter_texts

    code = "def f():\n    return 1\n\n\ndef g():\n    return 2"
    path = _write_jsonl(tmp_path / "code.jsonl", [code, "Hà Nội"])

    assert list(iter_texts([path])) == [code, "Hà Nội"]


def test_iter_texts_caps_each_file(tmp_path):
    from tokenizer.train_tokenizer import iter_texts

    doc = "a" * 1000
    big = _write_jsonl(tmp_path / "big.jsonl", [doc] * 50)
    small = _write_jsonl(tmp_path / "small.jsonl", ["b" * 10])

    texts = list(iter_texts([big, small], max_mb_per_file=0.003))

    assert texts.count(doc) == 3
    assert texts[-1] == "b" * 10


def test_build_tokenizer_from_jsonl_keeps_special_ids(tmp_path):
    path = _write_jsonl(tmp_path / "mix.jsonl", [SAMPLE_TEXT, "def f():\n    return 1\n"] * 5)

    tokenizer = build_tokenizer([path], VOCAB_SIZE)

    for expected_id, token in enumerate(SPECIAL_TOKENS):
        assert tokenizer.token_to_id(token) == expected_id
    code = "def f():\n    return 1\n"
    assert tokenizer.decode(tokenizer.encode(code).ids) == code
