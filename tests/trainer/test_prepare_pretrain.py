import json

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from scripts.prepare_pretrain import (
    BOS_TOKEN,
    DTYPE,
    EOS_TOKEN,
    encode_documents,
    iter_documents,
    write_bin,
    write_meta,
)
from tokenizer.train_tokenizer import build_tokenizer

DOCUMENTS = [
    "Hà Nội là thủ đô.",
    "Người Việt nói tiếng Việt.\nDòng thứ hai.",
    "def f():\n    return 1\n\n\ndef g():\n    return 2",
]
SAMPLE_TEXT = "\n\n".join(DOCUMENTS) + "\n"


def write_jsonl(path, texts):
    path.write_text("".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts), encoding="utf-8")
    return path


@pytest.fixture
def tokenizer(tmp_path):
    """Tokenizer tí hon train trên chính văn bản mẫu."""
    data_file = tmp_path / "train.txt"
    data_file.write_text(SAMPLE_TEXT * 30, encoding="utf-8")
    return build_tokenizer([str(data_file)], vocab_size=400)


@pytest.fixture
def text_file(tmp_path):
    return write_jsonl(tmp_path / "sample.jsonl", DOCUMENTS)


def test_iter_documents_reads_one_document_per_line(text_file):
    assert list(iter_documents([text_file])) == DOCUMENTS


def test_iter_documents_keeps_blank_lines_inside_document(text_file):
    """Code có dòng trống giữa các hàm vẫn là MỘT văn bản."""
    code = list(iter_documents([text_file]))[2]

    assert "\n\n\n" in code


def test_iter_documents_reads_multiple_files(text_file):
    assert list(iter_documents([text_file, text_file])) == DOCUMENTS * 2


def test_iter_documents_skips_empty_lines_and_empty_text(tmp_path):
    path = tmp_path / "gaps.jsonl"
    path.write_text('{"text": "a"}\n\n{"text": ""}\n{"text": "b"}\n', encoding="utf-8")

    assert list(iter_documents([path])) == ["a", "b"]


def test_iter_documents_rejects_bad_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text": "a"}\nkhông phải json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="bad.jsonl:2"):
        list(iter_documents([path]))


def test_encode_documents_wraps_each_document(tokenizer):
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    ids = encode_documents(tokenizer, DOCUMENTS[:1], bos_id, eos_id)

    assert ids[0] == bos_id
    assert ids[-1] == eos_id
    assert ids[1:-1] == tokenizer.encode(DOCUMENTS[0]).ids


def test_write_bin_round_trip(tokenizer, text_file, tmp_path):
    out_bin = tmp_path / "tokens" / "train.bin"

    n_tokens = write_bin([text_file], tokenizer, out_bin)
    tokens = np.fromfile(out_bin, dtype=DTYPE)

    assert tokens.size == n_tokens
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    assert int(tokens[0]) == bos_id
    assert int(tokens[-1]) == eos_id
    assert list(tokens).count(bos_id) == len(DOCUMENTS)

    # Cắt theo bos/eos rồi decode lại phải ra đúng văn bản gốc
    decoded = []
    current: list[int] = []
    for token_id in tokens.tolist():
        if token_id == bos_id:
            current = []
        elif token_id == eos_id:
            decoded.append(tokenizer.decode(current))
        else:
            current.append(token_id)
    assert decoded == DOCUMENTS


def test_write_bin_respects_batching(tokenizer, text_file, tmp_path):
    """Chia lô nhỏ hay lớn đều cho cùng kết quả."""
    one_batch = tmp_path / "a.bin"
    many_batches = tmp_path / "b.bin"

    write_bin([text_file], tokenizer, one_batch, docs_per_batch=100)
    write_bin([text_file], tokenizer, many_batches, docs_per_batch=1)

    assert one_batch.read_bytes() == many_batches.read_bytes()


def test_write_meta(tokenizer, text_file, tmp_path):
    out_meta = tmp_path / "train_meta.json"

    write_meta(out_meta, tmp_path / "tokenizer.json", 1234, tokenizer.get_vocab_size(), [text_file])
    meta = json.loads(out_meta.read_text(encoding="utf-8"))

    assert meta["n_tokens"] == 1234
    assert meta["dtype"] == "uint16"
    assert meta["vocab_size"] == tokenizer.get_vocab_size()


def test_tokenizer_without_special_tokens_raises(tmp_path, text_file):
    plain = Tokenizer(models.BPE())
    plain.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    plain.train([str(text_file)], trainers.BpeTrainer(vocab_size=300))

    with pytest.raises(ValueError, match="bos"):
        write_bin([text_file], plain, tmp_path / "out.bin")