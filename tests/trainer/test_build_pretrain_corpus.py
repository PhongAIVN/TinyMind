import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.build_pretrain_corpus import (
    EVAL_EVERY,
    MIN_WEB_CHARS,
    Source,
    build_source,
    format_alpaca,
    format_alpaca_vi,
    format_openorca,
    format_python_code,
    format_viquad,
    format_web,
    format_wiki,
    iter_records,
    qa,
)
from scripts.prepare_pretrain import iter_documents


def read_jsonl(path):
    return [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_qa_template_and_missing_side():
    assert qa(" Thủ đô? ", "Hà Nội") == "Câu hỏi: Thủ đô?\nTrả lời: Hà Nội"
    assert qa("Thủ đô?", "  ") is None
    assert qa(None, "Hà Nội") is None


def test_format_alpaca_joins_instruction_and_input():
    record = {"instruction": "Dịch sang tiếng Anh.", "input": "Xin chào", "output": "Hello"}

    assert format_alpaca(record) == "Câu hỏi: Dịch sang tiếng Anh.\nXin chào\nTrả lời: Hello"
    assert format_alpaca({**record, "input": ""}) == "Câu hỏi: Dịch sang tiếng Anh.\nTrả lời: Hello"


def test_format_alpaca_vi_uses_vietnamese_fields():
    record = {"instruction_vi": "Kể tên 1 nước.", "input_vi": "", "output_vi": "Ý", "output_en": "Italy"}

    assert format_alpaca_vi(record) == "Câu hỏi: Kể tên 1 nước.\nTrả lời: Ý"


def test_format_openorca_drops_system_prompt():
    record = {"instruction": "Bạn là trợ lý.", "input": "1+1?", "output": "2"}

    assert format_openorca(record) == "Câu hỏi: 1+1?\nTrả lời: 2"


def test_format_web_drops_short_text():
    assert format_web({"text": "ngắn"}) is None
    assert format_web({"text": "a" * MIN_WEB_CHARS}) == "a" * MIN_WEB_CHARS


def test_format_wiki_puts_title_first():
    assert format_wiki({"title": "Hà Nội", "text": "Thủ đô."}) == "Hà Nội\n\nThủ đô."
    assert format_wiki({"title": "Rỗng", "text": ""}) is None


def test_format_viquad_skips_impossible_question():
    record = {"context": "Đoạn văn.", "question": "Hỏi?", "answers": {"text": ["Đáp"]}, "is_impossible": False}

    assert format_viquad(record) == "Đoạn văn.\n\nCâu hỏi: Hỏi?\nTrả lời: Đáp"
    assert format_viquad({**record, "is_impossible": True}) is None
    assert format_viquad({**record, "answers": {"text": []}}) is None


def test_format_python_code_keeps_only_python():
    code = "def f():\n    return 1\n\n\ndef g():\n    return 2"

    assert format_python_code({"program_lang": "python", "text": code}) == code
    assert format_python_code({"program_lang": "java", "text": "class A {}"}) is None


def test_iter_records_reads_all_formats(tmp_path):
    rows = [{"text": "a"}, {"text": "b"}]
    (tmp_path / "x.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    (tmp_path / "x.json").write_text(json.dumps(rows), encoding="utf-8")
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "x.parquet")

    for suffix in ("jsonl", "json", "parquet"):
        assert list(iter_records(tmp_path / f"x.{suffix}")) == rows


def test_iter_records_rejects_json_that_is_not_a_list(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"text": "a"}', encoding="utf-8")

    with pytest.raises(ValueError, match="list"):
        list(iter_records(path))


@pytest.fixture
def code_source(tmp_path):
    """Nguồn giả: 2 * EVAL_EVERY đoạn code Python có dòng trống bên trong, xen code Java."""
    raw = tmp_path / "raw" / "opc_code"
    raw.mkdir(parents=True)
    rows = []
    for i in range(2 * EVAL_EVERY):
        rows.append({"program_lang": "python", "text": f"def f{i}():\n    return {i}\n\n\nprint(f{i}())"})
        rows.append({"program_lang": "java", "text": "class A {}"})
    (raw / "1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return Source("opc_python", "opc_code/**/*.jsonl", format_python_code)


def test_build_source_splits_train_eval(code_source, tmp_path):
    stats = build_source(code_source, tmp_path / "raw", tmp_path / "out")

    assert stats["train"] == 2 * EVAL_EVERY - 2
    assert stats["eval"] == 2
    eval_docs = read_jsonl(tmp_path / "out" / "eval" / "opc_python.jsonl")
    assert eval_docs[0].startswith(f"def f{EVAL_EVERY - 1}()")


def test_build_source_output_survives_prepare_pretrain(code_source, tmp_path):
    """Code có dòng trống đi qua cả 2 bước vẫn nguyên một văn bản."""
    build_source(code_source, tmp_path / "raw", tmp_path / "out")

    docs = list(iter_documents([tmp_path / "out" / "train" / "opc_python.jsonl"]))

    assert docs[0] == "def f0():\n    return 0\n\n\nprint(f0())"


def test_build_source_respects_limits(code_source, tmp_path):
    assert build_source(code_source, tmp_path / "raw", tmp_path / "a", max_docs=10)["train"] == 10

    one_doc_mb = len("def f0():\n    return 0\n\n\nprint(f0())".encode()) / 1_000_000
    capped = Source(code_source.name, code_source.pattern, code_source.formatter, max_mb=one_doc_mb * 3)
    assert build_source(capped, tmp_path / "raw", tmp_path / "b")["train"] == 3


def test_build_source_ignores_hf_cache_metadata(code_source, tmp_path):
    """`hf download` để lại .cache/huggingface/*.json - không được đọc như dữ liệu."""
    cache = tmp_path / "raw" / "opc_code" / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "meta.jsonl").write_text('{"program_lang": "python", "text": "RÁC"}\n', encoding="utf-8")

    build_source(code_source, tmp_path / "raw", tmp_path / "out")

    assert "RÁC" not in (tmp_path / "out" / "train" / "opc_python.jsonl").read_text(encoding="utf-8")


def test_build_source_missing_files_raises(tmp_path):
    source = Source("wiki_vi", "wiki_vi/**/*.parquet", format_wiki)

    with pytest.raises(FileNotFoundError, match="wiki_vi"):
        build_source(source, tmp_path, tmp_path / "out")
