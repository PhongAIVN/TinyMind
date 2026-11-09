"""Gom dữ liệu thô (tải bằng `hf download`) thành file .jsonl cho pretrain.

Mỗi nguồn ghi ra 2 file: <out_dir>/train/<tên>.jsonl và <out_dir>/eval/<tên>.jsonl,
mỗi dòng là {"text": "..."}. Cứ EVAL_EVERY văn bản thì 1 văn bản vào eval.
Dữ liệu hỏi đáp được ghép thành văn bản thường theo QA_TEMPLATE.
"""

import argparse
import json
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

QA_TEMPLATE = "Câu hỏi: {question}\nTrả lời: {answer}"
EVAL_EVERY = 200
MIN_WEB_CHARS = 100
BYTES_PER_MB = 1_000_000
PARQUET_BATCH_ROWS = 1000

Record = dict[str, Any]


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def qa(question: Any, answer: Any) -> str | None:
    """Ghép một cặp hỏi đáp; thiếu một vế thì bỏ."""
    question, answer = _clean(question), _clean(answer)
    if not question or not answer:
        return None
    return QA_TEMPLATE.format(question=question, answer=answer)


def join_question(instruction: Any, extra_input: Any) -> str:
    """Câu hỏi kiểu alpaca = instruction + input (nếu có)."""
    return "\n".join(part for part in (_clean(instruction), _clean(extra_input)) if part)


def format_web(record: Record) -> str | None:
    text = _clean(record.get("text"))
    return text if len(text) >= MIN_WEB_CHARS else None


def format_wiki(record: Record) -> str | None:
    title, text = _clean(record.get("title")), _clean(record.get("text"))
    if not text:
        return None
    return f"{title}\n\n{text}" if title else text


def format_alpaca(record: Record) -> str | None:
    return qa(join_question(record.get("instruction"), record.get("input")), record.get("output"))


def format_alpaca_vi(record: Record) -> str | None:
    return qa(join_question(record.get("instruction_vi"), record.get("input_vi")), record.get("output_vi"))


def format_openorca(record: Record) -> str | None:
    # "instruction" của OpenOrca là system prompt chung chung -> bỏ.
    return qa(record.get("input"), record.get("output"))


def format_metamath(record: Record) -> str | None:
    return qa(record.get("query_vi"), record.get("response_vi"))


def format_viquad(record: Record) -> str | None:
    """Đọc hiểu: đoạn văn rồi hỏi đáp. Bỏ câu hỏi không có đáp án."""
    if record.get("is_impossible"):
        return None
    answers = (record.get("answers") or {}).get("text") or []
    pair = qa(record.get("question"), answers[0] if answers else None)
    context = _clean(record.get("context"))
    if pair is None or not context:
        return None
    return f"{context}\n\n{pair}"


def format_code_vi(record: Record) -> str | None:
    return qa(record.get("instruction"), record.get("answer"))


def format_python_code(record: Record) -> str | None:
    if record.get("program_lang") != "python":
        return None
    return _clean(record.get("text")) or None


@dataclass(frozen=True)
class Source:
    """Một nguồn dữ liệu: file nào (glob trong raw_dir), định dạng ra sao, lấy tối đa bao nhiêu."""

    name: str
    pattern: str
    formatter: Callable[[Record], str | None]
    columns: tuple[str, ...] | None = None  # chỉ dùng cho parquet: đọc đúng cột cần
    max_mb: float | None = None


SOURCES = (
    Source("fineweb_vi", "fineweb_vi/**/*.parquet", format_web, ("text",)),
    Source("wiki_vi", "wiki_vi/**/*.parquet", format_wiki, ("title", "text")),
    Source("openorca_vi", "openorca_vi/**/*.jsonl", format_openorca),
    Source("alpaca_gpt4_vi", "alpaca_gpt4_vi/**/*.json", format_alpaca_vi),
    Source("vi_alpaca", "vi_alpaca/**/*.parquet", format_alpaca),
    Source("metamath_vi", "metamath_vi/**/*.json", format_metamath),
    Source("viquad", "viquad/**/train-*.parquet", format_viquad),
    Source("code_vi", "code_vi/**/*.parquet", format_code_vi),
    Source("python_instruct", "python_instruct/**/*.parquet", format_alpaca),
    Source("opc_python", "opc_code/**/*.jsonl", format_python_code, max_mb=600),
)


def iter_records(path: Path, columns: tuple[str, ...] | None = None) -> Iterator[Record]:
    """Đọc từng bản ghi từ .parquet / .jsonl / .json (một list)."""
    if path.suffix == ".parquet":
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=PARQUET_BATCH_ROWS, columns=list(columns) if columns else None):
            yield from batch.to_pylist()
    elif path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as jsonl_file:
            for line in jsonl_file:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path}: cần một list bản ghi, nhận {type(data).__name__}")
        yield from data
    else:
        raise ValueError(f"{path}: không hỗ trợ đuôi {path.suffix}")


def iter_documents(source: Source, paths: list[Path]) -> Iterator[str]:
    for path in paths:
        for record in iter_records(path, source.columns):
            document = source.formatter(record)
            if document:
                yield document


def source_paths(source: Source, raw_dir: Path) -> list[Path]:
    """Các file của một nguồn, đã sắp xếp. Không có file nào thì báo FileNotFoundError."""
    # Bỏ thư mục ẩn: `hf download` ghi metadata .json vào <local-dir>/.cache/
    paths = sorted(p for p in raw_dir.glob(source.pattern) if not any(part.startswith(".") for part in p.relative_to(raw_dir).parts))
    if not paths:
        raise FileNotFoundError(f"{source.name}: không có file khớp {raw_dir / source.pattern}")
    return paths


def build_source(source: Source, raw_dir: Path, out_dir: Path, max_docs: int | None = None) -> dict[str, float]:
    """Ghi một nguồn ra train/eval .jsonl. Trả về số văn bản và dung lượng chữ (MB)."""
    paths = source_paths(source, raw_dir)

    budget_bytes = source.max_mb * BYTES_PER_MB if source.max_mb else math.inf
    doc_limit = max_docs if max_docs is not None else math.inf
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "eval").mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "eval": 0}
    written_bytes = 0
    with open(out_dir / "train" / f"{source.name}.jsonl", "w", encoding="utf-8") as train_file, \
         open(out_dir / "eval" / f"{source.name}.jsonl", "w", encoding="utf-8") as eval_file:
        for index, document in enumerate(iter_documents(source, paths)):
            if index >= doc_limit or written_bytes >= budget_bytes:
                break
            split = "eval" if index % EVAL_EVERY == EVAL_EVERY - 1 else "train"
            target = eval_file if split == "eval" else train_file
            target.write(json.dumps({"text": document}, ensure_ascii=False) + "\n")
            counts[split] += 1
            written_bytes += len(document.encode("utf-8"))

    return {**counts, "mb": written_bytes / BYTES_PER_MB}


def main() -> None:
    """Chạy: python -m scripts.build_pretrain_corpus [--only wiki_vi code_vi] [--max-docs 1000]"""
    names = [source.name for source in SOURCES]
    parser = argparse.ArgumentParser(description="dữ liệu thô -> pretrain .jsonl")
    parser.add_argument("--raw-dir", default="dataset/raw", help="thư mục chứa dữ liệu đã tải")
    parser.add_argument("--out-dir", default="dataset/pretrain/jsonl")
    parser.add_argument("--only", nargs="+", choices=names, help="chỉ chạy các nguồn này")
    parser.add_argument("--max-docs", type=int, default=None, help="giới hạn số văn bản mỗi nguồn (chạy thử)")
    parser.add_argument("--skip-missing", action="store_true", help="bỏ qua nguồn chưa tải thay vì báo lỗi")
    args = parser.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    selected = [source for source in SOURCES if not args.only or source.name in args.only]
    total_mb = 0.0
    for source in selected:
        try:
            stats = build_source(source, raw_dir, out_dir, args.max_docs)
        except FileNotFoundError as error:
            if not args.skip_missing:
                raise
            print(f"[bỏ qua] {error}")
            continue
        total_mb += stats["mb"]
        print(f"{source.name:16s} train {stats['train']:>9,} | eval {stats['eval']:>7,} | {stats['mb']:9.1f} MB")
    print(f"Tổng: {total_mb:,.1f} MB chữ -> {out_dir}")


if __name__ == "__main__":
    main()
