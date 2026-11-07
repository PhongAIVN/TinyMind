"""Chuyển file .jsonl thành file nhị phân token ids (.bin) + meta.json.

Mỗi dòng .jsonl là một văn bản: {"text": "..."}. Dùng jsonl thay vì txt
để văn bản có dòng trống bên trong (code, bài viết nhiều đoạn) không bị cắt.
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

DTYPE = np.uint16
MAX_VOCAB_FOR_DTYPE = int(np.iinfo(DTYPE).max) + 1
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
DOCS_PER_BATCH = 1000


def iter_documents(paths: list[Path]) -> Iterator[str]:
    """Đọc từng văn bản từ các file .jsonl; mỗi dòng là {"text": ...}. Bỏ qua dòng trống."""
    for path in paths:
        with open(path, encoding="utf-8") as jsonl_file:
            for line_number, line in enumerate(jsonl_file, start=1):
                if not line.strip():
                    continue
                try:
                    text = json.loads(line)["text"]
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(f"{path}:{line_number} không phải {{\"text\": ...}}") from error
                if text:
                    yield text

def encode_documents(tokenizer, documents: list[str], bos_id: int, eos_id: int) -> list[int]:
    """Tokenize một lô văn bản, mỗi văn bản bọc bos ... eos."""
    # với mỗi encoding trong tokenizer.encode_batch(documents):
    #             thêm bos_id, rồi encoding.ids, rồi eos_id vào list kết quả
    ids: list[int] = []
    for encoding in tokenizer.encode_batch(documents):
        ids.append(bos_id)
        ids.extend(encoding.ids)
        ids.append(eos_id)
    return ids

def write_bin(paths, tokenizer, out_bin: Path, docs_per_batch: int = DOCS_PER_BATCH) -> int:
    """Ghi token ids ra file nhị phân uint16. Trả về tổng số token đã ghi."""
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > MAX_VOCAB_FOR_DTYPE:
        raise ValueError(f"vocab_size={vocab_size} vượt {MAX_VOCAB_FOR_DTYPE}, uint16 không đủ")

    # lấy bos_id, eos_id bằng tokenizer.token_to_id(...); thiếu → raise ValueError
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if bos_id is None or eos_id is None:
        raise ValueError(f"tokenizer thiếu {BOS_TOKEN} hoặc {EOS_TOKEN}")

    out_bin = Path(out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    batch: list[str] = []
    with open(out_bin, "wb") as out_file:
        for document in iter_documents(paths):
            batch.append(document)
            if len(batch) >= docs_per_batch:
                total += _flush(out_file, tokenizer, batch, bos_id, eos_id)
                batch = []
        if batch:
            total += _flush(out_file, tokenizer, batch, bos_id, eos_id)
    return total


def _flush(out_file, tokenizer, batch: list[str], bos_id: int, eos_id: int) -> int:
    """Tokenize lô hiện tại rồi ghi xuống đĩa. Trả về số token vừa ghi."""
    ids = encode_documents(tokenizer, batch, bos_id, eos_id)
    np.asarray(ids, dtype=DTYPE).tofile(out_file)
    return len(ids)


def write_meta(out_meta, tokenizer_path, n_tokens: int, vocab_size: int, sources) -> None:
    """Ghi thông tin đi kèm file .bin để lúc train kiểm tra lại."""
    meta = {
        "dtype": np.dtype(DTYPE).name,
        "n_tokens": n_tokens,
        "vocab_size": vocab_size,
        "tokenizer": str(tokenizer_path),
        "sources": [str(path) for path in sources],
    }
    Path(out_meta).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def main() -> None:
    """Chạy từ dòng lệnh: đọc thư mục .jsonl, ghi <out_prefix>.bin và <out_prefix>_meta.json."""
    parser = argparse.ArgumentParser(description="jsonl -> token ids (.bin)")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--in-dir", required=True, help="thư mục chứa file .jsonl")
    parser.add_argument("--out-prefix", required=True, help="ví dụ dataset/pretrain/tokens/train")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    paths = sorted(Path(args.in_dir).glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"không có file .jsonl trong {args.in_dir}")

    out_bin = Path(f"{args.out_prefix}.bin")
    n_tokens = write_bin(paths, tokenizer, out_bin)
    write_meta(
        Path(f"{args.out_prefix}_meta.json"),
        Path(args.tokenizer),
        n_tokens,
        tokenizer.get_vocab_size(),
        paths,
    )
    print(f"Đã ghi {out_bin}: {n_tokens:,} token ({out_bin.stat().st_size / 1e6:.1f} MB)")



if __name__ == "__main__":
    main()