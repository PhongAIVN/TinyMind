import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

BYTES_PER_MB = 1_000_000
N_RESERVED = 8
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
] + [f"<|reserved_{i}|>" for i in range(N_RESERVED)]


def iter_texts(files: list[str], max_mb_per_file: float | None = None) -> Iterator[str]:
    """Đọc chữ từ .txt (từng dòng) hoặc .jsonl (trường "text"), mỗi file tối đa max_mb_per_file MB.

    Giới hạn theo file để trộn mẫu cân bằng: nguồn lớn (web) không lấn át nguồn nhỏ (code).
    """
    budget = max_mb_per_file * BYTES_PER_MB if max_mb_per_file else None
    for file in files:
        used = 0
        with open(file, encoding="utf-8") as handle:
            for line in handle:
                text = json.loads(line)["text"] if file.endswith(".jsonl") and line.strip() else line
                if not text:
                    continue
                yield text
                used += len(text.encode("utf-8"))
                if budget is not None and used >= budget:
                    break


def build_tokenizer(files: list[str], vocab_size: int, max_mb_per_file: float | None = None) -> Tokenizer:
    """Train byte-level BPE tokenizer trên các file văn bản.

    Args:
        files: Đường dẫn các file .txt hoặc .jsonl.
        vocab_size: Kích thước vocab mong muốn (đã tính cả token đặc biệt).
        max_mb_per_file: Lấy tối đa bao nhiêu MB chữ mỗi file; None = lấy hết.

    Returns:
        Tokenizer đã train. Token đặc biệt có id 0..len(SPECIAL_TOKENS)-1.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    tokenizer.train_from_iterator(iter_texts(files, max_mb_per_file), trainer)

    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train byte-level BPE tokenizer"
    )

    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16000,
    )

    parser.add_argument(
        "--out",
        default="tokenizer/tokenizer.json",
    )

    parser.add_argument(
        "--max-mb-per-file",
        type=float,
        default=None,
        help="lấy tối đa bao nhiêu MB chữ mỗi file (trộn mẫu cân bằng)",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "dataset" / "pretrain" / "jsonl" / "train"
    print(f"data_dir: {data_dir}")
    if args.files is None:
        args.files = sorted(
            str(p)
            for p in data_dir.glob("*.jsonl")
        )

    if not args.files:
        raise FileNotFoundError(
            f"Không tìm thấy file .jsonl trong folder: {data_dir}"
        )

    print("Training files:")
    for file in args.files:
        print(f"  {file}")

    tokenizer = build_tokenizer(
        args.files,
        args.vocab_size,
        args.max_mb_per_file,
    )

    out_path = base_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(out_path))

    print(
        f"Đã lưu {out_path}, "
        f"vocab_size={tokenizer.get_vocab_size()}"
    )



if __name__ == "__main__":
    main()