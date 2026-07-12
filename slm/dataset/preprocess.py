"""Preprocess raw text into a packed binary token file.

Pipeline:
    raw .txt files → clean → tokenize → one long uint16 stream → .bin

Every document is tokenized, terminated with <eos>, and concatenated
into a single flat array saved with numpy. Training then slices fixed
windows out of this stream ("packing"), so no padding is ever needed
and every batch token contributes to the loss.

uint16 is enough because vocab_size = 32000 < 65536; it halves disk
and memory versus int32.

Usage:
    python -m slm.dataset.preprocess \\
        --input data/raw/*.txt \\
        --tokenizer tokenizer/slm.model \\
        --output data/train.bin
"""

import argparse
import unicodedata
from pathlib import Path

import numpy as np
from tqdm import tqdm

from slm.tokenizer.tokenizer import Tokenizer

# Vocab must fit in uint16.
MAX_UINT16 = 65535


def clean_text(text: str) -> str:
    """Normalize a document before tokenization.

    - NFC unicode normalization: Vietnamese diacritics have two byte-level
      encodings (composed vs decomposed); NFC unifies them so the same
      word always maps to the same tokens.
    - Normalize Windows line endings.
    - Strip trailing whitespace per line and collapse 3+ blank lines.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n")
    lines = [line.rstrip() for line in text.split("\n")]

    cleaned_lines: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run > 2:
                continue
        else:
            blank_run = 0
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def preprocess_files(
    input_files: list[str | Path],
    tokenizer: Tokenizer,
    output_path: str | Path,
) -> int:
    """Tokenize input files into one packed uint16 binary file.

    Each input file is treated as one document: cleaned, tokenized with
    a trailing <eos>, and appended to the stream.

    Returns:
        Total number of tokens written.
    """
    if tokenizer.vocab_size > MAX_UINT16 + 1:
        raise ValueError(
            f"vocab_size {tokenizer.vocab_size} does not fit in uint16; "
            "switch the output dtype to uint32"
        )

    all_token_ids: list[int] = []
    for input_file in tqdm(input_files, desc="Tokenizing files"):
        raw_text = Path(input_file).read_text(encoding="utf-8")
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue
        all_token_ids.extend(tokenizer.encode(cleaned, add_eos=True))

    if not all_token_ids:
        raise ValueError("No tokens produced — are the input files empty?")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_array = np.array(all_token_ids, dtype=np.uint16)
    token_array.tofile(output_path)

    return len(token_array)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Preprocess text into packed tokens")
    parser.add_argument("--input", nargs="+", required=True, help="Input .txt files")
    parser.add_argument("--tokenizer", required=True, help="Path to .model tokenizer file")
    parser.add_argument("--output", required=True, help="Output .bin path")
    args = parser.parse_args()

    tokenizer = Tokenizer(args.tokenizer)
    num_tokens = preprocess_files(args.input, tokenizer, args.output)
    print(f"Wrote {num_tokens:,} tokens to {args.output}")


if __name__ == "__main__":
    main()
