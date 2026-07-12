"""Train a SentencePiece tokenizer on raw text files.

Usage:
    python -m slm.tokenizer.train_tokenizer \\
        --input data/corpus_vi.txt data/corpus_en.txt \\
        --output-prefix tokenizer/slm \\
        --vocab-size 32000 \\
        --model-type unigram

Notes for Vietnamese + English:
    - ``character_coverage=0.9995`` keeps all Vietnamese diacritics while
      dropping only the rarest symbols (emoji, exotic scripts).
    - ``byte_fallback=True`` guarantees no <unk> ever appears: any
      character outside the vocab decomposes into byte tokens.
"""

import argparse
from pathlib import Path

import sentencepiece as spm

# Fixed special token ids, shared with Tokenizer in tokenizer.py.
UNK_ID = 0
BOS_ID = 1
EOS_ID = 2
PAD_ID = 3


def train_tokenizer(
    input_files: list[str],
    output_prefix: str,
    vocab_size: int = 32000,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
) -> str:
    """Train a SentencePiece model and return the path to the .model file.

    Args:
        input_files: Plain-text training files, one sentence/paragraph per line.
        output_prefix: Output path prefix; produces ``<prefix>.model`` and
            ``<prefix>.vocab``.
        vocab_size: Total vocabulary size including special tokens.
        model_type: ``"unigram"`` or ``"bpe"``.
        character_coverage: Fraction of corpus characters guaranteed to get
            their own token.

    Returns:
        Path to the trained ``.model`` file.
    """
    if model_type not in ("unigram", "bpe"):
        raise ValueError(f"model_type must be 'unigram' or 'bpe', got {model_type!r}")
    for input_file in input_files:
        if not Path(input_file).is_file():
            raise FileNotFoundError(f"Training file not found: {input_file}")

    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=",".join(input_files),
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
        byte_fallback=True,
        normalization_rule_name="nmt_nfkc",
        remove_extra_whitespaces=False,
        # Split digits into single tokens — helps arithmetic generalization.
        split_digits=True,
        # Allow whitespace-only pieces so code/indentation survives.
        allow_whitespace_only_pieces=True,
    )
    return f"{output_prefix}.model"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer")
    parser.add_argument("--input", nargs="+", required=True, help="Text files to train on")
    parser.add_argument("--output-prefix", required=True, help="Output path prefix")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--model-type", choices=["unigram", "bpe"], default="unigram")
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    args = parser.parse_args()

    model_path = train_tokenizer(
        input_files=args.input,
        output_prefix=args.output_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
    )
    print(f"Tokenizer saved to {model_path}")


if __name__ == "__main__":
    main()
