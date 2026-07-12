"""Thin, explicit wrapper around a trained SentencePiece model.

Purpose:
    Give the rest of the codebase one small, typed interface for
    text ↔ token-id conversion, with special token handling in one place.

Special tokens (fixed at training time, see train_tokenizer.py):
    unk=0, bos=1, eos=2, pad=3
"""

from pathlib import Path

import sentencepiece as spm


class Tokenizer:
    """Encode/decode text with a trained SentencePiece model."""

    def __init__(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")

        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))
        self.unk_id: int = self.processor.unk_id()
        self.bos_id: int = self.processor.bos_id()
        self.eos_id: int = self.processor.eos_id()
        self.pad_id: int = self.processor.pad_id()

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return self.processor.vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Convert text to a list of token ids.

        Args:
            text: Input string.
            add_bos: Prepend the beginning-of-sequence token.
            add_eos: Append the end-of-sequence token.
        """
        token_ids: list[int] = self.processor.encode(text)
        if add_bos:
            token_ids = [self.bos_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_id]
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Convert token ids back to text (special tokens are dropped)."""
        return self.processor.decode(token_ids)

    def id_to_piece(self, token_id: int) -> str:
        """Return the raw subword piece for one id (for debugging)."""
        return self.processor.id_to_piece(token_id)
