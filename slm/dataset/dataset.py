"""PyTorch dataset over a packed binary token file.

Purpose:
    Serve fixed-length ``(input, target)`` windows for next-token
    prediction. For a window of tokens t[0..L], the model input is
    t[0..L-1] and the target is t[1..L] — the target is the input
    shifted left by one (teacher forcing).

The .bin file is opened with ``np.memmap``, so the OS pages data in
lazily: a multi-GB corpus can be "loaded" without using RAM.

Shapes:
    input_ids:  (sequence_length,) int64
    target_ids: (sequence_length,) int64
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset):
    """Non-overlapping fixed-length windows over a packed token stream."""

    def __init__(self, data_path: str | Path, sequence_length: int) -> None:
        data_path = Path(data_path)
        if not data_path.is_file():
            raise FileNotFoundError(f"Token file not found: {data_path}")
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

        self.tokens = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.sequence_length = sequence_length

        # Each window needs sequence_length + 1 tokens (input + shifted target).
        self.num_windows = (len(self.tokens) - 1) // sequence_length
        if self.num_windows < 1:
            raise ValueError(
                f"Corpus has {len(self.tokens)} tokens — too few for even one "
                f"window of length {sequence_length}"
            )

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``index``-th (input, target) window.

        Windows are laid out back-to-back; window i covers tokens
        ``[i*L, i*L + L + 1)``. Consecutive windows share one boundary
        token (the last target of window i is the first input of i+1).
        """
        start = index * self.sequence_length
        chunk = self.tokens[start : start + self.sequence_length + 1]
        # Copy out of the memmap and widen to int64 (required by embedding
        # lookups and cross entropy).
        chunk = torch.from_numpy(chunk.astype(np.int64))
        return chunk[:-1], chunk[1:]
