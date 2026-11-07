"""Dataset đọc file .bin token ids, cắt thành các cửa sổ (seq_len + 1)."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DTYPE = np.uint16


class PretrainDataset(Dataset):
    """Cắt dải token thành các cửa sổ không chồng lấn.

    Mục đích:
        Cấp (x, y) cho pretrain, trong đó y là x dịch 1 vị trí.
    Shape:
        x: (seq_len,) long, y: (seq_len,) long
    Độ phức tạp:
        O(seq_len) mỗi mẫu. File đọc bằng memmap nên không nạp hết vào RAM.
    Lưu ý:
        Token được nối gói (packing): một cửa sổ có thể chứa nhiều văn bản.
    """

    def __init__(self, bin_path: str | Path, seq_len: int) -> None:
        """
        Args:
            bin_path: File .bin chứa token ids kiểu uint16.
            seq_len: Số token mỗi mẫu (T). Mỗi cửa sổ đọc T + 1 token.
        """
        if seq_len < 1:
            raise ValueError(f"seq_len={seq_len} phải >= 1")

        self.seq_len = seq_len
        self.tokens = np.memmap(bin_path, dtype=DTYPE, mode="r")
        if self.tokens.size < seq_len + 1:
            raise ValueError(f"file chỉ có {self.tokens.size} token, cần ít nhất {seq_len + 1}")

    def __len__(self) -> int:
        return (self.tokens.size - 1) // self.seq_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"index={index} ngoài khoảng [0, {len(self) - 1}]")

        start = index * self.seq_len
        window = self.tokens[start: start + self.seq_len + 1].astype(np.int64)
        chunk = torch.from_numpy(window)
        return chunk[:-1], chunk[1:]