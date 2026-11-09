"""Dữ liệu DPO: mỗi mẫu là một prompt và hai câu trả lời, chosen (tốt) và rejected (kém).

Mỗi dòng .jsonl: {"prompt": [{"role": "user", "content": ...}], "chosen": "...", "rejected": "..."}.
Hai câu dùng chung prompt; nhãn chỉ phủ phần trả lời (như SFT), để log-prob đo đúng câu trả lời.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from trainer.losses import IGNORE_INDEX
from trainer.sft_dataset import ChatEncoder, to_inputs_and_targets


class DPODataset(Dataset):
    """Mỗi phần tử: (x_chosen, y_chosen, x_rejected, y_rejected), mỗi tensor shape (seq_len,)."""

    def __init__(self, path: str | Path, tokenizer, seq_len: int) -> None:
        self.seq_len = seq_len
        self.encoder = ChatEncoder(tokenizer)
        self.pairs = []
        self.dropped = 0
        with open(path, encoding="utf-8") as jsonl_file:
            for line_number, line in enumerate(jsonl_file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not all(isinstance(record.get(key), str) for key in ("chosen", "rejected")):
                    raise ValueError(f"{path}:{line_number}: thiếu 'chosen' hoặc 'rejected'")
                pair = [self._encode(record["prompt"], record[key]) for key in ("chosen", "rejected")]
                if any(encoded is None for encoded in pair):
                    self.dropped += 1
                    continue
                self.pairs.append(pair)
        if not self.pairs:
            raise ValueError(f"{path}: không có cặp nào dùng được")

    def _encode(self, prompt: list[dict], answer: str):
        ids, labels = self.encoder.encode_conversation(prompt + [{"role": "assistant", "content": answer}])
        ids, labels = ids[: self.seq_len + 1], labels[: self.seq_len + 1]
        if all(label == IGNORE_INDEX for label in labels[1:]):
            return None  # câu trả lời bị cắt mất hết
        return ids, labels

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        (chosen_ids, chosen_labels), (rejected_ids, rejected_labels) = self.pairs[index]
        pad_id = self.encoder.pad_id
        return (
            *to_inputs_and_targets(chosen_ids, chosen_labels, self.seq_len, pad_id),
            *to_inputs_and_targets(rejected_ids, rejected_labels, self.seq_len, pad_id),
        )
