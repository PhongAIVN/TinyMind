"""Dữ liệu SFT: hội thoại -> (x, y), nhãn chỉ phủ phần trả lời của assistant.

Khuôn hội thoại (chat template):
    <|bos|><|im_start|>user\\n{câu hỏi}<|im_end|>\\n<|im_start|>assistant\\n{trả lời}<|im_end|>\\n

Nhãn được dựng theo từng tin nhắn (không dò chuỗi token): token của assistant
(nội dung + <|im_end|>) được học; mọi thứ khác = IGNORE_INDEX.
Học cả <|im_end|> để model biết lúc nào dừng.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from trainer.losses import IGNORE_INDEX

BOS_TOKEN = "<|bos|>"
PAD_TOKEN = "<|pad|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ROLES = ("system", "user", "assistant")
TRAINED_ROLE = "assistant"
TURN_SEPARATOR = "\n"


class ChatEncoder:
    """Mã hóa hội thoại theo khuôn ở đầu file. Dùng chung cho train và sinh chữ."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.bos_id = self._special_id(BOS_TOKEN)
        self.pad_id = self._special_id(PAD_TOKEN)
        self.im_start_id = self._special_id(IM_START)
        self.im_end_id = self._special_id(IM_END)
        self.separator = self._encode(TURN_SEPARATOR)
        self.headers = {role: [self.im_start_id] + self._encode(f"{role}\n") for role in ROLES}

    def _special_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"tokenizer thiếu token {token}")
        return token_id

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def encode_conversation(self, messages: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
        """Trả về (ids, labels) cùng độ dài; labels[i] = ids[i] nếu token đó được học."""
        if not messages:
            raise ValueError("hội thoại rỗng")
        ids, labels = [self.bos_id], [IGNORE_INDEX]
        for message in messages:
            role, content = message.get("role"), message.get("content")
            if role not in ROLES:
                raise ValueError(f"role={role!r} phải thuộc {ROLES}")
            if not isinstance(content, str):
                raise ValueError(f"content của {role} phải là chuỗi")
            header = self.headers[role]
            body = self._encode(content) + [self.im_end_id]
            ids += header + body + self.separator
            trained_body = body if role == TRAINED_ROLE else [IGNORE_INDEX] * len(body)
            labels += [IGNORE_INDEX] * len(header) + trained_body + [IGNORE_INDEX] * len(self.separator)
        return ids, labels

    def encode_prompt(self, user_text: str, system_text: str | None = None) -> list[int]:
        """Prompt để sinh chữ: hội thoại tới header của assistant, model viết tiếp phần trả lời."""
        messages = [{"role": "system", "content": system_text}] if system_text else []
        messages.append({"role": "user", "content": user_text})
        ids, _ = self.encode_conversation(messages)
        return ids + self.headers[TRAINED_ROLE]


def read_conversations(path: str | Path) -> list[list[dict[str, Any]]]:
    """Đọc .jsonl, mỗi dòng {"messages": [{"role": ..., "content": ...}, ...]}."""
    conversations = []
    with open(path, encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue
            messages = json.loads(line).get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"{path}:{line_number}: thiếu danh sách 'messages'")
            conversations.append(messages)
    return conversations


class SFTDataset(Dataset):
    """Mỗi phần tử: (x, y) shape (seq_len,), y = IGNORE_INDEX ở chỗ không học và ở phần đệm.

    Hội thoại dài hơn seq_len + 1 token bị cắt đuôi; hội thoại không còn token nào
    để học sau khi cắt thì bỏ (đếm trong self.dropped).
    """

    def __init__(self, path: str | Path, tokenizer, seq_len: int) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len={seq_len} phải > 0")
        self.seq_len = seq_len
        self.encoder = ChatEncoder(tokenizer)
        self.examples: list[tuple[np.ndarray, np.ndarray]] = []
        self.dropped = 0
        for messages in read_conversations(path):
            ids, labels = self.encoder.encode_conversation(messages)
            ids, labels = ids[: seq_len + 1], labels[: seq_len + 1]
            if all(label == IGNORE_INDEX for label in labels[1:]):
                self.dropped += 1
                continue
            self.examples.append((np.array(ids, dtype=np.int32), np.array(labels, dtype=np.int32)))
        if not self.examples:
            raise ValueError(f"{path}: không có hội thoại nào dùng được")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids, labels = self.examples[index]
        # shortcut: luôn đệm tới seq_len (lãng phí tính toán với câu ngắn); cần nhanh hơn thì gói nhiều hội thoại vào một chuỗi.
        padded_ids = np.full(self.seq_len + 1, self.encoder.pad_id, dtype=np.int64)
        padded_labels = np.full(self.seq_len + 1, IGNORE_INDEX, dtype=np.int64)
        padded_ids[: len(ids)] = ids
        padded_labels[: len(labels)] = labels
        x = torch.from_numpy(padded_ids[:-1])
        y = torch.from_numpy(padded_labels[1:])
        return x, y
