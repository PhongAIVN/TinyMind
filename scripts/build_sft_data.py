"""Gom dữ liệu hỏi đáp thô (cùng dataset/raw với pretrain) thành hội thoại cho SFT.

Ra 2 file: <out_dir>/train.jsonl và <out_dir>/eval.jsonl,
mỗi dòng {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}.
Cứ EVAL_EVERY hội thoại thì 1 vào eval; eval được xáo để 20 batch đầu có đủ mọi nguồn.
--extra: thêm file .jsonl đã đúng dạng {"messages": [...]} (vd Vietverse-SFT-1K-Gold.jsonl).
"""

import argparse
import json
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.build_pretrain_corpus import EVAL_EVERY, SOURCES, Record, _clean, iter_records, join_question, source_paths
from trainer.sft_dataset import read_conversations

DEFAULT_MAX_PER_SOURCE = 20_000
SHUFFLE_SEED = 0

Pair = tuple[str, str]
Messages = list[dict[str, str]]


def pair(question, answer) -> Pair | None:
    question, answer = _clean(question), _clean(answer)
    return (question, answer) if question and answer else None


def alpaca_pair(record: Record) -> Pair | None:
    return pair(join_question(record.get("instruction"), record.get("input")), record.get("output"))


def alpaca_vi_pair(record: Record) -> Pair | None:
    return pair(join_question(record.get("instruction_vi"), record.get("input_vi")), record.get("output_vi"))


def openorca_pair(record: Record) -> Pair | None:
    return pair(record.get("input"), record.get("output"))


def metamath_pair(record: Record) -> Pair | None:
    return pair(record.get("query_vi"), record.get("response_vi"))


def viquad_pair(record: Record) -> Pair | None:
    """Đọc hiểu: user đưa đoạn văn + câu hỏi. Bỏ câu không có đáp án."""
    if record.get("is_impossible"):
        return None
    answers = (record.get("answers") or {}).get("text") or []
    context = _clean(record.get("context"))
    question_answer = pair(record.get("question"), answers[0] if answers else None)
    if question_answer is None or not context:
        return None
    return f"{context}\n\n{question_answer[0]}", question_answer[1]


def code_vi_pair(record: Record) -> Pair | None:
    return pair(record.get("instruction"), record.get("answer"))


def to_messages(question_answer: Pair | None) -> Messages | None:
    if question_answer is None:
        return None
    question, answer = question_answer
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


SHAREGPT_ROLES = {"system": "system", "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}


def clean_turns(turns: list[tuple[Any, Any]]) -> Messages | None:
    """(role, content) -> messages. Bỏ cả hội thoại nếu có role lạ, lượt rỗng hoặc không có assistant."""
    messages = []
    for role, content in turns:
        role, content = SHAREGPT_ROLES.get(role), _clean(content)
        if role is None or not content:
            return None
        messages.append({"role": role, "content": content})
    return messages if any(m["role"] == "assistant" for m in messages) else None


def sharegpt_messages(record: Record) -> Messages | None:
    """{"conversations": [{"from": "human"|"gpt", "value": ...}]}"""
    turns = record.get("conversations") or []
    return clean_turns([(turn.get("from"), turn.get("value")) for turn in turns])


def chat_messages(record: Record) -> Messages | None:
    """{"messages": [{"role": ..., "content": ...}]} (kiểu UltraChat)."""
    turns = record.get("messages")
    if turns is None or not hasattr(turns, "__iter__"):
        return None
    return clean_turns([(turn.get("role"), turn.get("content")) for turn in turns])


@dataclass(frozen=True)
class SftSource:
    """Một nguồn SFT: glob trong raw_dir và hàm đổi bản ghi -> messages (None = bỏ)."""

    name: str
    pattern: str
    convert: Callable[[Record], Messages | None]


def _from_pretrain(name: str, extract_pair: Callable[[Record], Pair | None]) -> SftSource:
    """Nguồn hỏi đáp đã tải cho pretrain: dùng lại đường dẫn của nó."""
    pattern = next(source.pattern for source in SOURCES if source.name == name)
    return SftSource(name, pattern, lambda record: to_messages(extract_pair(record)))


SFT_SOURCES = (
    # Hỏi đáp một lượt (web, wiki, opc không phải hỏi đáp nên bỏ).
    _from_pretrain("openorca_vi", openorca_pair),
    _from_pretrain("alpaca_gpt4_vi", alpaca_vi_pair),
    _from_pretrain("vi_alpaca", alpaca_pair),
    _from_pretrain("metamath_vi", metamath_pair),
    _from_pretrain("viquad", viquad_pair),
    _from_pretrain("code_vi", code_vi_pair),
    _from_pretrain("python_instruct", alpaca_pair),
    # Hội thoại nhiều lượt, chỉ dùng cho SFT.
    SftSource("chat_alpaca_vi", "chat_alpaca_vi/**/*.json", sharegpt_messages),
    SftSource("ecommerce_chat_vi", "ecommerce_chat_vi/**/*.json", sharegpt_messages),
    SftSource("ultrachat_vi", "ultrachat_vi/**/*.parquet", chat_messages),
)
SFT_SOURCE_NAMES = [source.name for source in SFT_SOURCES]


def iter_source_conversations(source: SftSource, raw_dir: Path, max_per_source: int) -> Iterator[Messages]:
    count = 0
    for path in source_paths(source, raw_dir):
        for record in iter_records(path):
            messages = source.convert(record)
            if messages is None:
                continue
            yield messages
            count += 1
            if count >= max_per_source:
                return


def build_sft_data(
    raw_dir: Path,
    out_dir: Path,
    names: list[str],
    max_per_source: int,
    extra_files: list[Path] = (),
    skip_missing: bool = False,
) -> dict[str, int]:
    """Ghi train.jsonl / eval.jsonl. Trả về số hội thoại mỗi tập."""
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: list[tuple[str, Iterator]] = []
    for source in (source for source in SFT_SOURCES if source.name in names):
        try:
            source_paths(source, raw_dir)
        except FileNotFoundError as error:
            if not skip_missing:
                raise
            print(f"[bỏ qua] {error}")
            continue
        groups.append((source.name, iter_source_conversations(source, raw_dir, max_per_source)))
    groups += [(path.name, iter(read_conversations(path))) for path in extra_files]

    eval_conversations = []
    train_count = 0
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as train_file:
        for name, conversations in groups:
            per_source = 0
            for index, messages in enumerate(conversations):
                if index % EVAL_EVERY == EVAL_EVERY - 1:
                    eval_conversations.append(messages)
                else:
                    train_file.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                    train_count += 1
                per_source += 1
            print(f"{name:32s} {per_source:>8,} hội thoại")

    random.Random(SHUFFLE_SEED).shuffle(eval_conversations)
    with open(out_dir / "eval.jsonl", "w", encoding="utf-8") as eval_file:
        for messages in eval_conversations:
            eval_file.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    return {"train": train_count, "eval": len(eval_conversations)}


def main() -> None:
    """python -m scripts.build_sft_data [--max-per-source 20000] [--extra dataset/SFT/Vietverse-SFT-1K-Gold.jsonl]"""
    parser = argparse.ArgumentParser(description="dữ liệu hỏi đáp thô -> hội thoại SFT .jsonl")
    parser.add_argument("--raw-dir", default="dataset/raw")
    parser.add_argument("--out-dir", default="dataset/sft")
    parser.add_argument("--only", nargs="+", choices=SFT_SOURCE_NAMES, default=SFT_SOURCE_NAMES)
    parser.add_argument("--max-per-source", type=int, default=DEFAULT_MAX_PER_SOURCE)
    parser.add_argument("--extra", nargs="*", type=Path, default=[], help="file .jsonl {messages} có sẵn")
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    counts = build_sft_data(
        Path(args.raw_dir), Path(args.out_dir), args.only, args.max_per_source, args.extra, args.skip_missing
    )
    print(f"Tổng: train {counts['train']:,} | eval {counts['eval']:,} -> {args.out_dir}")


if __name__ == "__main__":
    main()
