"""Chấm mọi checkpoint đã có trên thế giới đồ chơi, ghi bảng ra logs/eval_results.md."""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from inference.eval_toy_math import evaluate_toy_math
from inference.generate import load_model

# (checkpoint, chế độ): mỗi chặng chấm ở chế độ nó được dạy
EVALUATIONS = (
    ("sft.pt", "plain"),
    ("lora_merged.pt", "plain"),
    ("dpo.pt", "plain"),
    ("tool.pt", "tools"),
    ("think.pt", "no_think"),
    ("think.pt", "think"),
    ("student.pt", "think"),
    ("grpo.pt", "think"),
    ("cispo.pt", "think"),
    ("ppo.pt", "think"),
    ("agentic.pt", "tools"),
)


def main() -> None:
    """python -m scripts.eval_all [--limit 300]"""
    parser = argparse.ArgumentParser(description="bảng độ chính xác của mọi chặng")
    parser.add_argument("--checkpoint-dir", default="checkpoints/50m")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--eval", default="dataset/toy_math/eval.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="logs/eval_results.md")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    rows = ["| Checkpoint | Chế độ | Đúng | Độ chính xác |", "| ---------- | ------ | ---- | ------------ |"]
    for name, mode in EVALUATIONS:
        path = Path(args.checkpoint_dir) / name
        if not path.exists():
            print(f"[bỏ qua] chưa có {path}")
            continue
        model = load_model(path, args.device)
        with torch.no_grad():
            stats = evaluate_toy_math(model, tokenizer, args.eval, args.limit, mode)
        line = f"| {name} | {mode} | {stats['correct']}/{stats['total']} | {stats['accuracy']:.1%} |"
        print(line, flush=True)
        rows.append(line)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
