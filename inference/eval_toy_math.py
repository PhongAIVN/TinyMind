"""Đo độ chính xác trên thế giới đồ chơi: sinh tham lam (greedy), so đáp số với kết quả đúng.

Chế độ: plain (trả lời thẳng), think / no_think (bật / tắt tư duy bằng prefill), tools (được gọi máy tính).
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from inference.agent import final_answer, greedy_sampler, run_agent
from inference.generate import load_model
from scripts.build_toy_math import parse_answer
from trainer.sft_dataset import ChatEncoder

MAX_ANSWER_TOKENS = 32
MAX_THINKING_TOKENS = 96
MAX_TOOL_TURNS = 2
THINK_ON = "<think>\n"
THINK_OFF = "<think>\n\n</think>\n"
MODES = {"plain": "", "think": THINK_ON, "no_think": THINK_OFF, "tools": ""}


def prompt_for(encoder: ChatEncoder, question: str, mode: str) -> list[int]:
    if mode not in MODES:
        raise ValueError(f"mode={mode!r} phải thuộc {tuple(MODES)}")
    return encoder.encode_prompt(question, prefill=MODES[mode])


def answer_question(model, encoder: ChatEncoder, question: str, mode: str = "plain") -> str:
    """Câu trả lời cuối (lượt assistant sau kết quả tool cuối, nếu có)."""
    max_tokens = MAX_THINKING_TOKENS if mode == "think" else MAX_ANSWER_TOKENS
    max_tool_turns = MAX_TOOL_TURNS if mode == "tools" else 0
    sampler = greedy_sampler(model, encoder, max_tokens)
    ids, generated, _ = run_agent(encoder, prompt_for(encoder, question, mode), sampler, max_tool_turns)
    return final_answer(encoder, ids, generated)


def evaluate_toy_math(model, tokenizer, eval_path: str | Path, limit: int | None = None, mode: str = "plain") -> dict:
    """Trả về {"accuracy", "correct", "total", "samples"}; samples = vài câu đầu để xem bằng mắt."""
    encoder = ChatEncoder(tokenizer)
    lines = Path(eval_path).read_text(encoding="utf-8").splitlines()[:limit]
    correct, samples = 0, []
    for line in lines:
        example = json.loads(line)
        question = example["messages"][0]["content"]
        answer = answer_question(model, encoder, question, mode)
        is_correct = parse_answer(answer) == example["result"]
        correct += is_correct
        if len(samples) < 5:
            samples.append((question, answer, is_correct))
    total = len(lines)
    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total, "samples": samples}


def main() -> None:
    """python -m inference.eval_toy_math --checkpoint checkpoints/50m/sft.pt [--mode tools]"""
    parser = argparse.ArgumentParser(description="độ chính xác trên bộ phép tính đồ chơi")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--eval", default="dataset/toy_math/eval.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="chỉ chấm N câu đầu")
    parser.add_argument("--mode", choices=list(MODES), default="plain")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = load_model(args.checkpoint)
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError(f"tokenizer vocab {tokenizer.get_vocab_size()} khác model vocab {model.config.vocab_size}")
    stats = evaluate_toy_math(model, tokenizer, args.eval, args.limit, args.mode)
    for question, answer, is_correct in stats["samples"]:
        print(f"[{'đúng' if is_correct else 'sai '}] {question} -> {answer}")
    print(f"[{args.mode}] Độ chính xác: {stats['correct']}/{stats['total']} = {stats['accuracy']:.1%}")


if __name__ == "__main__":
    main()
