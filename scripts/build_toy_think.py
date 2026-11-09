"""Dữ liệu tư duy thích ứng: cùng một bài, khi thì có phần <think> lập luận, khi thì <think> rỗng.

Bật/tắt lúc sinh chữ bằng prefill (xem inference/eval_toy_math.py):
    bật: assistant viết tiếp sau "<think>\\n"
    tắt: assistant viết tiếp sau "<think>\\n\\n</think>\\n"
Lập luận tách hàng chục / hàng đơn vị, vì tokenizer gộp cả số "29" thành một token
nên model khó cộng thẳng hai số hai chữ số.
"""

import argparse
import json
import random
from pathlib import Path

from scripts.build_toy_dpo import read_eval_problems
from scripts.build_toy_math import Problem, make_answer, make_question, random_problem, solve

THINK_TEMPLATE = "<think>\n{reasoning}\n</think>\n{answer}"
NO_THINK_TEMPLATE = "<think>\n\n</think>\n{answer}"
TEN = 10


def reasoning_steps(problem: Problem) -> str:
    """Các bước tính trung gian, mỗi bước đều đúng."""
    symbol, a, b = problem
    result = solve(problem)
    if symbol == "+":
        if a < TEN or b < TEN:
            return f"{a} + {b} = {result}"
        tens, units = a // TEN * TEN + b // TEN * TEN, a % TEN + b % TEN
        return f"{a} + {b} = {a // TEN * TEN} + {b // TEN * TEN} + {a % TEN} + {b % TEN} = {tens} + {units} = {result}"
    if symbol == "-":
        if b < TEN:
            return f"{a} - {b} = {result}"
        after_tens = a - b // TEN * TEN
        return f"{a} - {b} = {a} - {b // TEN * TEN} - {b % TEN} = {after_tens} - {b % TEN} = {result}"
    if a < TEN:
        return f"{a} × {b} = {result}"
    return f"{a} × {b} = {TEN} × {b} + {a - TEN} × {b} = {TEN * b} + {(a - TEN) * b} = {result}"


def to_think_example(problem: Problem, thinking: bool, rng: random.Random) -> dict:
    answer = make_answer(problem, solve(problem))
    content = THINK_TEMPLATE.format(reasoning=reasoning_steps(problem), answer=answer) if thinking \
        else NO_THINK_TEMPLATE.format(answer=answer)
    return {
        "messages": [{"role": "user", "content": make_question(problem, rng)}, {"role": "assistant", "content": content}],
        "result": solve(problem),
        "thinking": thinking,
    }


def build_toy_think(out_dir: Path, toy_eval_path: Path, n_train: int, think_ratio: float = 0.5, seed: int = 3) -> dict[str, int]:
    rng = random.Random(seed)
    eval_problems = read_eval_problems(toy_eval_path)
    excluded = set(eval_problems)
    train_problems = []
    while len(train_problems) < n_train:
        problem = random_problem(rng)
        if problem not in excluded:
            train_problems.append(problem)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as out_file:
        for problem in train_problems:
            example = to_think_example(problem, rng.random() < think_ratio, rng)
            out_file.write(json.dumps(example, ensure_ascii=False) + "\n")
    with open(out_dir / "eval.jsonl", "w", encoding="utf-8") as out_file:
        for problem in eval_problems:
            out_file.write(json.dumps(to_think_example(problem, True, rng), ensure_ascii=False) + "\n")
    return {"train": len(train_problems), "eval": len(eval_problems)}


def main() -> None:
    """python -m scripts.build_toy_think [--n-train 12000]"""
    parser = argparse.ArgumentParser(description="dữ liệu bật/tắt tư duy trên thế giới đồ chơi")
    parser.add_argument("--out-dir", default="dataset/toy_think")
    parser.add_argument("--toy-eval", default="dataset/toy_math/eval.jsonl")
    parser.add_argument("--n-train", type=int, default=12_000)
    parser.add_argument("--think-ratio", type=float, default=0.5)
    args = parser.parse_args()
    counts = build_toy_think(Path(args.out_dir), Path(args.toy_eval), args.n_train, args.think_ratio)
    print(f"train {counts['train']:,} | eval {counts['eval']:,} -> {args.out_dir}")


if __name__ == "__main__":
    main()
