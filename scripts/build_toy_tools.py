"""Dữ liệu Tool Use trên thế giới đồ chơi: model gọi máy tính thay vì tự nhẩm.

Hội thoại mẫu:
    user:      72 cộng 97 bằng bao nhiêu?
    assistant: <tool_call>{"name": "calculator", "arguments": {"expression": "72 + 97"}}</tool_call>
    tool:      169                       (không được học)
    assistant: 72 + 97 = 169.
Bài eval = đúng các bài của dataset/toy_math/eval.jsonl; bài train không trùng.
"""

import argparse
import json
import random
from pathlib import Path

from inference.tools import calculator, tool_call_text
from scripts.build_toy_dpo import read_eval_problems
from scripts.build_toy_math import Problem, make_answer, make_question, random_problem, solve


def to_tool_example(problem: Problem, rng: random.Random) -> dict:
    symbol, a, b = problem
    expression = f"{a} {symbol} {b}"
    result = calculator(expression)
    if result != str(solve(problem)):
        raise AssertionError(f"máy tính sai với {problem}: {result}")
    return {
        "messages": [
            {"role": "user", "content": make_question(problem, rng)},
            {"role": "assistant", "content": tool_call_text(expression)},
            {"role": "tool", "content": result},
            {"role": "assistant", "content": make_answer(problem, solve(problem))},
        ],
        "result": solve(problem),
    }


def build_toy_tools(out_dir: Path, toy_eval_path: Path, n_train: int, seed: int = 2) -> dict[str, int]:
    rng = random.Random(seed)
    eval_problems = read_eval_problems(toy_eval_path)
    excluded = set(eval_problems)
    train_problems = []
    while len(train_problems) < n_train:
        problem = random_problem(rng)
        if problem not in excluded:
            train_problems.append(problem)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, problems in (("train", train_problems), ("eval", eval_problems)):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as out_file:
            for problem in problems:
                out_file.write(json.dumps(to_tool_example(problem, rng), ensure_ascii=False) + "\n")
    return {"train": len(train_problems), "eval": len(eval_problems)}


def main() -> None:
    """python -m scripts.build_toy_tools [--n-train 8000]"""
    parser = argparse.ArgumentParser(description="hội thoại gọi máy tính trên thế giới đồ chơi")
    parser.add_argument("--out-dir", default="dataset/toy_tools")
    parser.add_argument("--toy-eval", default="dataset/toy_math/eval.jsonl")
    parser.add_argument("--n-train", type=int, default=8_000)
    args = parser.parse_args()
    counts = build_toy_tools(Path(args.out_dir), Path(args.toy_eval), args.n_train)
    print(f"train {counts['train']:,} | eval {counts['eval']:,} -> {args.out_dir}")


if __name__ == "__main__":
    main()
