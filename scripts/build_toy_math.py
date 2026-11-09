"""Thế giới đồ chơi: phép tính hỏi bằng tiếng Việt, chấm đúng/sai tự động.

Dùng chung cho LoRA, DPO, tool use, RL: chặng nào cũng đo được độ chính xác trên máy yếu.
Ra <out_dir>/train.jsonl và eval.jsonl, mỗi dòng {"messages": [...], "result": <đáp số>}.
Bộ eval không trùng cặp (phép tính, a, b) nào với bộ train.
"""

import argparse
import json
import random
import re
from pathlib import Path

OPERATIONS = {
    "+": ("cộng", lambda a, b: a + b),
    "-": ("trừ", lambda a, b: a - b),
    "×": ("nhân", lambda a, b: a * b),
}
MAX_OPERAND = {"+": 99, "-": 99, "×": 12}
QUESTION_TEMPLATES = (
    "{a} {word} {b} bằng bao nhiêu?",
    "Tính {a} {symbol} {b}.",
    "Kết quả của {a} {word} {b} là gì?",
    "Hãy tính giúp mình: {a} {symbol} {b}",
)
ANSWER_TEMPLATE = "{a} {symbol} {b} = {result}."
INTEGER_PATTERN = re.compile(r"-?\d+")
PROBLEM_PATTERN = re.compile(r"(\d+) ([+\-×]) (\d+) =")

Problem = tuple[str, int, int]  # (phép tính, a, b)


def solve(problem: Problem) -> int:
    symbol, a, b = problem
    return OPERATIONS[symbol][1](a, b)


def random_problem(rng: random.Random) -> Problem:
    symbol = rng.choice(list(OPERATIONS))
    a, b = rng.randint(0, MAX_OPERAND[symbol]), rng.randint(0, MAX_OPERAND[symbol])
    if symbol == "-" and a < b:
        a, b = b, a  # không ra số âm
    return symbol, a, b


def make_question(problem: Problem, rng: random.Random) -> str:
    symbol, a, b = problem
    return rng.choice(QUESTION_TEMPLATES).format(a=a, b=b, word=OPERATIONS[symbol][0], symbol=symbol)


def make_answer(problem: Problem, result: int) -> str:
    symbol, a, b = problem
    return ANSWER_TEMPLATE.format(a=a, b=b, symbol=symbol, result=result)


def parse_problem(answer: str) -> Problem:
    """Ngược của make_answer: "12 + 7 = 19." -> ("+", 12, 7)."""
    match = PROBLEM_PATTERN.match(answer)
    if match is None:
        raise ValueError(f"không đọc được phép tính từ {answer!r}")
    return match.group(2), int(match.group(1)), int(match.group(3))


def to_example(problem: Problem, rng: random.Random) -> dict:
    result = solve(problem)
    return {
        "messages": [
            {"role": "user", "content": make_question(problem, rng)},
            {"role": "assistant", "content": make_answer(problem, result)},
        ],
        "result": result,
    }


def parse_answer(text: str) -> int | None:
    """Đáp số = số nguyên cuối cùng trong câu trả lời (khớp ANSWER_TEMPLATE); không có số thì None."""
    numbers = INTEGER_PATTERN.findall(text)
    return int(numbers[-1]) if numbers else None


def build_toy_math(out_dir: Path, n_train: int, n_eval: int, seed: int = 0) -> dict[str, int]:
    """Sinh n_eval bài eval trước, rồi n_train bài train không trùng bài eval nào."""
    rng = random.Random(seed)
    eval_problems: set[Problem] = set()
    while len(eval_problems) < n_eval:
        eval_problems.add(random_problem(rng))
    train_problems = []
    while len(train_problems) < n_train:
        problem = random_problem(rng)
        if problem not in eval_problems:
            train_problems.append(problem)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, problems in (("eval", sorted(eval_problems)), ("train", train_problems)):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as out_file:
            for problem in problems:
                out_file.write(json.dumps(to_example(problem, rng), ensure_ascii=False) + "\n")
    return {"train": len(train_problems), "eval": len(eval_problems)}


def main() -> None:
    """python -m scripts.build_toy_math [--n-train 20000] [--n-eval 300]"""
    parser = argparse.ArgumentParser(description="sinh bộ phép tính đồ chơi")
    parser.add_argument("--out-dir", default="dataset/toy_math")
    parser.add_argument("--n-train", type=int, default=20_000)
    parser.add_argument("--n-eval", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    counts = build_toy_math(Path(args.out_dir), args.n_train, args.n_eval, args.seed)
    print(f"train {counts['train']:,} | eval {counts['eval']:,} -> {args.out_dir}")


if __name__ == "__main__":
    main()
