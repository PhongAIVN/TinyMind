"""Cặp DPO trên thế giới đồ chơi: chosen = đáp số đúng, rejected = đáp số sai kiểu model hay mắc.

Kiểu sai lấy từ lỗi thật của model sau LoRA: chép lại một vế ("2 + 35 = 35"), lệch vài đơn vị.
Bộ eval dùng đúng các bài của dataset/toy_math/eval.jsonl; bộ train không trùng bài nào với nó.
Ra <out_dir>/train.jsonl và eval.jsonl, mỗi dòng {"prompt": [...], "chosen": ..., "rejected": ..., "result": ...}.
"""

import argparse
import json
import random
from pathlib import Path

from scripts.build_toy_math import Problem, make_answer, make_question, parse_problem, random_problem, solve

MAX_OFFSET = 10


def wrong_result(problem: Problem, rng: random.Random) -> int:
    """Một đáp số sai, khác đáp số đúng."""
    _, a, b = problem
    result = solve(problem)
    candidates = [a, b] + [result + offset for offset in range(-MAX_OFFSET, MAX_OFFSET + 1) if offset]
    candidates = [value for value in candidates if value != result and value >= 0]
    return rng.choice(candidates)


def to_pair(problem: Problem, rng: random.Random) -> dict:
    result = solve(problem)
    return {
        "prompt": [{"role": "user", "content": make_question(problem, rng)}],
        "chosen": make_answer(problem, result),
        "rejected": make_answer(problem, wrong_result(problem, rng)),
        "result": result,
    }


def read_eval_problems(toy_eval_path: Path) -> list[Problem]:
    problems = []
    for line in toy_eval_path.read_text(encoding="utf-8").splitlines():
        problems.append(parse_problem(json.loads(line)["messages"][1]["content"]))
    return problems


def build_toy_dpo(out_dir: Path, toy_eval_path: Path, n_train: int, seed: int = 0) -> dict[str, int]:
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
                out_file.write(json.dumps(to_pair(problem, rng), ensure_ascii=False) + "\n")
    return {"train": len(train_problems), "eval": len(eval_problems)}


def main() -> None:
    """python -m scripts.build_toy_dpo [--n-train 8000]"""
    parser = argparse.ArgumentParser(description="cặp chosen/rejected cho DPO trên thế giới đồ chơi")
    parser.add_argument("--out-dir", default="dataset/toy_dpo")
    parser.add_argument("--toy-eval", default="dataset/toy_math/eval.jsonl")
    parser.add_argument("--n-train", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    counts = build_toy_dpo(Path(args.out_dir), Path(args.toy_eval), args.n_train, args.seed)
    print(f"train {counts['train']:,} | eval {counts['eval']:,} -> {args.out_dir}")


if __name__ == "__main__":
    main()
