"""Tư duy thích ứng: lập luận đúng từng bước; prefill bật/tắt trùng token với dữ liệu train."""

import json
import re
from pathlib import Path

import pytest

from inference.eval_toy_math import THINK_OFF, THINK_ON, prompt_for
from scripts.build_toy_math import OPERATIONS, MAX_OPERAND, solve
from scripts.build_toy_think import build_toy_think, reasoning_steps, to_think_example
from trainer.losses import IGNORE_INDEX
from trainer.sft_dataset import ChatEncoder

REAL_TOKENIZER = Path("tokenizer/tokenizer.json")
needs_tokenizer = pytest.mark.skipif(not REAL_TOKENIZER.exists(), reason="cần tokenizer thật")
STEP = re.compile(r"(\d+) ([+\-×]) (\d+)")


def evaluate_expression(expression: str) -> int:
    """Tính biểu thức chỉ gồm + - × giữa các số nguyên (× trước)."""
    terms = re.split(r" ([+\-]) ", expression)
    value = 0
    sign = "+"
    for index, term in enumerate(terms):
        if index % 2:
            sign = term
            continue
        product = 1
        for factor in term.split(" × "):
            product *= int(factor)
        value = value + product if sign == "+" else value - product
    return value


def test_every_reasoning_step_is_equal_to_result():
    for symbol in OPERATIONS:
        for a in range(MAX_OPERAND[symbol] + 1):
            for b in range(0, MAX_OPERAND[symbol] + 1, 7):
                problem = (symbol, max(a, b), min(a, b)) if symbol == "-" else (symbol, a, b)
                for part in reasoning_steps(problem).split(" = "):
                    assert evaluate_expression(part) == solve(problem), (problem, part)


@needs_tokenizer
@pytest.mark.parametrize("thinking, prefill", [(True, THINK_ON), (False, THINK_OFF)])
def test_prefill_matches_training_tokens(thinking, prefill):
    from tokenizers import Tokenizer
    import random

    encoder = ChatEncoder(Tokenizer.from_file(str(REAL_TOKENIZER)))
    example = to_think_example(("+", 72, 97), thinking, random.Random(0))
    ids, labels = encoder.encode_conversation(example["messages"])
    mode = "think" if thinking else "no_think"
    prompt = prompt_for(encoder, example["messages"][0]["content"], mode)

    assert ids[: len(prompt)] == prompt  # lúc sinh chữ, prefill trùng từng token với lúc train
    assert labels[len(prompt)] != IGNORE_INDEX


def test_build_toy_think_mixes_modes(tmp_path):
    from scripts.build_toy_math import build_toy_math

    build_toy_math(tmp_path / "math", n_train=5, n_eval=10)
    build_toy_think(tmp_path / "think", tmp_path / "math" / "eval.jsonl", n_train=400)
    flags = [json.loads(line)["thinking"] for line in (tmp_path / "think" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert 0.35 < sum(flags) / len(flags) < 0.65
