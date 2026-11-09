"""Tool Use: máy tính an toàn, đọc lời gọi tool, vòng lặp agent khớp khuôn train, kết quả tool không được học."""

import json
from pathlib import Path

import pytest

from inference.agent import final_answer, run_agent, tool_result_tokens
from inference.tools import calculator, find_tool_call, run_tool, tool_call_text
from trainer.losses import IGNORE_INDEX
from trainer.sft_dataset import ChatEncoder

REAL_TOKENIZER = Path("tokenizer/tokenizer.json")
needs_tokenizer = pytest.mark.skipif(not REAL_TOKENIZER.exists(), reason="cần tokenizer thật")
CONVERSATION = [
    {"role": "user", "content": "Tính 72 + 97."},
    {"role": "assistant", "content": tool_call_text("72 + 97")},
    {"role": "tool", "content": "169"},
    {"role": "assistant", "content": "72 + 97 = 169."},
]


@pytest.fixture(scope="module")
def encoder():
    from tokenizers import Tokenizer

    return ChatEncoder(Tokenizer.from_file(str(REAL_TOKENIZER)))


@pytest.mark.parametrize("expression, expected", [
    ("72 + 97", "169"), ("97 - 57", "40"), ("12 × 7", "84"), ("3*4", "12"), ("8 / 2", "4"), ("7 / 2", "3.5"),
])
def test_calculator(expression, expected):
    assert calculator(expression) == expected


@pytest.mark.parametrize("expression", ["__import__('os')", "1 + 2 + 3", "", "5 / 0"])
def test_calculator_rejects_without_running_code(expression):
    assert calculator(expression).startswith("Lỗi")


def test_find_tool_call_roundtrip_and_bad_input():
    assert find_tool_call("xx" + tool_call_text("1 + 2")) == ("calculator", {"expression": "1 + 2"})
    assert find_tool_call("<tool_call>{hỏng</tool_call>") is None
    assert find_tool_call("không gọi gì") is None
    assert run_tool("rm", {}).startswith("Lỗi")
    assert run_tool("calculator", {"x": 1}).startswith("Lỗi")


@needs_tokenizer
def test_tool_response_tokens_are_not_learned(encoder):
    ids, labels = encoder.encode_conversation(CONVERSATION)
    learned_ids = [token for token, label in zip(ids, labels) if label != IGNORE_INDEX]
    learned = encoder.tokenizer.decode(learned_ids, skip_special_tokens=False)
    assert "<tool_response>" not in learned and "169</tool_response>" not in learned
    assert learned.startswith("<tool_call>") and learned.endswith("72 + 97 = 169.<|im_end|>")


@needs_tokenizer
def test_agent_builds_exactly_the_training_sequence(encoder):
    """Model 'giả' sinh đúng 2 lượt assistant; chuỗi agent dựng phải trùng hội thoại train."""
    full_ids, _ = encoder.encode_conversation(CONVERSATION)
    header = len(encoder.headers["assistant"])
    separator = len(encoder.separator)
    turns = [encoder.encode_turn("assistant", m["content"])[0] for m in CONVERSATION if m["role"] == "assistant"]
    replies = iter([turn[header:-separator] for turn in turns])  # nội dung + <|im_end|>

    ids, generated, results = run_agent(encoder, encoder.encode_prompt("Tính 72 + 97."), lambda _ids: next(replies))

    assert results == ["169"]
    assert ids == full_ids[: len(ids)] and len(full_ids) - len(ids) == separator
    assert final_answer(encoder, ids, generated) == "72 + 97 = 169."
    tool_part = tool_result_tokens(encoder, "169")
    start = next(i for i in range(len(ids)) if ids[i : i + len(tool_part)] == tool_part)
    assert not any(generated[start : start + len(tool_part)])  # kết quả tool không do model sinh


def test_toy_tool_examples_use_correct_results(tmp_path):
    from scripts.build_toy_math import build_toy_math
    from scripts.build_toy_tools import build_toy_tools

    build_toy_math(tmp_path / "math", n_train=5, n_eval=20)
    counts = build_toy_tools(tmp_path / "tools", tmp_path / "math" / "eval.jsonl", n_train=200)
    assert counts == {"train": 200, "eval": 20}
    for line in (tmp_path / "tools" / "train.jsonl").read_text(encoding="utf-8").splitlines():
        example = json.loads(line)
        assert [m["role"] for m in example["messages"]] == ["user", "assistant", "tool", "assistant"]
        assert example["messages"][2]["content"] == str(example["result"])
