"""Run from backend/:  python -m pytest tests -q"""

import pytest
from tokenizers import Tokenizer

from app.db import ChatStore
from app.engine import ChatFormat, fit_history
from app.harness import parse_tool_call
from app.settings import TOKENIZER_PATH
from mcp_servers.basic_tools import calculator


@pytest.fixture(scope="module")
def fmt() -> ChatFormat:
    return ChatFormat(Tokenizer.from_file(str(TOKENIZER_PATH)))


@pytest.mark.parametrize("expression, expected", [
    ("72 + 97", "169"), ("12 × 7", "84"), ("(3 + 4) * 5", "35"), ("7 / 2", "3.5"), ("-3 - 4", "-7"),
])
def test_calculator_evaluates_arithmetic(expression, expected):
    assert calculator(expression) == expected


@pytest.mark.parametrize("expression", ["__import__('os').system('dir')", "a + 1", "2 ** 1000", "1 / 0"])
def test_calculator_rejects_non_arithmetic_and_errors(expression):
    assert calculator(expression).startswith("Lỗi: ")


def test_parse_tool_call():
    assert parse_tool_call('{"name": "calculator", "arguments": {"expression": "1 + 2"}}') == (
        "calculator", {"expression": "1 + 2"})
    assert parse_tool_call("{not json") is None
    assert parse_tool_call('{"name": "x", "arguments": [1]}') is None


def test_prompt_matches_training_template(fmt):
    ids = fmt.prompt([{"role": "user", "content": "Xin chào"}])
    assert fmt.decode(ids) == "<|bos|><|im_start|>user\nXin chào<|im_end|>\n<|im_start|>assistant\n"


def test_fit_history_drops_oldest_turns_first(fmt):
    old = [{"role": "user", "content": "câu cũ " * 100}, {"role": "assistant", "content": "trả lời " * 100}]
    newest = {"role": "user", "content": "câu mới"}
    ids = fit_history(fmt, old + [newest], "", max_len=128, max_new=32)
    assert len(ids) + 32 <= 128
    assert fmt.decode(ids).startswith("<|bos|><|im_start|>user\ncâu mới")


def test_fit_history_rejects_message_that_cannot_fit(fmt):
    with pytest.raises(ValueError):
        fit_history(fmt, [{"role": "user", "content": "rất dài " * 200}], "", max_len=128, max_new=32)


def test_chat_store_roundtrip_and_cascade_delete(tmp_path):
    store = ChatStore(tmp_path / "chat.db")
    conversation = store.create_conversation("Chào", "sft")
    store.add_message(conversation["id"], "user", "Chào")
    store.add_message(conversation["id"], "assistant", "Xin chào!", tool_calls=[{"name": "calculator"}])
    loaded = store.get_conversation(conversation["id"])
    assert [m["role"] for m in loaded["messages"]] == ["user", "assistant"]
    assert loaded["messages"][1]["tool_calls"] == [{"name": "calculator"}]
    assert store.rename(conversation["id"], "Mới") and store.list_conversations()[0]["title"] == "Mới"
    assert store.delete(conversation["id"]) and store.get_conversation(conversation["id"]) is None
    assert store.delete_all() == 0
