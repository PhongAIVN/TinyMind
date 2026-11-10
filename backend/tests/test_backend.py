"""Run from backend/:  python -m pytest tests -q"""

import pytest
from tokenizers import Tokenizer

from app.db import ChatStore
from app.engine import ChatFormat, build_prompt, drop_boilerplate
from app.memory import extract_facts, recall_answer
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


def test_build_prompt_shortens_old_answers(fmt):
    history = [{"role": "user", "content": "Kể chuyện"}, {"role": "assistant", "content": "ngày xưa " * 100}]
    text = fmt.decode(build_prompt(fmt, history, "Tiếp đi", "", max_len=512, max_new=100))
    assert "Kể chuyện" in text and text.count("ngày xưa") < 40 and "…" in text


def test_build_prompt_drops_oldest_turns_but_keeps_their_questions(fmt):
    history = [{"role": "user", "content": "câu hỏi đầu " + "dài " * 80}, {"role": "assistant", "content": "trả lời"},
               {"role": "user", "content": "câu hai"}, {"role": "assistant", "content": "đáp hai"}]
    ids = build_prompt(fmt, history, "câu mới", "", max_len=128, max_new=32)
    text = fmt.decode(ids)
    assert len(ids) + 32 <= 128
    assert "Trước đó người dùng đã hỏi: câu hỏi đầu" in text
    assert text.rstrip().endswith("user\ncâu mới<|im_end|>\n<|im_start|>assistant")


def test_build_prompt_has_no_system_turn_when_nothing_dropped(fmt):
    text = fmt.decode(build_prompt(fmt, [], "Tên mình là gì?", "", 512, 100))
    assert text.startswith("<|bos|><|im_start|>user\n")


def test_build_prompt_rejects_message_that_cannot_fit(fmt):
    with pytest.raises(ValueError):
        build_prompt(fmt, [], "rất dài " * 200, "", max_len=128, max_new=32)


@pytest.mark.parametrize("message, expected", [
    ("Mình tên là Phong, mình 22 tuổi", ["Tên: Phong", "Tuổi: 22"]),
    ("Tôi tên Phong và tôi thích bóng đá", ["Tên: Phong", "Thích: bóng đá"]),
    ("Mình đang sống ở Hà Nội", ["Sống ở: Hà Nội"]),
    ("Ghi nhớ: thứ 6 có họp", ["Ghi nhớ: thứ 6 có họp"]),
    ("Tôi tên gì?", []),
    ("Nhớ mẹ quá", []),
])
def test_extract_facts(message, expected):
    assert [fact.text for fact in extract_facts(message)] == expected


def test_memory_single_valued_kind_replaces_old_value(tmp_path):
    store = ChatStore(tmp_path / "chat.db")
    store.remember("age", "22", single_valued=True)
    store.remember("age", "23", single_valued=True)
    store.remember("likes", "AI", single_valued=False)
    store.remember("likes", "ai", single_valued=False)
    assert [(m["kind"], m["value"]) for m in store.list_memories()] == [("age", "23"), ("likes", "ai")]
    assert store.forget_all() == 2


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


def test_recall_answer_uses_only_matching_memories():
    memories = [{"kind": "name", "value": "Phong"}, {"kind": "age", "value": "22"}]
    assert recall_answer("Mình bao nhiêu tuổi?", memories) == "Mình nhớ: tuổi 22."
    assert recall_answer("Bạn nhớ gì về mình?", memories) == "Mình nhớ: tên Phong; tuổi 22."
    assert recall_answer("Mình sống ở đâu?", memories) is None
    assert recall_answer("Thủ đô của Pháp là gì?", memories) is None


@pytest.mark.parametrize("message, expected", [
    ("Mình tên là Phong, mình 22 tuổi và sống ở Hà Nội", ["Tên: Phong", "Tuổi: 22", "Sống ở: Hà Nội"]),
    ("Tên của mình là gì vậy bạn", []),
    ("Mình thích ăn gì nhỉ", []),
])
def test_extract_facts_handles_questions_without_question_mark(message, expected):
    assert [fact.text for fact in extract_facts(message)] == expected


def test_drop_boilerplate_removes_filler_answer_and_its_question():
    history = [
        {"role": "user", "content": "Thủ đô của Pháp?"}, {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "Chính trị?"},
        {"role": "assistant", "content": "Xin lỗi, là một mô hình ngôn ngữ AI, tôi không thể..."},
    ]
    assert drop_boilerplate(history) == history[:2]
