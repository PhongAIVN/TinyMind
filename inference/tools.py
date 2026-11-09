"""Công cụ cho model gọi, và cách đọc lời gọi tool trong câu trả lời.

Lời gọi: <tool_call>{"name": "calculator", "arguments": {"expression": "72 + 97"}}</tool_call>
Máy tính chỉ nhận "số phép-tính số" (không dùng eval: model sinh ra chuỗi gì cũng không chạy được code).
"""

import json
import re
from collections.abc import Callable

TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
EXPRESSION_PATTERN = re.compile(r"^\s*(-?\d+)\s*([+\-×x*/])\s*(-?\d+)\s*$")
CALCULATOR = "calculator"
ERROR_PREFIX = "Lỗi: "


def calculator(expression: str) -> str:
    """Tính "a op b" với op thuộc + - × x * /. Biểu thức lạ thì trả chuỗi lỗi (không ném lỗi)."""
    match = EXPRESSION_PATTERN.match(str(expression))
    if match is None:
        return f"{ERROR_PREFIX}chỉ tính được 'a op b', nhận {expression!r}"
    a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
    if op == "+":
        return str(a + b)
    if op == "-":
        return str(a - b)
    if op in "×x*":
        return str(a * b)
    if b == 0:
        return f"{ERROR_PREFIX}chia cho 0"
    return str(a // b) if a % b == 0 else f"{a / b:.4g}"


TOOLS: dict[str, Callable[..., str]] = {CALCULATOR: calculator}


def tool_call_text(expression: str) -> str:
    """Nội dung assistant khi gọi máy tính (dùng để sinh dữ liệu train)."""
    payload = {"name": CALCULATOR, "arguments": {"expression": expression}}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"


def find_tool_call(text: str) -> tuple[str, dict] | None:
    """Lời gọi tool đầu tiên trong text -> (tên, arguments); không có hoặc JSON hỏng -> None."""
    match = TOOL_CALL_PATTERN.search(text)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("arguments"), dict):
        return None
    return payload.get("name"), payload["arguments"]


def run_tool(name: str, arguments: dict) -> str:
    """Chạy tool theo tên; tên lạ hoặc tham số sai thì trả chuỗi lỗi để model đọc."""
    tool = TOOLS.get(name)
    if tool is None:
        return f"{ERROR_PREFIX}không có tool {name!r}"
    try:
        return tool(**arguments)
    except TypeError as error:
        return f"{ERROR_PREFIX}tham số sai ({error})"
