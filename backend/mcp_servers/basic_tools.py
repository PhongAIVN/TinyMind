"""Basic MCP server (stdio): calculator, current time, Wikipedia summary.

The backend starts this as a subprocess (see mcp_servers.json). It also works in any MCP client:
    python mcp_servers/basic_tools.py
"""

import ast
import operator
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import httpx
from mcp.server.mcpserver import MCPServer

ERROR_PREFIX = "Lỗi: "
MAX_EXPRESSION_LENGTH = 200
MAX_POWER = 100
WIKIPEDIA_LANGS = ("vi", "en")
WIKIPEDIA_TIMEOUT_S = 8.0
SUMMARY_MAX_CHARS = 600
# Wikimedia rejects generic user agents (403); their policy asks for a project URL.
USER_AGENT = "TinyMind/0.1 (https://github.com/PhongSEVN/Small-Language-Model)"

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

server = MCPServer("tinymind-basic-tools")


def _evaluate(node: ast.AST) -> float:
    """Walk an arithmetic AST; anything other than numbers and + - * / // % ** raises ValueError."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POWER:
            raise ValueError("số mũ quá lớn")
        return _OPERATORS[type(node.op)](left, right)
    raise ValueError("chỉ nhận số và + - * / // % **")


def _format_number(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value) if isinstance(value, int) else f"{value:.6g}"


@server.tool()
def calculator(expression: str) -> str:
    """Tính biểu thức số học, ví dụ "72 + 97" hoặc "(3 + 4) × 5"."""
    text = str(expression).replace("×", "*").replace("x", "*").replace("÷", "/")
    if len(text) > MAX_EXPRESSION_LENGTH:
        return f"{ERROR_PREFIX}biểu thức quá dài"
    try:
        return _format_number(_evaluate(ast.parse(text, mode="eval").body))
    except ZeroDivisionError:
        return f"{ERROR_PREFIX}chia cho 0"
    except (SyntaxError, ValueError, OverflowError) as error:
        return f"{ERROR_PREFIX}{error}"


@server.tool()
def current_time(utc_offset_hours: int = 7) -> str:
    """Ngày giờ hiện tại theo múi giờ UTC+offset (mặc định giờ Việt Nam, UTC+7)."""
    if not -12 <= utc_offset_hours <= 14:
        return f"{ERROR_PREFIX}utc_offset_hours phải trong [-12, 14]"
    now = datetime.now(timezone(timedelta(hours=utc_offset_hours)))
    return now.strftime("%H:%M ngày %d/%m/%Y") + f" (UTC{utc_offset_hours:+d})"


@server.tool()
def wikipedia_summary(title: str, lang: str = "vi") -> str:
    """Đoạn tóm tắt đầu bài Wikipedia của một chủ đề."""
    if lang not in WIKIPEDIA_LANGS:
        return f"{ERROR_PREFIX}lang phải thuộc {WIKIPEDIA_LANGS}"
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title.strip().replace(' ', '_'))}"
    try:
        response = httpx.get(url, timeout=WIKIPEDIA_TIMEOUT_S, follow_redirects=True,
                             headers={"User-Agent": USER_AGENT})
    except httpx.HTTPError as error:
        return f"{ERROR_PREFIX}không gọi được Wikipedia ({type(error).__name__})"
    if response.status_code == 404:
        return f"{ERROR_PREFIX}không có bài '{title}'"
    if response.status_code != 200:
        return f"{ERROR_PREFIX}Wikipedia trả mã {response.status_code}"
    return response.json().get("extract", "")[:SUMMARY_MAX_CHARS]


if __name__ == "__main__":
    server.run("stdio")
