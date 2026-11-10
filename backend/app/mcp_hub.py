"""MCP client side: start the servers listed in mcp_servers.json and route tool calls to them.

mcp_servers.json:
    {"servers": {"basic": {"command": "python", "args": ["mcp_servers/basic_tools.py"]}}}
"python" means the interpreter running the backend; relative paths resolve against backend/.
"""

import json
import logging
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client

from app.settings import BACKEND_DIR

log = logging.getLogger(__name__)
ERROR_PREFIX = "Lỗi: "
CALL_TIMEOUT_S = 15.0


@dataclass(frozen=True)
class ToolInfo:
    name: str
    description: str
    input_schema: dict
    server: str


class McpHub:
    """Holds one ClientSession per server. Tool names must be unique across servers."""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self.tools: dict[str, ToolInfo] = {}
        self.errors: dict[str, str] = {}

    async def start(self, config_path: Path) -> None:
        if not config_path.exists():
            return
        servers = json.loads(config_path.read_text(encoding="utf-8")).get("servers", {})
        for name, spec in servers.items():
            try:
                await self._connect(name, spec)
            except Exception as error:  # one broken server must not stop the backend
                log.exception("MCP server %s failed to start", name)
                self.errors[name] = f"{type(error).__name__}: {error}"

    async def _connect(self, name: str, spec: dict) -> None:
        command = sys.executable if spec["command"] == "python" else spec["command"]
        params = StdioServerParameters(command=command, args=spec.get("args", []),
                                       env=spec.get("env"), cwd=str(BACKEND_DIR))
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[name] = session
        for tool in (await session.list_tools()).tools:
            if tool.name in self.tools:
                log.warning("tool %s from %s shadows %s", tool.name, name, self.tools[tool.name].server)
            self.tools[tool.name] = ToolInfo(tool.name, tool.description or "", tool.input_schema, name)
        log.info("MCP server %s: %d tools", name, len(self.tools))

    async def call(self, name: str, arguments: dict) -> str:
        """Run a tool; failures come back as text so the model (and the user) can read them."""
        tool = self.tools.get(name)
        if tool is None:
            return f"{ERROR_PREFIX}không có tool {name!r}"
        try:
            result = await self._sessions[tool.server].call_tool(name, arguments, read_timeout_seconds=CALL_TIMEOUT_S)
        except Exception as error:
            return f"{ERROR_PREFIX}{type(error).__name__}: {error}"
        text = "\n".join(block.text for block in getattr(result, "content", []) if getattr(block, "type", "") == "text")
        return text if not getattr(result, "is_error", False) else f"{ERROR_PREFIX}{text}"

    async def close(self) -> None:
        await self._stack.aclose()
