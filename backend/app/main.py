"""TinyMind backend: chat with streaming, history in SQLite, tools over MCP.

Run from backend/:  python -m uvicorn app.main:app --port 8000
"""

import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager

import anyio.from_thread
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.db import ChatStore
from app.engine import THINK_OFF_PREFILL, THINK_ON_PREFILL, ModelStore, build_prompt
from app.harness import Reply, run_completion, run_turn
from app.mcp_hub import McpHub
from app.memory import SINGLE_VALUED, extract_facts, recall_answer
from app.settings import (
    DB_PATH, DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, MAX_NEW_TOKENS_LIMIT,
    MAX_TOOL_TURNS, MCP_CONFIG_PATH, TITLE_LENGTH, VARIANTS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tinymind")
MAX_MESSAGE_CHARS = 4000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    conversation_id: str | None = None
    model: str = DEFAULT_MODEL
    think: bool = True
    use_memory: bool = True
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0, le=2)
    top_k: int | None = Field(DEFAULT_TOP_K, ge=1, le=1000)
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=1, le=MAX_NEW_TOKENS_LIMIT)


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = ChatStore(DB_PATH)
        app.state.models = ModelStore()
        app.state.mcp = McpHub()
        await app.state.mcp.start(MCP_CONFIG_PATH)
        log.info("device=%s models=%s tools=%s", app.state.models.device, app.state.models.available(),
                 list(app.state.mcp.tools))
        yield
        await app.state.mcp.close()

    app = FastAPI(title="TinyMind", lifespan=lifespan)

    @app.get("/api/health")
    def health(request: Request) -> dict:
        state = request.app.state
        return {"status": "ok", "device": state.models.device, "mcp_errors": state.mcp.errors}

    @app.get("/api/models")
    def models(request: Request) -> list[dict]:
        return [{"id": name, **VARIANTS[name]} for name in request.app.state.models.available()]

    @app.get("/api/tools")
    def tools(request: Request) -> list[dict]:
        return [vars(tool) for tool in request.app.state.mcp.tools.values()]

    @app.get("/api/conversations")
    def list_conversations(request: Request) -> list[dict]:
        return request.app.state.store.list_conversations()

    @app.get("/api/conversations/{conversation_id}")
    def get_conversation(conversation_id: str, request: Request) -> dict:
        conversation = request.app.state.store.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(404, "conversation not found")
        return conversation

    @app.patch("/api/conversations/{conversation_id}")
    def rename(conversation_id: str, body: RenameRequest, request: Request) -> dict:
        if not request.app.state.store.rename(conversation_id, body.title.strip()):
            raise HTTPException(404, "conversation not found")
        return {"ok": True}

    @app.delete("/api/conversations/{conversation_id}")
    def delete(conversation_id: str, request: Request) -> dict:
        if not request.app.state.store.delete(conversation_id):
            raise HTTPException(404, "conversation not found")
        return {"ok": True}

    @app.delete("/api/conversations")
    def delete_all(request: Request) -> dict:
        return {"deleted": request.app.state.store.delete_all()}

    @app.get("/api/memories")
    def list_memories(request: Request) -> list[dict]:
        return request.app.state.store.list_memories()

    @app.delete("/api/memories/{memory_id}")
    def forget(memory_id: int, request: Request) -> dict:
        if not request.app.state.store.forget(memory_id):
            raise HTTPException(404, "memory not found")
        return {"ok": True}

    @app.delete("/api/memories")
    def forget_all(request: Request) -> dict:
        return {"deleted": request.app.state.store.forget_all()}

    @app.post("/api/chat")
    def chat(body: ChatRequest, request: Request) -> StreamingResponse:
        state = request.app.state
        if body.model not in state.models.available():
            raise HTTPException(400, f"unknown model {body.model!r}")
        store: ChatStore = state.store
        if body.conversation_id:
            conversation = store.get_conversation(body.conversation_id)
            if conversation is None:
                raise HTTPException(404, "conversation not found")
        else:
            conversation = {**store.create_conversation(body.message.strip()[:TITLE_LENGTH], body.model),
                            "messages": []}
        history = [{"role": m["role"], "content": m["content"]} for m in conversation["messages"] if m["content"]]
        user_message = store.add_message(conversation["id"], "user", body.message, model=body.model)
        return StreamingResponse(
            _stream_reply(state, conversation["id"], user_message, history, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _stream_reply(state, conversation_id: str, user_message: dict, history: list[dict],
                  body: ChatRequest) -> Iterator[str]:
    """Runs in a worker thread (sync generator); MCP calls hop back to the event loop."""
    yield sse({"type": "start", "conversation_id": conversation_id, "user_message": user_message})
    variant = VARIANTS[body.model]
    reply = Reply()
    finished = False
    try:
        with state.models.generate_lock:
            loaded = state.models.get(body.model)
            fmt = state.models.fmt
            if not variant["chat"]:
                events = run_completion(loaded, fmt, body.message, max_new=body.max_new_tokens,
                                        temperature=body.temperature, top_k=body.top_k, reply=reply)
            else:
                thinking = variant["think"] and body.think
                prefill = (THINK_ON_PREFILL if thinking else THINK_OFF_PREFILL) if variant["think"] else ""
                stored: list[dict] = []
                if body.use_memory:
                    saved = [state.store.remember(f.kind, f.value, f.kind in SINGLE_VALUED)
                             for f in extract_facts(body.message)]
                    if saved:
                        yield sse({"type": "memory", "items": saved})
                    stored = state.store.list_memories()
                recalled = recall_answer(body.message, stored)
                if recalled:
                    reply.content = recalled
                    events = iter([{"type": "text", "text": recalled}])
                else:
                    prompt = build_prompt(fmt, history, body.message, prefill, loaded.max_len, body.max_new_tokens)
                    events = run_turn(
                        loaded, fmt, prompt, thinking=thinking, max_new=body.max_new_tokens,
                        temperature=body.temperature, top_k=body.top_k, max_tool_turns=MAX_TOOL_TURNS,
                        run_tool=lambda name, args: anyio.from_thread.run(state.mcp.call, name, args),
                        reply=reply,
                    )
            for event in events:
                yield sse(event)
        finished = True
    except ValueError as error:
        yield sse({"type": "error", "message": str(error)})
    except Exception as error:
        log.exception("generation failed")
        yield sse({"type": "error", "message": f"{type(error).__name__}: {error}"})
    finally:
        # Also runs when the client disconnects mid-stream: keep whatever was generated.
        if reply.content or reply.thinking or reply.tool_calls:
            saved = state.store.add_message(conversation_id, "assistant", reply.content.strip(),
                                            thinking=reply.thinking.strip(), tool_calls=reply.tool_calls,
                                            model=body.model)
            if finished:
                yield sse({"type": "done", "message": saved})
    if not finished and not (reply.content or reply.thinking):
        yield sse({"type": "done", "message": None})


app = create_app()
