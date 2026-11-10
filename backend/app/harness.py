"""Agent loop: generate -> if the model wrote <tool_call>{json}</tool_call>, run it via MCP -> continue.

Output is a stream of events for the client:
    {"type": "think", "text": delta}      reasoning inside <think>...</think>
    {"type": "text", "text": delta}       visible answer
    {"type": "tool", "name", "arguments", "result"}
The <think>/<tool_call> markers are single special tokens, so the stream is split by token id.
"""

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from app.engine import ChatFormat, LoadedModel, stream_tokens

ToolRunner = Callable[[str, dict], str]
REPLACEMENT_CHAR = "�"


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """JSON body of a <tool_call> -> (name, arguments); malformed -> None."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        return None
    arguments = payload.get("arguments", {})
    return (payload["name"], arguments) if isinstance(arguments, dict) else None


@dataclass
class Reply:
    """What gets stored for one assistant message."""

    content: str = ""
    thinking: str = ""
    tool_calls: list[dict] = field(default_factory=list)


class _Channel:
    """Decodes one run of tokens incrementally; holds back output while a UTF-8 char is incomplete."""

    def __init__(self, fmt: ChatFormat) -> None:
        self.fmt, self.ids, self.sent = fmt, [], 0

    def push(self, token: int) -> str:
        self.ids.append(token)
        text = self.fmt.decode(self.ids)
        if text.endswith(REPLACEMENT_CHAR):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta

    def text(self) -> str:
        return self.fmt.decode(self.ids)


def run_turn(
    loaded: LoadedModel,
    fmt: ChatFormat,
    prompt_ids: list[int],
    *,
    thinking: bool,
    max_new: int,
    temperature: float,
    top_k: int | None,
    run_tool: ToolRunner | None,
    max_tool_turns: int,
    reply: Reply,
) -> Iterator[dict]:
    """Stream one assistant reply (with tool rounds). Fills `reply` as it goes."""
    ids = list(prompt_ids)
    for round_index in range(max_tool_turns + 1):
        mode = "think" if thinking and round_index == 0 else "text"
        channel, call_ids, in_call, stopped = _Channel(fmt), None, False, False
        budget = min(max_new, loaded.max_len - len(ids))
        new_ids: list[int] = []
        for token in stream_tokens(loaded.model, ids, budget, temperature, top_k, fmt.im_end_id):
            new_ids.append(token)
            if token == fmt.im_end_id:
                stopped = True
            elif token == fmt.think_start_id:
                mode, channel = "think", _Channel(fmt)
            elif token == fmt.think_end_id:
                mode, channel = "text", _Channel(fmt)
            elif token == fmt.call_start_id:
                call_ids, in_call = [], True
            elif token == fmt.call_end_id:
                in_call = False
            elif in_call:
                call_ids.append(token)
            else:
                delta = channel.push(token)
                if delta:
                    if mode == "think":
                        reply.thinking += delta
                    else:
                        reply.content += delta
                    yield {"type": mode, "text": delta}
        ids += new_ids

        call = parse_tool_call(fmt.decode(call_ids)) if call_ids else None
        if call is None or run_tool is None or not stopped or round_index == max_tool_turns:
            return
        name, arguments = call
        result = run_tool(name, arguments)
        reply.tool_calls.append({"name": name, "arguments": arguments, "result": result})
        yield {"type": "tool", "name": name, "arguments": arguments, "result": result}
        ids += fmt.after_tool(result)
        if len(ids) >= loaded.max_len:
            return


def run_completion(
    loaded: LoadedModel, fmt: ChatFormat, text: str, *, max_new: int, temperature: float, top_k: int | None,
    reply: Reply,
) -> Iterator[dict]:
    """Base (pretrain) model: continue plain text until <|eos|>."""
    ids = [fmt.bos_id] + fmt.encode(text)
    ids = ids[-(loaded.max_len - max_new):] if len(ids) + max_new > loaded.max_len else ids
    channel = _Channel(fmt)
    for token in stream_tokens(loaded.model, ids, min(max_new, loaded.max_len - len(ids)), temperature, top_k,
                               fmt.eos_id):
        if token == fmt.eos_id:
            return
        delta = channel.push(token)
        if delta:
            reply.content += delta
            yield {"type": "text", "text": delta}
