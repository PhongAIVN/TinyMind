"""Load safetensors variants, build chat prompts, stream tokens.

Chat template (same as training):
    <|bos|><|im_start|>user\\n{text}<|im_end|>\\n<|im_start|>assistant\\n{answer}<|im_end|>\\n
Tool results go in a "tool" turn wrapped in <tool_response>...</tool_response>.
"""

import json
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_model
from tokenizers import Tokenizer

from app.settings import MODELS_DIR, TOKENIZER_PATH, VARIANTS
from core.config import ModelConfig
from core.model import CausalLM

ROLES = ("system", "user", "assistant", "tool")
THINK_ON_PREFILL = "<think>\n"
THINK_OFF_PREFILL = "<think>\n\n</think>\n"
MIN_ANSWER_TOKENS = 16
OLD_ANSWER_TOKENS = 60
SUMMARY_QUESTION_TOKENS = 24
ELLIPSIS = "…"
# Filler the SFT data taught the model (translated OpenOrca/Alpaca "As an AI language model..." replies).
BOILERPLATE = re.compile(
    r"xin lỗi|rất tiếc|(?:mô hình|người mẫu) ngôn ngữ|với tư cách là một (?:AI|trợ lý)|\[tên của bạn\]", re.IGNORECASE
)


class ChatFormat:
    """Token-level chat template. Mirrors the encoder used for SFT so prompts match training exactly."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.bos_id = self.special("<|bos|>")
        self.eos_id = self.special("<|eos|>")
        self.im_end_id = self.special("<|im_end|>")
        self.think_start_id = self.special("<think>")
        self.think_end_id = self.special("</think>")
        self.call_start_id = self.special("<tool_call>")
        self.call_end_id = self.special("</tool_call>")
        im_start_id = self.special("<|im_start|>")
        self.separator = self.encode("\n")
        self.headers = {role: [im_start_id] + self.encode(f"{role}\n") for role in ROLES}

    def special(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"tokenizer is missing {token}")
        return token_id

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def turn(self, role: str, content: str) -> list[int]:
        if role == "tool":
            content = f"<tool_response>{content}</tool_response>"
        return self.headers[role] + self.encode(content) + [self.im_end_id] + self.separator

    def prompt(self, messages: list[dict], prefill: str = "") -> list[int]:
        """Conversation up to the assistant header, plus an optional prefilled start of the answer."""
        ids = [self.bos_id]
        for message in messages:
            ids += self.turn(message["role"], message["content"])
        return ids + self.headers["assistant"] + (self.encode(prefill) if prefill else [])

    def after_tool(self, result: str) -> list[int]:
        """Tokens appended after an assistant turn that ended with a tool call."""
        return self.separator + self.turn("tool", result) + self.headers["assistant"]


def shorten(fmt: ChatFormat, text: str, max_tokens: int) -> str:
    """First max_tokens tokens of text, with an ellipsis if anything was cut."""
    ids = fmt.encode(text)
    return text if len(ids) <= max_tokens else fmt.decode(ids[:max_tokens]).rstrip() + ELLIPSIS


def drop_boilerplate(history: list[dict]) -> list[dict]:
    """Remove each assistant answer that is "xin lỗi / là một mô hình ngôn ngữ AI" filler, with its question.

    Left in the history, the 57M model copies the pattern into later answers: replaying 6 questions
    as one chat gave boilerplate in 9/24 answers with full history vs 3/24 with none.
    """
    kept: list[dict] = []
    for message in history:
        if message["role"] == "assistant" and BOILERPLATE.search(message["content"]):
            if kept and kept[-1]["role"] == "user":
                kept.pop()
            continue
        kept.append(message)
    return kept


def system_text(earlier_questions: list[str]) -> str:
    return "Trước đó người dùng đã hỏi: " + "; ".join(earlier_questions) + "." if earlier_questions else ""


def build_prompt(
    fmt: ChatFormat,
    history: list[dict],
    user_text: str,
    prefill: str,
    max_len: int,
    max_new: int,
) -> list[int]:
    """Fit history + the new question into max_len - max_new tokens.

    Old assistant answers are cut to OLD_ANSWER_TOKENS so more turns fit. When turns still have
    to go, the oldest are dropped but their questions stay as a one-line summary in the system turn.
    Shedding order: whole turns, then summary items (oldest first).
    User memories are deliberately NOT put in the prompt: with them the sft model gave the
    "là một mô hình ngôn ngữ AI" boilerplate 3x as often and mixed the facts into unrelated answers.
    Raises ValueError if the new question alone leaves fewer than MIN_ANSWER_TOKENS.
    """
    turns = [
        {**m, "content": shorten(fmt, m["content"], OLD_ANSWER_TOKENS)} if m["role"] == "assistant" else m
        for m in drop_boilerplate(history)
    ] + [{"role": "user", "content": user_text}]
    start, skip_summary = 0, 0
    while True:
        dropped = [shorten(fmt, m["content"], SUMMARY_QUESTION_TOKENS) for m in turns[:start] if m["role"] == "user"]
        system = system_text(dropped[skip_summary:])
        ids = fmt.prompt(([{"role": "system", "content": system}] if system else []) + turns[start:], prefill)
        if len(ids) + max_new <= max_len:
            return ids
        if start < len(turns) - 1:
            start += 1
        elif skip_summary < len(dropped):
            skip_summary += 1
        else:
            break
    if max_len - len(ids) < MIN_ANSWER_TOKENS:
        raise ValueError(f"Tin nhắn quá dài: {len(ids)} token, model chỉ đọc được {max_len}")
    return ids


def sample(logits: torch.Tensor, temperature: float, top_k: int | None, generator: torch.Generator | None) -> int:
    """logits (V,) -> token id. temperature 0 = greedy."""
    if temperature == 0:
        return int(logits.argmax())
    scores = logits.float() / temperature
    if top_k:
        kth = scores.topk(min(top_k, scores.numel())).values[-1]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    return int(torch.multinomial(F.softmax(scores, dim=-1), 1, generator=generator))


@torch.no_grad()
def stream_tokens(
    model: CausalLM, ids: list[int], max_new: int, temperature: float, top_k: int | None, stop_id: int
) -> Iterator[int]:
    """Yield new token ids one by one (KV cache). The stop token is yielded last, then iteration ends."""
    device = next(model.parameters()).device
    step_input = torch.tensor([ids], device=device)
    past = None
    for _ in range(max_new):
        logits, past = model(step_input, past_kvs=past, use_cache=True)
        token = sample(logits[0, -1], temperature, top_k, None)
        yield token
        if token == stop_id:
            return
        step_input = torch.tensor([[token]], device=device)


@dataclass
class LoadedModel:
    name: str
    model: CausalLM

    @property
    def max_len(self) -> int:
        return self.model.config.max_seq_len


class ModelStore:
    """Lazily loads variants from backend/models/<name>/ and keeps them in memory."""

    def __init__(self, models_dir: Path = MODELS_DIR, device: str | None = None) -> None:
        self.models_dir = models_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fmt = ChatFormat(Tokenizer.from_file(str(TOKENIZER_PATH)))
        self._loaded: dict[str, LoadedModel] = {}
        self._load_lock = threading.Lock()
        # One generation at a time: the models share one CPU/GPU.
        self.generate_lock = threading.Lock()

    def available(self) -> list[str]:
        return [name for name in VARIANTS if (self.models_dir / name / "model.safetensors").exists()]

    def get(self, name: str) -> LoadedModel:
        if name not in self.available():
            raise KeyError(name)
        with self._load_lock:
            if name not in self._loaded:
                folder = self.models_dir / name
                config = ModelConfig.from_dict(json.loads((folder / "config.json").read_text(encoding="utf-8")))
                if config.vocab_size != self.fmt.tokenizer.get_vocab_size():
                    raise ValueError(f"{name}: vocab {config.vocab_size} != tokenizer")
                model = CausalLM(config)
                # load_model (not load_state_dict) restores the tied embed = lm_head weight.
                load_model(model, str(folder / "model.safetensors"))
                self._loaded[name] = LoadedModel(name, model.to(self.device).eval())
            return self._loaded[name]
