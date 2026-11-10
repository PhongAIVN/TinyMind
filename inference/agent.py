"""Vòng lặp dùng tool lúc sinh chữ: model gọi tool -> chạy tool -> nối kết quả -> model viết tiếp.

Token nối thêm (kết quả tool + header assistant) dựng bằng đúng ChatEncoder lúc train,
nên chuỗi lúc chạy trùng từng token với hội thoại mẫu trong dữ liệu.
"""

from collections.abc import Callable

import torch

from inference.generate import generate
from inference.tools import find_tool_call, run_tool
from trainer.sft_dataset import TRAINED_ROLE, ChatEncoder

MAX_TOOL_TURNS = 3
# sample_fn(ids) -> các token model viết tiếp (kết thúc bằng <|im_end|> nếu model tự dừng)
SampleFn = Callable[[list[int]], list[int]]


def tool_result_tokens(encoder: ChatEncoder, result: str) -> list[int]:
    """Phần nối sau lượt assistant gọi tool: xuống dòng, tin nhắn tool, header assistant."""
    tool_ids, _ = encoder.encode_turn("tool", result)
    return encoder.separator + tool_ids + encoder.headers[TRAINED_ROLE]


def run_agent(
    encoder: ChatEncoder, prompt_ids: list[int], sample_fn: SampleFn, max_tool_turns: int = MAX_TOOL_TURNS
) -> tuple[list[int], list[bool], list[str]]:
    """Chạy tới khi model trả lời mà không gọi tool (hoặc hết lượt).

    Trả về (ids đầy đủ, generated[i] = True nếu ids[i] do model sinh, danh sách kết quả tool).
    generated = False cho prompt, kết quả tool, header: RL phải bỏ qua các token này.
    """
    ids, generated, results = list(prompt_ids), [False] * len(prompt_ids), []
    for turn in range(max_tool_turns + 1):
        new_tokens = sample_fn(ids)
        ids += new_tokens
        generated += [True] * len(new_tokens)
        text = encoder.tokenizer.decode(new_tokens, skip_special_tokens=False)
        call = find_tool_call(text)
        stopped = bool(new_tokens) and new_tokens[-1] == encoder.im_end_id
        if call is None or not stopped or turn == max_tool_turns:
            break
        result = run_tool(*call)
        results.append(result)
        appended = tool_result_tokens(encoder, result)
        ids += appended
        generated += [False] * len(appended)
    return ids, generated, results


def final_answer(encoder: ChatEncoder, ids: list[int], generated: list[bool]) -> str:
    """Lượt assistant cuối cùng (sau kết quả tool cuối, nếu có)."""
    last_start = max(i for i, flag in enumerate(generated) if not flag) + 1
    return encoder.tokenizer.decode(ids[last_start:])


def greedy_sampler(model, encoder: ChatEncoder, max_new_tokens: int) -> SampleFn:
    device = next(model.parameters()).device  # model có thể nằm trên GPU

    def sample(ids: list[int]) -> list[int]:
        input_ids = torch.tensor([ids], device=device)
        output = generate(model, input_ids, max_new_tokens, temperature=0.0, eos_id=encoder.im_end_id)
        return output[0, len(ids):].tolist()

    return sample
