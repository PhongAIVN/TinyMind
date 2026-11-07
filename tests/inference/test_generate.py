import pytest
import torch

from core.model import CausalLM
from inference.generate import generate, sample_next_token

PROMPT_LEN = 4
NEW_TOKENS = 6


class FixedLogitsModel(torch.nn.Module):
    """Model giả luôn trả cùng một bộ logits — để kiểm tra phần điều khiển vòng lặp."""

    def __init__(self, vocab_size: int, favourite: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.favourite = favourite
        self.calls = 0

    def forward(self, input_ids, past_kvs=None, use_cache=False):
        self.calls += 1
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size)
        logits[..., self.favourite] = 10.0
        if use_cache:
            return logits, [(torch.zeros(batch, 1, seq_len, 1), torch.zeros(batch, 1, seq_len, 1))]
        return logits


# ---------------------------------------------------------------- sample_next_token


def test_zero_temperature_is_argmax():
    logits = torch.tensor([[1.0, 5.0, 2.0], [9.0, 0.0, 0.0]])

    assert sample_next_token(logits, temperature=0.0).tolist() == [[1], [0]]


def test_top_k_one_equals_argmax():
    logits = torch.randn(4, 50)

    greedy = sample_next_token(logits, temperature=0.0)
    top1 = sample_next_token(logits, temperature=1.0, top_k=1)

    assert torch.equal(greedy, top1)


def test_top_k_never_picks_outside_top_k():
    logits = torch.randn(1, 100)
    allowed = set(logits.topk(5, dim=-1).indices[0].tolist())

    picks = {sample_next_token(logits, temperature=1.0, top_k=5).item() for _ in range(50)}

    assert picks <= allowed


def test_same_generator_seed_gives_same_token():
    logits = torch.randn(2, 20)

    first = sample_next_token(logits, generator=torch.Generator().manual_seed(0))
    second = sample_next_token(logits, generator=torch.Generator().manual_seed(0))

    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("kwargs", [{"temperature": -1.0}, {"top_k": 0}])
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        sample_next_token(torch.randn(1, 10), **kwargs)


# ---------------------------------------------------------------- generate


def test_output_grows_by_max_new_tokens(tiny_config):
    model = CausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, PROMPT_LEN))

    output = generate(model, input_ids, max_new_tokens=NEW_TOKENS, temperature=0.0)

    assert output.shape == (2, PROMPT_LEN + NEW_TOKENS)
    torch.testing.assert_close(output[:, :PROMPT_LEN], input_ids)


def test_cache_and_no_cache_give_same_tokens(tiny_config):
    """Tiêu chí xong chặng 3: sinh từng bước có cache = forward cả chuỗi."""
    model = CausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, PROMPT_LEN))

    with_cache = generate(model, input_ids, NEW_TOKENS, temperature=0.0, use_cache=True)
    without_cache = generate(model, input_ids, NEW_TOKENS, temperature=0.0, use_cache=False)

    torch.testing.assert_close(with_cache, without_cache)


def test_greedy_is_deterministic(tiny_config):
    model = CausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, PROMPT_LEN))

    first = generate(model, input_ids, NEW_TOKENS, temperature=0.0)
    second = generate(model, input_ids, NEW_TOKENS, temperature=0.0)

    torch.testing.assert_close(first, second)


def test_zero_new_tokens_returns_prompt(tiny_config):
    model = CausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, PROMPT_LEN))

    torch.testing.assert_close(generate(model, input_ids, 0), input_ids)


def test_stops_early_at_eos():
    eos_id = 3
    model = FixedLogitsModel(vocab_size=10, favourite=eos_id)
    input_ids = torch.zeros(1, 2, dtype=torch.long)

    output = generate(model, input_ids, max_new_tokens=20, temperature=0.0, eos_id=eos_id)

    assert output.shape == (1, 3)  # prompt 2 token + đúng 1 token eos
    assert output[0, -1].item() == eos_id


def test_finished_sequences_keep_padding_with_eos():
    """Chuỗi nào ra eos trước thì các bước sau chỉ thêm eos, không sinh chữ khác."""
    eos_id = 3
    model = FixedLogitsModel(vocab_size=10, favourite=eos_id)
    input_ids = torch.zeros(2, 2, dtype=torch.long)

    output = generate(model, input_ids, max_new_tokens=5, temperature=0.0, eos_id=eos_id)

    assert (output[:, 2:] == eos_id).all()


def test_cache_makes_model_see_one_token_per_step(tiny_config):
    """Có cache: mỗi bước chỉ đưa 1 token mới vào model."""
    model = FixedLogitsModel(vocab_size=tiny_config.vocab_size, favourite=1)
    input_ids = torch.zeros(1, PROMPT_LEN, dtype=torch.long)

    generate(model, input_ids, max_new_tokens=NEW_TOKENS, temperature=0.0, use_cache=True)

    assert model.calls == NEW_TOKENS


def test_generation_beyond_max_seq_len_raises(tiny_config):
    """Prompt dài đúng max_seq_len: sinh 1 token vẫn được, token thứ 2 thì vượt giới hạn."""
    model = CausalLM(tiny_config)
    input_ids = torch.zeros(1, tiny_config.max_seq_len, dtype=torch.long)

    generate(model, input_ids, max_new_tokens=1, temperature=0.0)

    with pytest.raises(ValueError, match="max_seq_len"):
        generate(model, input_ids, max_new_tokens=2, temperature=0.0)