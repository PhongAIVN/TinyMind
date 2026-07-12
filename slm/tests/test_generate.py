"""Tests for the generation loop."""

import torch

from slm.inference.generate import generate, generate_stream
from slm.model.transformer import Transformer
from slm.utils.config import ModelConfig


def test_generate_length(tiny_config: ModelConfig) -> None:
    model = Transformer(tiny_config)
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 5))
    output = generate(model, prompt, max_new_tokens=8, temperature=1.0)
    assert output.shape == (1, 13)
    torch.testing.assert_close(output[:, :5], prompt)


def test_generate_stops_at_context_length(tiny_config: ModelConfig) -> None:
    model = Transformer(tiny_config)
    prompt_length = tiny_config.context_length - 3
    prompt = torch.randint(0, tiny_config.vocab_size, (1, prompt_length))
    output = generate(model, prompt, max_new_tokens=100, temperature=1.0)
    assert output.shape[1] <= tiny_config.context_length


def test_greedy_generation_is_deterministic(tiny_config: ModelConfig) -> None:
    model = Transformer(tiny_config)
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))
    first = generate(model, prompt, max_new_tokens=6, temperature=0.0)
    second = generate(model, prompt, max_new_tokens=6, temperature=0.0)
    torch.testing.assert_close(first, second)


def test_stream_yields_one_token_at_a_time(tiny_config: ModelConfig) -> None:
    model = Transformer(tiny_config)
    prompt = torch.randint(0, tiny_config.vocab_size, (2, 3))
    steps = list(generate_stream(model, prompt, max_new_tokens=5, temperature=1.0))
    assert len(steps) == 5
    for token in steps:
        assert token.shape == (2, 1)


def test_batched_generation(tiny_config: ModelConfig) -> None:
    model = Transformer(tiny_config)
    prompt = torch.randint(0, tiny_config.vocab_size, (3, 4))
    output = generate(model, prompt, max_new_tokens=5, temperature=0.0)
    assert output.shape == (3, 9)
