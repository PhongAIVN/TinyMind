"""LoRA: lúc đầu không đổi đầu ra, gộp không đổi đầu ra, chỉ A/B được học."""

import pytest
import torch
import torch.nn as nn

from core.lora import LoRALinear, apply_lora, count_trainable, merge_lora
from core.model import CausalLM

RANK, ALPHA = 4, 8.0


@pytest.fixture
def model(tiny_config):
    torch.manual_seed(0)
    return CausalLM(tiny_config)


def randomize_lora(model):
    """B = 0 lúc đầu thì gộp hay không cũng như nhau: cho B ngẫu nhiên để test có ý nghĩa."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.normal_(std=0.1)


def ids_for(config):
    return torch.randint(0, config.vocab_size, (2, 8), generator=torch.Generator().manual_seed(1))


def test_initial_output_equals_base_model(model, tiny_config):
    ids = ids_for(tiny_config)
    before = model(ids)
    apply_lora(model, RANK, ALPHA)
    torch.testing.assert_close(model(ids), before)


def test_merge_keeps_output(model, tiny_config):
    ids = ids_for(tiny_config)
    apply_lora(model, RANK, ALPHA)
    randomize_lora(model)
    with_lora = model(ids)

    merge_lora(model)

    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    torch.testing.assert_close(model(ids), with_lora, atol=1e-5, rtol=1e-5)


def test_merged_state_dict_matches_original_keys(model):
    keys = set(model.state_dict())
    apply_lora(model, RANK, ALPHA)
    merge_lora(model)
    assert set(model.state_dict()) == keys


def test_only_lora_params_are_trainable(model, tiny_config):
    names = apply_lora(model, RANK, ALPHA, targets=("q_proj", "v_proj"))
    d = tiny_config.d_model
    trainable, _ = count_trainable(model)

    assert len(names) == 2 * tiny_config.n_layers
    assert trainable == len(names) * RANK * (d + d)
    assert all(("lora_" in n) == p.requires_grad for n, p in model.named_parameters())


def test_gradient_reaches_only_lora(model, tiny_config):
    apply_lora(model, RANK, ALPHA)
    randomize_lora(model)
    model(ids_for(tiny_config)).sum().backward()
    for name, param in model.named_parameters():
        assert (param.grad is not None) == ("lora_" in name), name


def test_delta_weight_uses_alpha_over_r():
    base = nn.Linear(6, 5, bias=False)
    layer = LoRALinear(base, r=2, alpha=6.0)
    with torch.no_grad():
        layer.lora_B.fill_(1.0)
    x = torch.randn(3, 6)
    expected = base(x) + 3.0 * (x @ layer.lora_A.t() @ layer.lora_B.t())
    torch.testing.assert_close(layer(x), expected)


def test_unknown_target_raises(model):
    with pytest.raises(ValueError, match="không có nn.Linear"):
        apply_lora(model, RANK, ALPHA, targets=("qproj",))


def test_lm_head_is_never_wrapped_by_default(model):
    names = apply_lora(model, RANK, ALPHA)
    assert all(name.split(".")[-1] in ("q_proj", "v_proj") for name in names)
