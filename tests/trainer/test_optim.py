import torch

from core.model import CausalLM
from trainer.optim import build_optimizer, split_decay_params

LR = 1e-3
WEIGHT_DECAY = 0.1


def test_two_groups_with_and_without_decay(tiny_config):
    optimizer = build_optimizer(CausalLM(tiny_config), lr=LR, weight_decay=WEIGHT_DECAY)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == WEIGHT_DECAY
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_matrices_decay_vectors_do_not(tiny_config):
    decay, no_decay = split_decay_params(CausalLM(tiny_config))

    assert all(param.dim() >= 2 for param in decay)
    assert all(param.dim() == 1 for param in no_decay)


def test_every_trainable_parameter_is_covered(tiny_config):
    model = CausalLM(tiny_config)
    decay, no_decay = split_decay_params(model)

    assert sum(p.numel() for p in decay) + sum(p.numel() for p in no_decay) == sum(
        p.numel() for p in model.parameters()
    )


def test_norm_weights_are_in_no_decay_group(tiny_config):
    model = CausalLM(tiny_config)
    _, no_decay = split_decay_params(model)

    norm_weight = model.model.final_norm.weight
    assert any(param is norm_weight for param in no_decay)


def test_frozen_parameters_are_skipped(tiny_config):
    """Chuẩn bị cho LoRA: tham số đóng băng không vào optimizer."""
    model = CausalLM(tiny_config)
    model.model.embed.weight.requires_grad_(False)

    decay, no_decay = split_decay_params(model)

    assert all(param.requires_grad for param in decay + no_decay)


def test_optimizer_updates_weights(tiny_config):
    model = CausalLM(tiny_config)
    optimizer = build_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)
    before = model.model.final_norm.weight.detach().clone()

    model(torch.randint(0, tiny_config.vocab_size, (2, 4))).sum().backward()
    optimizer.step()

    assert not torch.allclose(before, model.model.final_norm.weight)

def test_fused_only_on_gpu(tiny_config):
    """CPU không hỗ trợ fused AdamW -> phải tắt; trên GPU mới bật."""
    from core.model import CausalLM

    optimizer = build_optimizer(CausalLM(tiny_config), lr=1e-3, weight_decay=0.1)

    assert not optimizer.defaults["fused"]
