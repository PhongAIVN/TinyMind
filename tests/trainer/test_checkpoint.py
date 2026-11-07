import torch

from core.model import CausalLM
from trainer.checkpoint import load_checkpoint, load_weights, save_checkpoint
from trainer.optim import build_optimizer
from trainer.train_pretrain import CHECKPOINT_NAME, train


def make_model_and_optimizer(model_config):
    model = CausalLM(model_config)
    return model, build_optimizer(model, lr=1e-3, weight_decay=0.1)


def test_resume_matches_uninterrupted_run(tiny_model_config, make_train_config, tmp_path):
    """4 bước liền mạch phải ra đúng trọng số như 2 bước + resume 2 bước."""
    straight = make_train_config(max_steps=4, out_dir=str(tmp_path / "straight"))
    train(tiny_model_config, straight)

    first_half = make_train_config(max_steps=2, out_dir=str(tmp_path / "resumed"))
    train(tiny_model_config, first_half)
    second_half = make_train_config(max_steps=4, out_dir=str(tmp_path / "resumed"))
    train(tiny_model_config, second_half, resume=True)

    straight_state = torch.load(
        tmp_path / "straight" / CHECKPOINT_NAME, map_location="cpu", weights_only=False
    )["model"]
    resumed_state = torch.load(
        tmp_path / "resumed" / CHECKPOINT_NAME, map_location="cpu", weights_only=False
    )["model"]

    for key, value in straight_state.items():
        torch.testing.assert_close(resumed_state[key], value, msg=f"khác nhau ở {key}")


def test_saved_weights_are_float32(tiny_model_config, make_train_config, tmp_path):
    """Lưu fp16 như minimind sẽ làm resume không khớp."""
    model, optimizer = make_model_and_optimizer(tiny_model_config)
    model.half()
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, optimizer, 1, tiny_model_config, make_train_config())
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert all(tensor.dtype == torch.float32 for tensor in payload["model"].values())


def test_checkpoint_stores_step_and_configs(tiny_model_config, make_train_config, tmp_path):
    model, optimizer = make_model_and_optimizer(tiny_model_config)
    train_config = make_train_config()
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, optimizer, 7, tiny_model_config, train_config)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["step"] == 7
    assert payload["model_config"] == tiny_model_config.to_dict()
    assert payload["train_config"] == train_config.to_dict()


def test_load_checkpoint_restores_optimizer_state(tiny_model_config, make_train_config, tmp_path):
    model, optimizer = make_model_and_optimizer(tiny_model_config)
    model(torch.randint(0, tiny_model_config.vocab_size, (2, 4))).sum().backward()
    optimizer.step()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, 3, tiny_model_config, make_train_config())

    fresh_model, fresh_optimizer = make_model_and_optimizer(tiny_model_config)
    step = load_checkpoint(path, fresh_model, fresh_optimizer)

    assert step == 3

    saved_state = optimizer.state_dict()["state"]
    loaded_state = fresh_optimizer.state_dict()["state"]
    assert set(loaded_state) == set(saved_state)
    for index, entry in saved_state.items():
        torch.testing.assert_close(loaded_state[index]["exp_avg"], entry["exp_avg"])
        torch.testing.assert_close(loaded_state[index]["exp_avg_sq"], entry["exp_avg_sq"])


def test_random_state_is_restored(tiny_model_config, make_train_config, tmp_path):
    model, optimizer = make_model_and_optimizer(tiny_model_config)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, 1, tiny_model_config, make_train_config())

    expected = torch.rand(3)
    load_checkpoint(path, model, optimizer)

    torch.testing.assert_close(torch.rand(3), expected)


def test_load_weights_ignores_optimizer(tiny_model_config, make_train_config, tmp_path):
    """Bắt đầu giai đoạn mới: lấy trọng số, bỏ trạng thái optimizer."""
    model, optimizer = make_model_and_optimizer(tiny_model_config)
    model(torch.randint(0, tiny_model_config.vocab_size, (2, 4))).sum().backward()
    optimizer.step()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, 5, tiny_model_config, make_train_config())

    fresh_model, fresh_optimizer = make_model_and_optimizer(tiny_model_config)
    load_weights(path, fresh_model)

    for key, value in model.state_dict().items():
        torch.testing.assert_close(fresh_model.state_dict()[key], value.float())
    assert fresh_optimizer.state_dict()["state"] == {}


def test_atomic_write_leaves_no_temp_file(tiny_model_config, make_train_config, tmp_path):
    model, optimizer = make_model_and_optimizer(tiny_model_config)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, optimizer, 1, tiny_model_config, make_train_config())

    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))