"""Bộ điều phối: thứ tự bước, bỏ qua bước đã xong, tự --resume khi có checkpoint dở."""

from pathlib import Path

import pytest

from scripts import train_all
from scripts.train_all import STAGES, Stage, command_for, select_stages


def test_stage_order_follows_the_roadmap():
    names = [stage.name for stage in STAGES]
    order = ["pretrain", "sft", "lora", "dpo", "tool", "think", "distill", "grpo", "cispo", "ppo", "agentic", "eval"]
    assert [name for name in names if name in order] == order
    assert names.index("pretrain_data") < names.index("pretrain") < names.index("sft_data") < names.index("sft")


def test_every_config_in_commands_exists():
    for stage in STAGES:
        for index, part in enumerate(stage.command):
            if part.endswith(".json"):
                assert Path(part).exists(), f"{stage.name}: thiếu {part}"


def test_select_from_and_only():
    assert [s.name for s in select_stages(None, "ppo")] == ["ppo", "agentic", "eval"]
    assert [s.name for s in select_stages(["sft", "dpo"], None)] == ["sft", "dpo"]


def test_resume_added_only_for_unfinished_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(train_all, "MARKER_DIR", tmp_path / "done")
    checkpoint = tmp_path / "x.pt"
    stage = Stage("x", ["python", "-m", "trainer.x"], checkpoint)
    assert "--resume" not in command_for(stage)
    checkpoint.write_text("")
    assert command_for(stage)[-1] == "--resume"
    (tmp_path / "done").mkdir()
    (tmp_path / "done" / "x.done").write_text("")
    assert "--resume" not in command_for(stage)
