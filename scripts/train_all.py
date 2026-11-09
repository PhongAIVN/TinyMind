"""Chạy trọn chuỗi train từ dữ liệu thô tới chặng cuối, theo thứ tự, mỗi bước một log.

    python -m scripts.train_all              # chạy mọi bước chưa xong
    python -m scripts.train_all --list       # xem các bước và trạng thái
    python -m scripts.train_all --only sft   # chỉ chạy một vài bước
    python -m scripts.train_all --from dpo   # chạy từ bước dpo trở đi
    python -m scripts.train_all --dry-run    # in lệnh, không chạy

Bước xong được đánh dấu bằng file <MARKER_DIR>/<bước>.done -> chạy lại thì bỏ qua.
Bước train bị ngắt giữa chừng mà đã có checkpoint -> tự thêm --resume.
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

CHECKPOINTS = Path("checkpoints/50m")
MARKER_DIR = CHECKPOINTS / "done"
LOG_DIR = Path("logs")
PY = [sys.executable, "-u", "-m"]
MODEL = ["--model-config", "configs/model_50m.json"]


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    checkpoint: Path | None = None  # có checkpoint mà chưa xong -> thêm --resume
    description: str = ""


STAGES = (
    Stage("pretrain_data", ["powershell", "-ExecutionPolicy", "Bypass", "-File", r"scripts\windows\build_pretrain_data.ps1"],
          description="dữ liệu thô -> jsonl -> train tokenizer -> train.bin/eval.bin"),
    Stage("pretrain", PY + ["trainer.train_pretrain", *MODEL, "--train-config", "configs/pretrain_50m.json"],
          CHECKPOINTS / "pretrain.pt", "pretrain 57M (khoảng 4,5 giờ trên RTX 5070)"),
    Stage("sft_data", PY + ["scripts.build_sft_data", "--skip-missing"], description="hỏi đáp + hội thoại -> dataset/sft"),
    Stage("sft", PY + ["trainer.train_sft", *MODEL, "--train-config", "configs/sft_50m.json"],
          CHECKPOINTS / "sft.pt", "SFT khuôn hội thoại"),
    Stage("toy_data", PY + ["scripts.build_toy_math"], description="thế giới đồ chơi: phép tính tiếng Việt"),
    Stage("toy_dpo_data", PY + ["scripts.build_toy_dpo"], description="cặp chosen/rejected"),
    Stage("toy_tools_data", PY + ["scripts.build_toy_tools"], description="hội thoại gọi máy tính"),
    Stage("toy_think_data", PY + ["scripts.build_toy_think"], description="bật/tắt <think>"),
    Stage("lora", PY + ["trainer.train_lora", *MODEL, "--train-config", "configs/lora_50m.json", "--rank", "8", "--alpha", "16"],
          CHECKPOINTS / "lora.pt", "LoRA r=8 trên phép tính, gộp ra lora_merged.pt"),
    Stage("dpo", PY + ["trainer.train_dpo", *MODEL, "--train-config", "configs/dpo_50m.json", "--beta", "0.1", "--sft-weight", "1.0"],
          CHECKPOINTS / "dpo.pt", "DPO + loss SFT trên chosen"),
    Stage("tool", PY + ["trainer.train_sft", *MODEL, "--train-config", "configs/tool_50m.json", "--checkpoint-name", "tool.pt"],
          CHECKPOINTS / "tool.pt", "Tool Use: gọi máy tính"),
    Stage("think", PY + ["trainer.train_sft", *MODEL, "--train-config", "configs/think_50m.json", "--checkpoint-name", "think.pt"],
          CHECKPOINTS / "think.pt", "tư duy thích ứng"),
    Stage("distill", PY + ["trainer.train_distill", "--teacher", str(CHECKPOINTS / "think.pt"),
                           "--model-config", "configs/model_student.json", "--train-config", "configs/distill_50m.json"],
          CHECKPOINTS / "student.pt", "chưng cất 57M -> 13M"),
    Stage("grpo", PY + ["trainer.train_grpo", *MODEL, "--rl-config", "configs/grpo_50m.json"], description="GRPO (bật tư duy)"),
    Stage("cispo", PY + ["trainer.train_grpo", *MODEL, "--rl-config", "configs/cispo_50m.json"], description="CISPO (bật tư duy)"),
    Stage("ppo", PY + ["trainer.train_ppo", *MODEL, "--rl-config", "configs/ppo_50m.json"], description="PPO + critic + GAE"),
    Stage("agentic", PY + ["trainer.train_grpo", *MODEL, "--rl-config", "configs/agentic_50m.json"],
          description="Agentic RL: GRPO có gọi máy tính"),
    Stage("eval", PY + ["scripts.eval_all"], description="chấm mọi checkpoint -> logs/eval_results.md"),
)
STAGE_NAMES = [stage.name for stage in STAGES]


def marker(stage: Stage) -> Path:
    return MARKER_DIR / f"{stage.name}.done"


def command_for(stage: Stage) -> list[str]:
    resume = stage.checkpoint is not None and stage.checkpoint.exists() and not marker(stage).exists()
    return stage.command + (["--resume"] if resume else [])


def run_logged(command: list[str], log_path: Path) -> int:
    """Chạy lệnh, in ra màn hình đồng thời ghi vào log. Trả về mã thoát."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n$ {' '.join(command)}\n")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                   text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return process.wait()


def select_stages(only: list[str] | None, start: str | None) -> list[Stage]:
    if only:
        return [stage for stage in STAGES if stage.name in only]
    if start:
        return list(STAGES[STAGE_NAMES.index(start):])
    return list(STAGES)


def main() -> None:
    parser = argparse.ArgumentParser(description="chạy trọn chuỗi train TinyMind")
    parser.add_argument("--only", nargs="+", choices=STAGE_NAMES)
    parser.add_argument("--from", dest="start", choices=STAGE_NAMES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true", help="chạy lại cả bước đã đánh dấu xong")
    args = parser.parse_args()

    stages = select_stages(args.only, args.start)
    if args.list:
        for stage in STAGES:
            state = "xong" if marker(stage).exists() else ("dở (sẽ --resume)" if stage.checkpoint and stage.checkpoint.exists() else "chưa")
            print(f"{stage.name:15s} [{state:16s}] {stage.description}")
        return
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    for stage in stages:
        if marker(stage).exists() and not args.force:
            print(f"[bỏ qua] {stage.name} (đã xong)")
            continue
        command = command_for(stage)
        print(f"\n=== {stage.name}: {stage.description}\n$ {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        started = time.time()
        code = run_logged(command, LOG_DIR / f"{stage.name}.log")
        minutes = (time.time() - started) / 60
        if code != 0:
            print(f"[LỖI] {stage.name} thoát mã {code} sau {minutes:.1f} phút. Xem logs/{stage.name}.log; sửa xong chạy lại lệnh cũ.")
            sys.exit(code)
        marker(stage).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        print(f"[xong] {stage.name} sau {minutes:.1f} phút")
    print("\nHoàn tất. Kết quả: logs/eval_results.md")


if __name__ == "__main__":
    main()
