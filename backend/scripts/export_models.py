"""Convert training checkpoints (.pt) into backend/models/<name>/model.safetensors + config.json.

Run from backend/:  python -m scripts.export_models --source ../checkpoints/50m
Each export is reloaded and compared logit-for-logit with the original before moving on.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_model, save_model

from app.settings import MODELS_DIR
from core.config import ModelConfig
from core.model import CausalLM

VARIANTS = ("pretrain", "sft", "lora_merged", "tool", "think", "cispo", "student")
PROBE_SHAPE = (2, 32)


def load_checkpoint(path: Path) -> CausalLM:
    # weights_only=False: the payload holds config dicts; only use on checkpoints you trained yourself.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = CausalLM(ModelConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model"])
    return model.eval()


def export(model: CausalLM, folder: Path, source: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    # save_model drops the duplicate tied tensor (embed = lm_head) instead of raising.
    save_model(model, str(folder / "model.safetensors"), metadata={"format": "pt", "source": source})
    (folder / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2) + "\n", encoding="utf-8")

    fresh = CausalLM(model.config).eval()
    load_model(fresh, str(folder / "model.safetensors"))
    probe = torch.randint(0, model.config.vocab_size, PROBE_SHAPE)
    with torch.no_grad():
        assert torch.equal(model(probe), fresh(probe)), f"{folder.name}: logits differ after export"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../checkpoints/50m"))
    args = parser.parse_args()
    for name in VARIANTS:
        path = args.source / f"{name}.pt"
        if not path.exists():
            print(f"skip {name}: {path} not found")
            continue
        export(load_checkpoint(path), MODELS_DIR / name, path.name)
        size_mb = (MODELS_DIR / name / "model.safetensors").stat().st_size / 2**20
        print(f"{name:12s} {size_mb:7.1f} MB  ok")


if __name__ == "__main__":
    main()
