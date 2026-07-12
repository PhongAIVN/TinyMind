"""AdamW optimizer construction with selective weight decay.

Purpose:
    Build the optimizer following standard LLM practice:

    - Weight decay applies only to matrices (linear/embedding weights).
      Decaying 1-D parameters (RMSNorm scales) pulls them toward zero,
      which fights their job of preserving activation scale.
    - betas = (0.9, 0.95): the lower beta2 (vs the 0.999 default) makes
      the second-moment estimate react faster, which is more stable for
      the noisy gradients of LM training.
"""

import torch

from slm.utils.config import TrainConfig


def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.AdamW:
    """Create AdamW with weight decay on >=2D parameters only.

    Args:
        model: The model whose parameters will be optimized.
        config: Training config providing lr, betas, and weight decay.

    Returns:
        Configured AdamW optimizer with two parameter groups.
    """
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    parameter_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        parameter_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
