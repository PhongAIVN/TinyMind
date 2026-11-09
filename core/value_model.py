"""Critic cho PPO: thân Transformer (như Model) + đầu value d -> 1."""

import torch
import torch.nn as nn

from core.config import ModelConfig
from core.model import Model


class ValueModel(nn.Module):
    """Ước lượng giá trị V(s_t) tại mỗi vị trí.

    Shape:
        input_ids: (B, T)  ->  values: (B, T)
    Lưu ý:
        Đầu value khởi tạo 0 -> lúc đầu V = 0 ở mọi vị trí, advantage = reward thuần.
        Thân nạp từ policy bằng load_body (key "model.*" giống CausalLM).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Model(config)
        self.value_head = nn.Linear(config.d_model, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.model(input_ids)).squeeze(-1)

    def load_body(self, causal_lm_state: dict[str, torch.Tensor]) -> None:
        """Nạp thân từ state_dict của CausalLM (bỏ lm_head)."""
        body = {key[len("model."):]: value for key, value in causal_lm_state.items() if key.startswith("model.")}
        self.model.load_state_dict(body)
