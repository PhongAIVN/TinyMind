import torch
import torch.nn as nn

from core.block import Block
from core.config import ModelConfig
from core.embedding import TokenEmbedding
from core.rmsnorm import RMSNorm
from core.rope import precompute_rope

INIT_STD = 0.02


class Model(nn.Module):
    """Thân Transformer: input_ids -> hidden states.

    Shape:
        input_ids: (B, T)  ->  hidden: (B, T, d)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = TokenEmbedding(config.vocab_size, config.d_model, INIT_STD)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        cos, sin = precompute_rope(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ):
        batch, seq_len = input_ids.shape
        past_len = past_kvs[0][0].shape[-2] if past_kvs else 0
        if past_len + seq_len > self.config.max_seq_len:
            raise ValueError(
                f"past_len + seq_len = {past_len + seq_len} > max_seq_len = {self.config.max_seq_len}"
            )
        cos = self.rope_cos[past_len : past_len + seq_len]
        sin = self.rope_sin[past_len : past_len + seq_len]

        x = self.embed(input_ids)
        presents = []
        for layer, past_kv in zip(self.layers, past_kvs or [None] * len(self.layers)):
            out = layer(x, cos, sin, past_kv, use_cache)
            if use_cache:
                x, present = out
                presents.append(present)
            else:
                x = out

        hidden = self.final_norm(x)
        return (hidden, presents) if use_cache else hidden


class CausalLM(nn.Module):
    """Thân + đầu: input_ids -> logits.

    Shape:
        input_ids: (B, T)  ->  logits: (B, T, V)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Model(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(_init_weights)
        if config.tie_embedding:
            self.lm_head.weight = self.model.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ):
        out = self.model(input_ids, past_kvs, use_cache)
        if use_cache:
            hidden, presents = out
            return self.lm_head(hidden), presents
        return self.lm_head(out)

def _init_weights(module: nn.Module) -> None:
    """Khởi tạo mọi nn.Linear bằng N(0, INIT_STD²)."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, INIT_STD)
