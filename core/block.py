import torch
import torch.nn as nn

from core.attention import CausalSelfAttention
from core.config import ModelConfig
from core.feedforward import FeedForward
from core.rmsnorm import RMSNorm


class Block(nn.Module):
    """Một tầng Transformer kiểu pre-norm.

    Shape:
        x: (B, T, d), cos/sin: (T, hd // 2)  ->  (B, T, d)
    Công thức:
        x = x + Attention(RMSNorm(x))
        x = x + FeedForward(RMSNorm(x))
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = FeedForward(config.d_model, config.ffn_hidden)

    def forward(
            self,
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
            use_cache: bool = False,
    ):
        attn_out = self.attn(self.attn_norm(x), cos, sin, past_kv, use_cache)
        if use_cache:
            attn_out, present = attn_out

        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return (x, present) if use_cache else x