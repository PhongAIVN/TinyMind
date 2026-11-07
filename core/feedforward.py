import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """SwiGLU feed-forward.

    Mục đích:
        Biến đổi phi tuyến từng token riêng lẻ.
    Shape:
        x: (B, T, d) -> (B, T, d)
    Công thức:
        FFN(x) = down( SiLU(gate(x)) ⊙ up(x) )
    Độ phức tạp:
        O(B·T·d·ffn)
    """

    def __init__(self, d_model: int, ffn_hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_hidden, bias=False)
        self.up_proj   = nn.Linear(d_model, ffn_hidden, bias=False)
        self.down_proj = nn.Linear(ffn_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)