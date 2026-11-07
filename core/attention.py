import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import ModelConfig
from core.rope import apply_rope


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention có RoPE.

    Mục đích:
        Mỗi token tổng hợp thông tin từ các token đứng trước nó.

    Shape:
        x: (B, T, d)
        cos/sin: (T, hd // 2)
        output: (B, T, d)

    Công thức:
        Attention(q, k, v)
            = softmax(q·kᵀ / √hd + mask) · v

    Độ phức tạp:
        O(B · h · T² · hd) — bậc 2 theo T.

    Ổn định số học:
        - Chia √hd để điểm không quá lớn.
        - Softmax tính bằng float32.
        - Mask dùng -inf.
        - Đường chéo không bị mask nên không hàng nào toàn -inf.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # Model dimensions
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        # d_model = n_heads * head_dim
        d_model = config.d_model

        # Q, K, V projections
        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.k_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.v_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        """Chạy causal self-attention.

        Args:
            x:
                Tensor shape (B, T, d_model)

            cos:
                Tensor shape (T, head_dim // 2)

            sin:
                Tensor shape (T, head_dim // 2)

        Returns:
            Tensor shape (B, T, d_model)
        """

        # 1. Lấy shape input
        batch, seq_len, d_model = x.shape

        # 2. Tạo Q, K, V
        #
        # x:
        #   [B, T, d_model]
        #
        # q/k/v:
        #   [B, T, d_model]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 3. Chia d_model thành nhiều attention heads
        #
        # [B, T, d_model]
        #
        #       ↓ view
        #
        # [B, T, n_heads, head_dim]
        #
        #       ↓ transpose
        #
        # [B, n_heads, T, head_dim]

        q = q.view(
            batch,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        # 4. Áp dụng RoPE cho Q và K
        #
        # Q → Q'
        # K → K'
        #
        # V KHÔNG áp dụng RoPE.
        #
        # Shape vẫn:
        # [B, n_heads, T, head_dim]

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.shape[-2]
            k = torch.cat([past_k, k], dim=-2)  # cache cũ đứng trước, token mới nối sau
            v = torch.cat([past_v, v], dim=-2)
        present = (k, v) if use_cache else None

        # 5. Tính attention scores
        #
        # q:
        #   [B, h, T, hd]
        #
        # k.transpose(-2, -1):
        #   [B, h, hd, T]
        #
        # q @ k.T:
        #   [B, h, T, T]
        #
        # Sau đó chia sqrt(head_dim)

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # 6. Causal mask
        #
        # Không cho token hiện tại nhìn thấy token tương lai.
        #
        # Ví dụ T = 4:
        #
        #   False True  True  True
        #   False False True  True
        #   False False False True
        #   False False False False
        #
        # True = bị mask

        future = torch.triu(
            torch.ones(
                seq_len,
                past_len + seq_len,  # có cache: mỗi query nhìn cả token cũ
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1 + past_len,  # ô cấm: j > i + past_len
        )

        scores = scores.masked_fill(
            future,
            float("-inf"),
        )

        # 7. Softmax
        #
        # scores:
        #   [B, h, T, T]
        #
        # Softmax theo chiều cuối:
        #   mỗi query phân phối attention
        #   lên các key.
        #
        # Tính softmax bằng FP32 để ổn định số học,
        # sau đó đưa kết quả về dtype của q.

        weights = F.softmax(
            scores.float(),
            dim=-1,
        ).to(q.dtype)

        # 8. Attention weights × V
        #
        # weights:
        #   [B, h, T, T]
        #
        # v:
        #   [B, h, T, hd]
        #
        # output:
        #   [B, h, T, hd]

        out = weights @ v

        # 9. Ghép các attention heads lại
        #
        # [B, h, T, hd]
        #
        #       ↓ transpose
        #
        # [B, T, h, hd]
        #
        #       ↓ reshape
        #
        # [B, T, d_model]

        out = out.transpose(1, 2).reshape(
            batch,
            seq_len,
            d_model,
        )

        # 10. Output projection
        #
        # [B, T, d_model]
        #       ↓
        # [B, T, d_model]

        out = self.o_proj(out)

        return (out, present) if use_cache else out
    