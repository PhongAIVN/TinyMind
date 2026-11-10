# RoPE = Rotary Positional Embedding

import torch

def precompute_rope(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Tính sẵn bảng cos, sin cho RoPE

    Mục đích:
        Tính 1 lần góc xoay cho mọi (vị trí, cặp chiều), dùng lại ở mọi layer.
    Shape:
        ra: cos, sin — mỗi cái (max_seq_len, head_dim // 2)
    Công thức:
        tần_số_i  = 1 / theta^(2i / head_dim),   i = 0 .. head_dim/2 - 1
        góc[m, i] = m × tần_số_i
    Độ phức tạp:
        O(max_seq_len × head_dim)
    Ổn định số học:
        Tính bằng float32.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim={head_dim} phải là số chẵn")

    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inv_freq = 1.0 / (theta ** exponents)
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Xoay q hoặc k theo vị trí.

        Shape:
            x: (B, h, T, hd), cos/sin: (T, hd // 2)  ->  (B, h, T, hd)
        Công thức:
            x1 = nửa đầu, x2 = nửa sau
            ra = [x1·cos − x2·sin,  x1·sin + x2·cos]
        """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.to(x.dtype)