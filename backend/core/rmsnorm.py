import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Mục đích:
        Đưa độ lớn (RMS) của mỗi vector ẩn về ~1 rồi co giãn bằng trọng số học được.
        Khác LayerNorm: không trừ trung bình, không có bias → rẻ hơn, chất lượng tương đương.
    Shape:
        x: (..., d) -> (..., d)
    Công thức:
        RMS(x) = sqrt(mean(x_i^2 trên chiều cuối) + eps)
        y = (x / RMS(x)) * weight
    Độ phức tạp:
        Thời gian O(N·d), bộ nhớ O(N·d), với N là số token.
    Ổn định số học:
        - Tính trong fp32: fp16 tối đa ~65504 nên |x| > 256 thì x^2 tràn thành inf.
        - eps nằm trong căn, chống chia cho 0 khi x ≈ 0.

    Flow:
    x = [x1, x2, ..., xd]

        │
        ▼
     x ** 2

        │
        ▼
 mean(x ** 2)

        │
        ▼
 + eps

        │
        ▼
     rsqrt()
 = 1 / sqrt(...)

        │
        ▼
x * rsqrt(...)

        │
        ▼
 normalized x

        │
        ▼
normalized x * weight

        │
        ▼
      output
    """

    def __init__(self, dim: int, eps: float) -> None:
        """
        Args:
            dim: Kích thước chiều cuối d của đầu vào.
            eps: Hằng số nhỏ (> 0) cộng vào trong căn.
        """
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Chuẩn hoá x theo chiều cuối. Đầu ra cùng shape và dtype với x."""

        input_dtype = x.dtype # lưu dtype gốc của x
        x_float = x.float() # ép x sang fp32
        mean_square = x_float.pow(2).mean(dim=-1, keepdim=True) # tính mean(x^2) theo chiều cuối, giữ chiều (keepdim)
        x_normed = x_float * torch.rsqrt(mean_square + self.eps)
        return (x_normed * self.weight).to(input_dtype) # Nhân với weight rồi ép kết quả về dtype gốc rồi return

if __name__ == '__main__':
    torch.manual_seed(42)
    # 1. Tạo RMSNorm
    dim = 4
    eps = 1e-6

    norm = RMSNorm(dim=dim, eps=eps)

    print("=== RMSNorm ===")
    print(norm)
    print("weight:", norm.weight)
    print()

    # 2. Tạo input
    x = torch.tensor(
        [
            [3.0, 4.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=torch.float32
    )

    print("=== Input ===")
    print(x)
    print("shape:", x.shape)
    print("dtype:", x.dtype)
    print()

    # 3. Chạy RMSNorm
    y = norm(x)

    print("=== Output ===")
    print(y)
    print("shape:", y.shape)
    print("dtype:", y.dtype)
    print()

    # 4. Tự tính RMS của input
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_square + eps)

    print("=== Internal calculation ===")
    print("x^2:")
    print(x.pow(2))

    print("\nmean(x^2):")
    print(mean_square)

    print("\nrms:")
    print(rms)

    print("\nrsqrt(mean_square + eps):")
    print(torch.rsqrt(mean_square + eps))

    # 5. Kiểm tra RMS của output
    output_rms = torch.sqrt(
        y.pow(2).mean(dim=-1)
    )

    print("\n=== Output RMS ===")
    print(output_rms)

