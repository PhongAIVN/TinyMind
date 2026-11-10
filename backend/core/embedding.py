import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Bảng tra: token id -> vector.

    Mục đích:
        Đổi mỗi token id thành vector d chiều học được.
    Shape:
        input_ids: (B, T) long  ->  (B, T, d)
    Công thức:
        out[b, t] = weight[input_ids[b, t]]
    Độ phức tạp:
        O(B·T·d) thời gian; bảng O(V·d) bộ nhớ.
    Ổn định số học:
        Khởi tạo N(0, init_std²) với init_std nhỏ (0.02) để logits ban đầu gần 0.
    """

    def __init__(self, vocab_size: int, d_model: int, init_std: float) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model)) #torch.empty = cấp bộ nhớ, chưa gán giá trị
        nn.init.normal_(self.weight,mean=0.0, std=init_std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            raise ValueError(f"input_ids phải là torch.long, nhận {input_ids.dtype}")

        if input_ids.numel() >0 and (input_ids.min() < 0 or input_ids.max() >= self.vocab_size):
            raise ValueError(f"input_ids phải nằm trong [0, {self.vocab_size - 1}]")

        return self.weight[input_ids]

if __name__ == "__main__":
    # 1. Cấu hình
    vocab_size = 10
    d_model = 4
    init_std = 0.02

    # 2. Tạo TokenEmbedding
    embedding = TokenEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        init_std=init_std,
    )

    print("=== Embedding weight ===")
    print(embedding.weight)
    print("weight shape:", embedding.weight.shape)
    print()

    # 3. Input token IDs
    # B = 2, T = 5
    input_ids = torch.tensor([
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
    ], dtype=torch.long)

    print("=== Input ===")
    print(input_ids)
    print("input shape:", input_ids.shape)
    print("input dtype:", input_ids.dtype)
    print()

    # 4. Forward
    output = embedding(input_ids)

    print("=== Output ===")
    print(output)
    print("output shape:", output.shape)
    print("output dtype:", output.dtype)
    print()