# BPE - Byte Pair Encoding
from collections import Counter

N_BYTES = 256  # số giá trị 1 byte có thể nhận: 0..255


def count_pairs(ids: list[int]) -> Counter:
    """Đếm số lần mỗi cặp token liền kề xuất hiện.

    Ví dụ: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    return Counter(zip(ids, ids[1:]))


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Thay mọi lần xuất hiện của `pair` trong `ids` bằng `new_id` (không chồng lấn).

    Ví dụ: merge([1, 2, 3, 1, 2], (1, 2), 99) -> [99, 3, 99]
            merge([7, 7, 7], (7, 7), 99)      -> [99, 7]
    """
    merged = []
    i = 0

    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2  # nhảy qua cả 2 phần tử
        else:
            merged.append(ids[i])
            i += 1

    return merged

class ByteBPE:
    """Byte-level BPE tự viết — để hiểu thuật toán, không dùng cho dữ liệu lớn.

    Độ phức tạp:
        train: O(số_lần_gộp × số_byte_văn_bản) — rất chậm với văn bản lớn.
    """

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        """Đưa tokenizer về trạng thái chưa train: 0 luật gộp, vocab = 256 byte."""
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(N_BYTES)}

    def train(self, text: str, vocab_size: int) -> None:
        """Học (vocab_size - 256) luật gộp từ `text`.

        Mỗi lần gọi train đều học lại từ đầu, bỏ kết quả của lần train trước.
        """
        if vocab_size < N_BYTES:
            raise ValueError("`vocab_size` phải lớn hơn`N_BYTES`")

        # Xoá luật cũ: nếu giữ lại, new_id bắt đầu lại từ 256 sẽ ghi đè vocab
        # trong khi merges vẫn còn luật cũ trỏ tới id đó → decode ra sai chữ.
        self._reset()

        ids = list(text.encode("utf-8"))

        for new_id in range(N_BYTES, vocab_size):
            counts = count_pairs(ids)
            if not counts:
                break
            best_pair = max(counts, key=counts.get)
            ids = merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = (self.vocab[best_pair[0]] + self.vocab[best_pair[1]])


    def encode(self, text: str) -> list[int]:
        """chữ -> list token id."""
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            counts = count_pairs(ids)

            pair = min(
                counts,
                key=lambda p: self.merges.get(p, float("inf"))
            )

            if pair not in self.merges:
                break

            ids = merge(ids, pair, self.merges[pair])

        return ids

    def decode(self, ids: list[int]) -> str:
        """list token id -> chữ."""
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

if __name__ == "__main__":
    # 1. Tạo một tokenizer BPE
    test = ByteBPE()

    # 2. Xem vocab ban đầu
    print("=== VOCAB BAN ĐẦU ===")
    print("Số lượng token:", len(test.vocab))
    print("Token 97:", test.vocab[97])
    print("Token 98:", test.vocab[98])

    # 3. Train trên một text rất đơn giản
    text = "abababab"
    test.train(text, vocab_size=258)

    # 4. Xem các merge rule đã học
    print("\n=== MERGES ===")
    print(test.merges)

    # 5. Xem token mới được tạo ra
    print("\n=== VOCAB MỚI ===")
    for token_id in range(256, 258):
        print(token_id, "->", test.vocab[token_id])

    # 6. Encode text
    print("\n=== ENCODE ===")
    ids = test.encode("abab")
    print('"abab" ->', ids)

    # 7. Decode ngược lại
    print("\n=== DECODE ===")
    text_decoded = test.decode(ids)
    print(ids, "->", text_decoded)
