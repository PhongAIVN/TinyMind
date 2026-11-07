# CODE_STEP — Từ chữ tiếng Việt đến model đã train

Hướng dẫn code **từ đầu đến cuối**: câu chữ → tokenizer → token ids → embedding → các tầng Transformer → logits → loss → **train**.

Làm **lần lượt từng file**, file nào xong (test xanh) mới sang file sau.

---

## Bảng tiến độ

Đánh dấu `[x]` khi xong.

|     | #  | File code                                              | File test                                   | Việc                                  |
| --- | -- | ------------------------------------------------------ | ------------------------------------------- | -------------------------------------- |
| [x] | 0  | `core/config.py`                                     | `tests/core/test_config.py`               | Các con số của model                |
|     |    | **PHẦN A — TOKENIZER: chữ → số**            |                                             |                                        |
| [ ] | A1 | `tokenizer/bpe_toy.py`                               | `tests/tokenizer/test_bpe_toy.py`         | Tự viết BPE để hiểu thuật toán  |
| [ ] | A2 | `tokenizer/train_tokenizer.py`                       | `tests/tokenizer/test_train_tokenizer.py` | Train tokenizer thật bằng thư viện |
|     |    | **PHẦN B — MODEL: số → logits**              |                                             |                                        |
| [ ] | B1 | `core/embedding.py`                                  | `tests/core/test_embedding.py`            | Token id → vector                     |
| [ ] | B2 | `core/rmsnorm.py`                                    | `tests/core/test_rmsnorm.py`              | Chuẩn hoá vector                     |
| [ ] | B3 | `core/rope.py`                                       | `tests/core/test_rope.py`                 | Gắn thông tin vị trí               |
| [ ] | B4 | `core/attention.py`                                  | `tests/core/test_attention.py`            | Token nhìn các token trước         |
| [ ] | B5 | `core/feedforward.py`                                | `tests/core/test_feedforward.py`          | Xử lý từng token                    |
| [ ] | B6 | `core/block.py`                                      | `tests/core/test_block.py`                | Ghép B2–B5 thành 1 tầng            |
| [ ] | B7 | `core/model.py`                                      | `tests/core/test_model.py`                | Ghép cả model                        |
|     |    | **PHẦN C — GHÉP LẠI**                        |                                             |                                        |
| [ ] | C1 | `scripts/text_to_logits.py`                          | (chạy tay)                                 | Gõ câu tiếng Việt → xem logits    |
|     |    | **PHẦN D — PRETRAIN: dạy model tiếng Việt** |                                             |                                        |
| [ ] | D1 | `scripts/prepare_pretrain.py`                        | `tests/trainer/test_prepare_pretrain.py`  | txt → token ids (.bin)                |
| [ ] | D2 | `trainer/pretrain_dataset.py`                        | `tests/trainer/test_pretrain_dataset.py`  | Cắt cửa sổ (x, y)                   |
| [ ] | D3 | `trainer/losses.py`                                  | `tests/trainer/test_losses.py`            | Cross-entropy                          |
| [ ] | D4 | `trainer/optim.py`                                   | `tests/trainer/test_optim.py`             | AdamW 2 nhóm decay                    |
| [ ] | D5 | `trainer/lr_scheduler.py`                            | `tests/trainer/test_lr_scheduler.py`      | Warmup + cosine                        |
| [ ] | D6 | `trainer/checkpoint.py`                              | `tests/trainer/test_checkpoint.py`        | Lưu / resume                          |
| [ ] | D7 | `trainer/metrics.py`                                 | `tests/trainer/test_metrics.py`           | loss, grad norm, tok/s                 |
| [ ] | D8 | `trainer/train_config.py`                            | `tests/trainer/test_train_config.py`      | Tham số một lần chạy               |
| [ ] | D9 | `trainer/train_pretrain.py`                          | `tests/trainer/test_train_pretrain.py`    | Vòng lặp train                       |
|     |    | **PHẦN E — SINH CHỮ: cho model viết tiếp**  |                                             |                                        |
| [ ] | E1 | `core/attention.py`, `block.py`, `model.py`      | `tests/core/test_kv_cache.py`             | Thêm KV cache                         |
| [ ] | E2 | `inference/generate.py`                              | `tests/inference/test_generate.py`        | Vòng lặp sinh token                  |
| [ ] | E3 | `scripts/windows/generate.bat`                       | (chạy tay)                                 | Bọc lệnh cho tiện                   |
|     |    | **PHẦN F — SFT: dạy model trả lời**         |                                             |                                        |
| [ ] | F1 | `toyworld/arithmetic.py`                             | `tests/toyworld/test_arithmetic.py`       | Thế giới đồ chơi                  |
| [ ] | F2 | `trainer/chat_template.py`                           | `tests/trainer/test_chat_template.py`     | Tin nhắn → token + nhãn             |
| [ ] | F3 | `scripts/prepare_sft.py`                             | (chạy tay)                                 | Sinh`dataset/sft/*.jsonl`            |
| [ ] | F4 | `trainer/sft_dataset.py`                             | `tests/trainer/test_sft_dataset.py`       | (x, y) có pad                         |
| [ ] | F5 | `trainer/sft_config.py` + `configs/sft_tiny.json`  | (trong test F6)                             | Tham số SFT                           |
| [ ] | F6 | `trainer/train_sft.py`                               | `tests/trainer/test_train_sft.py`         | Vòng lặp SFT                         |
| [ ] | F7 | `inference/chat.py`                                  | (trong test F6)                             | Chat + đo độ chính xác            |
| [ ] | F8 | `scripts/windows/train_sft.bat`, `chat.bat`        | (chạy tay)                                 | Bọc lệnh                             |

---

## Toàn cảnh

```
"Hà Nội là thủ đô của Việt Nam"                   ← chữ
   │
   ▼
┌─ TOKENIZER ─────────────────────────────┐        [PHẦN A]
│  chữ → byte UTF-8 → gộp cặp hay gặp (BPE)│        A1: tự viết  |  A2: thư viện
└──────────────────────────────────────────┘
   │
   ▼
input_ids  [1523, 88, 402, ...]   shape (B, T)     ← số nguyên
   │
   ▼
TokenEmbedding                                     [B1: embedding.py]
   │  (B, T, d)
   ▼
┌─ Block ─── lặp L lần ───────────────────┐        [B6: block.py]
│   RMSNorm                                │        [B2: rmsnorm.py]
│   Attention  (bên trong dùng RoPE)       │        [B4: attention.py] + [B3: rope.py]
│   + cộng lại đầu vào (residual)          │
│   RMSNorm                                │        [B2: rmsnorm.py]
│   FeedForward                            │        [B5: feedforward.py]
│   + cộng lại đầu vào (residual)          │
└──────────────────────────────────────────┘
   │  (B, T, d)
   ▼
RMSNorm cuối  →  hidden (B, T, d)                  [B7: model.py]
   │
   ▼
lm_head       →  logits (B, T, V)                  [B7: model.py]
   │
   ▼
loss = cross_entropy(logits[vị trí t], token[t+1]) [C1: text_to_logits.py]
```

**Ký hiệu:**

| Ký hiệu | Nghĩa                      | Trong config                             |
| --------- | --------------------------- | ---------------------------------------- |
| `B`     | số câu trong 1 batch      | —                                       |
| `T`     | số token mỗi câu         | ≤`max_seq_len`                        |
| `V`     | số token trong vocab       | `vocab_size` = số token tokenizer có |
| `d`     | độ dài vector ẩn        | `d_model`                              |
| `h`     | số head                    | `n_heads`                              |
| `hd`    | độ dài mỗi head = d / h | `head_dim`                             |
| `L`     | số Block                   | `n_layers`                             |

Sơ đồ chi tiết phần model: `images/model_architecture.drawio`.

---

## Chuẩn bị (làm 1 lần)

### 1. Tạo thư mục và file rỗng

```
tokenizer/__init__.py        ← file rỗng, để import được "tokenizer.bpe_toy"
tests/tokenizer/             ← thư mục chứa test tokenizer
```

### 2. Cài thư viện `tokenizers` (cần cho A2 và C1)

```powershell
pip install tokenizers
```

Thêm dòng `tokenizers` vào `requirements.txt`.

### 3. Tạo `tests/conftest.py`

Cho mọi test dùng chung config nhỏ `tiny_config` và cố định số ngẫu nhiên. pytest tự nạp file này.

```python
import pytest
import torch

from core.config import ModelConfig


@pytest.fixture(autouse=True)
def fixed_seed() -> None:
    """Chạy trước MỌI test: cố định số ngẫu nhiên để kết quả lần nào cũng giống nhau."""
    torch.manual_seed(0)


@pytest.fixture
def tiny_config() -> ModelConfig:
    """Config tí hon. Test nào cần thì thêm tham số `tiny_config`."""
    return ModelConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=4,
        ffn_hidden=128,
        max_seq_len=32,
        rope_theta=10000.0,
        norm_eps=1e-6,
        tie_embedding=True,
    )
```

### 4. Lệnh chạy test

Luôn chạy từ thư mục gốc project:

```powershell
python -m pytest tests/tokenizer/test_bpe_toy.py -v    # 1 file
python -m pytest tests -v                              # tất cả
```

### 5. Mỗi phần bên dưới có đúng 6 mục

1. **Vị trí** — nằm ở đâu trong luồng
2. **Làm gì** — giải thích
3. **Vào → Ra** — shape / ví dụ số
4. **Khung code** — gõ vào file, điền `TODO`
5. **Test** — gõ vào file test, chạy
6. **Xong khi** — điều kiện sang file sau

**Lời giải** và **code test** giấu trong mục thu gọn. **Tự làm trước**, kẹt mới mở.

---

---

# PHẦN A — TOKENIZER

Model chỉ làm việc với **số**. Tokenizer biến chữ thành dãy số (encode) và biến số ngược lại thành chữ (decode).

---

# A1 — `tokenizer/bpe_toy.py` (tự viết BPE)

## 1. Vị trí

```
▶"Hà Nội..." → TOKENIZER → input_ids◀ → Embedding → [Block] × L → RMSNorm → lm_head → logits
```

Bản **tự viết để hiểu**. Chậm, chỉ dùng cho câu ngắn. Bản thật dùng thư viện ở A2 — cùng thuật toán.

## 2. Làm gì

### Bước 1: chữ → byte

Máy tính lưu chữ bằng **UTF-8**: mỗi ký tự là 1–4 byte (số 0–255). Tiếng Việt có dấu thường tốn 2–3 byte:

```
"Việt"  →  V     i     ệ              t
           0x56  0x69  0xE1 0xBB 0x87  0x74       = 6 byte
```

Nếu dùng thẳng byte làm token: vocab chỉ có 256, **không bao giờ gặp chữ lạ**, nhưng chuỗi rất dài (1 chữ "ệ" = 3 token).

### Bước 2: gộp cặp hay gặp nhất (BPE — Byte Pair Encoding)

Lặp lại: đếm mọi **cặp token đứng cạnh nhau**, lấy cặp xuất hiện nhiều nhất, **gộp** thành 1 token mới (id 256, 257, ...).

Ví dụ với chữ cái cho dễ nhìn — văn bản `aaabdaaabac`:

```
Lần 1:  đếm cặp: aa=4, ab=2, bd=1, da=1, ba=1, ac=1
        gộp "aa" → Z            aaabdaaabac  →  ZabdZabac

Lần 2:  đếm cặp: Za=2, ab=2, bd=1, dZ=1, ba=1, ac=1
        gộp "Za" → Y (gặp trước) ZabdZabac   →  YbdYbac
```

Sau 2 lần: 11 token → 7 token. Luật gộp học được: `aa → Z`, `Za → Y`.

**Train** = chạy gộp `vocab_size − 256` lần, ghi lại các luật gộp.
**Encode** câu mới = đổi ra byte, rồi áp các luật gộp **theo đúng thứ tự đã học**.
**Decode** = mỗi token tra ra chuỗi byte, nối lại, giải mã UTF-8.

### Cấu trúc file

| Tên                                | Loại  | Làm gì                                       |
| ----------------------------------- | ------ | ---------------------------------------------- |
| `count_pairs(ids)`                | hàm   | Đếm số lần mỗi cặp liền kề xuất hiện |
| `merge(ids, pair, new_id)`        | hàm   | Thay mọi chỗ có`pair` bằng `new_id`    |
| `ByteBPE.train(text, vocab_size)` | method | Học luật gộp                                |
| `ByteBPE.encode(text)`            | method | chữ → list id                                |
| `ByteBPE.decode(ids)`             | method | list id → chữ                                |

Hai bảng bên trong `ByteBPE`:

- `merges: dict[(int, int), int]` — luật gộp, ví dụ `{(97, 97): 256}` nghĩa là cặp `(a, a)` → token 256
- `vocab: dict[int, bytes]` — mỗi token là chuỗi byte nào, ví dụ `{97: b"a", 256: b"aa"}`

## 3. Vào → Ra

| Hàm                                   | Vào                    | Ra                                           |
| -------------------------------------- | ----------------------- | -------------------------------------------- |
| `count_pairs([1, 2, 3, 1, 2])`       | list int                | `{(1,2): 2, (2,3): 1, (3,1): 1}`           |
| `merge([1, 2, 3, 1, 2], (1, 2), 99)` | list int, cặp, id mới | `[99, 3, 99]`                              |
| `encode("ệ")` khi chưa train       | chữ                    | `[225, 187, 135]` (= `0xE1, 0xBB, 0x87`) |
| `decode(encode(câu))`               | —                      | đúng câu ban đầu                        |

## 4. Khung code

```python
from collections import Counter

N_BYTES = 256  # số giá trị 1 byte có thể nhận: 0..255


def count_pairs(ids: list[int]) -> Counter:
    """Đếm số lần mỗi cặp token liền kề xuất hiện.

    Ví dụ: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    # TODO 1: return Counter(zip(ids, ids[1:]))
    #         zip(ids, ids[1:]) tạo các cặp (ids[0], ids[1]), (ids[1], ids[2]), ...


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Thay mọi lần xuất hiện của `pair` trong `ids` bằng `new_id` (không chồng lấn).

    Ví dụ: merge([1, 2, 3, 1, 2], (1, 2), 99) -> [99, 3, 99]
            merge([7, 7, 7], (7, 7), 99)      -> [99, 7]
    """
    # TODO 2: merged = [] ; i = 0
    # TODO 3: while i < len(ids):
    #             nếu i + 1 < len(ids) VÀ (ids[i], ids[i + 1]) == pair:
    #                 thêm new_id vào merged ; i += 2      ← nhảy qua cả 2 phần tử
    #             ngược lại:
    #                 thêm ids[i] vào merged ; i += 1
    # TODO 4: return merged


class ByteBPE:
    """Byte-level BPE tự viết — để hiểu thuật toán, không dùng cho dữ liệu lớn.

    Độ phức tạp:
        train: O(số_lần_gộp × số_byte_văn_bản) — rất chậm với văn bản lớn.
    """

    def __init__(self) -> None:
        # TODO 5: self.merges: dict[tuple[int, int], int] = {}
        # TODO 6: self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(N_BYTES)}
        #         → 256 token đầu tiên là 256 byte

    def train(self, text: str, vocab_size: int) -> None:
        """Học (vocab_size - 256) luật gộp từ `text`."""
        # TODO 7: nếu vocab_size < N_BYTES → raise ValueError có chữ "vocab_size"
        # TODO 8: ids = list(text.encode("utf-8"))
        # TODO 9: for new_id in range(N_BYTES, vocab_size):
        #             counts = count_pairs(ids)
        #             nếu counts rỗng → break                     (văn bản quá ngắn)
        #             best_pair = max(counts, key=counts.get)     (cặp nhiều nhất)
        #             ids = merge(ids, best_pair, new_id)
        #             self.merges[best_pair] = new_id
        #             self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

    def encode(self, text: str) -> list[int]:
        """chữ -> list token id."""
        # TODO 10: ids = list(text.encode("utf-8"))
        # TODO 11: while len(ids) >= 2:
        #              counts = count_pairs(ids)
        #              # chọn cặp có luật gộp HỌC SỚM NHẤT (id nhỏ nhất); cặp không có luật → vô cực
        #              pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
        #              nếu pair không có trong self.merges → break  (hết cặp gộp được)
        #              ids = merge(ids, pair, self.merges[pair])
        # TODO 12: return ids

    def decode(self, ids: list[int]) -> str:
        """list token id -> chữ."""
        # TODO 13: raw = b"".join(self.vocab[i] for i in ids)
        # TODO 14: return raw.decode("utf-8", errors="replace")
        #          errors="replace": byte lỗi (ví dụ decode 1 token là nửa chữ "ệ") → hiện "�" thay vì crash
```

**Vì sao encode phải gộp theo thứ tự học?** Luật sau được học **trên kết quả** của luật trước (ví dụ `Za → Y` cần `Z` đã tồn tại). Gộp sai thứ tự → ra token khác lúc train.

<details>
<summary><b>Lời giải</b> (mở khi kẹt)</summary>

```python
from collections import Counter

N_BYTES = 256


def count_pairs(ids: list[int]) -> Counter:
    """(docstring như khung)"""
    return Counter(zip(ids, ids[1:]))


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """(docstring như khung)"""
    merged = []
    i = 0
    while i < len(ids):
        is_pair = i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair
        if is_pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


class ByteBPE:
    """(docstring như khung)"""

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(N_BYTES)}

    def train(self, text: str, vocab_size: int) -> None:
        if vocab_size < N_BYTES:
            raise ValueError(f"vocab_size={vocab_size} phải >= {N_BYTES}")

        ids = list(text.encode("utf-8"))
        for new_id in range(N_BYTES, vocab_size):
            counts = count_pairs(ids)
            if not counts:
                break
            best_pair = max(counts, key=counts.get)
            ids = merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = count_pairs(ids)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")
```

</details>

**Thử nhanh:**

```powershell
python -c "from tokenizer.bpe_toy import ByteBPE; b = ByteBPE(); b.train('Người Việt nói tiếng Việt. ' * 20, 300); ids = b.encode('tiếng Việt'); print(ids, [b.decode([i]) for i in ids], b.decode(ids))"
```

Sẽ thấy vài token, mỗi token là một mẩu chữ, và decode ra lại đúng `tiếng Việt`.

## 5. Test — `tests/tokenizer/test_bpe_toy.py`

| Test                                           | Chứng minh                                                          |
| ---------------------------------------------- | -------------------------------------------------------------------- |
| `test_count_pairs`                           | Đếm cặp đúng                                                    |
| `test_merge_replaces_every_occurrence`       | Gộp mọi chỗ                                                       |
| `test_merge_does_not_overlap`                | `[7,7,7]` → `[99, 7]`, không gộp chồng                       |
| `test_untrained_encode_is_utf8_bytes`        | Chưa train → encode = byte UTF-8                                   |
| `test_vocab_size_below_256_raises`           | vocab < 256 bị từ chối                                            |
| `test_training_creates_requested_vocab_size` | Train xong vocab đúng kích thước                                |
| `test_training_makes_sequences_shorter`      | Sau train, chuỗi token ngắn hơn chuỗi byte                       |
| `test_repeated_character_becomes_one_token`  | "ệ" (3 byte) gộp thành 1 token                                    |
| `test_decode_encode_round_trip`              | decode(encode(x)) == x, kể cả chữ chưa gặp, emoji, chuỗi rỗng |

<details>
<summary><b>Code test</b></summary>

```python
import pytest

from tokenizer.bpe_toy import ByteBPE, count_pairs, merge

TRAIN_TEXT = "Việt Nam là một quốc gia. Người Việt nói tiếng Việt. " * 20


def test_count_pairs():
    assert count_pairs([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_merge_replaces_every_occurrence():
    assert merge([1, 2, 3, 1, 2], pair=(1, 2), new_id=99) == [99, 3, 99]


def test_merge_does_not_overlap():
    assert merge([7, 7, 7], pair=(7, 7), new_id=99) == [99, 7]


def test_untrained_encode_is_utf8_bytes():
    bpe = ByteBPE()

    assert bpe.encode("ệ") == [0xE1, 0xBB, 0x87]  # "ệ" = 3 byte UTF-8


def test_vocab_size_below_256_raises():
    with pytest.raises(ValueError, match="vocab_size"):
        ByteBPE().train(TRAIN_TEXT, vocab_size=100)


def test_training_creates_requested_vocab_size():
    bpe = ByteBPE()
    bpe.train(TRAIN_TEXT, vocab_size=300)

    assert len(bpe.vocab) == 300
    assert len(bpe.merges) == 300 - 256


def test_training_makes_sequences_shorter():
    bpe = ByteBPE()
    bpe.train(TRAIN_TEXT, vocab_size=300)
    text = "Người Việt nói tiếng Việt."

    assert len(bpe.encode(text)) < len(text.encode("utf-8"))


def test_repeated_character_becomes_one_token():
    bpe = ByteBPE()
    bpe.train("ệ" * 50, vocab_size=258)  # 2 lần gộp: (E1,BB) rồi (256,87)

    assert len(bpe.encode("ệ")) == 1


@pytest.mark.parametrize("text", ["Người Việt nói tiếng Việt.", "Chưa từng thấy: ỹ ữ 😀", ""])
def test_decode_encode_round_trip(text):
    bpe = ByteBPE()
    bpe.train(TRAIN_TEXT, vocab_size=300)

    assert bpe.decode(bpe.encode(text)) == text
```

</details>

## 6. Xong khi

- [X] `python -m pytest tests/tokenizer/test_bpe_toy.py -v` → **11 passed**
- [ ] Gửi Claude review

---

---

# A2 — `tokenizer/train_tokenizer.py` (tokenizer thật)

> Phần này Claude **chưa chạy thử** vì máy chưa cài `tokenizers`. Gặp lỗi thì gửi Claude output.

## 1. Vị trí

```
▶"Hà Nội..." → TOKENIZER → input_ids◀ → Embedding → [Block] × L → RMSNorm → lm_head → logits
```

Tạo ra file **`tokenizer/tokenizer.json`** — dùng xuyên suốt mọi chặng sau.

## 2. Làm gì

Cùng thuật toán BPE như A1, nhưng thư viện `tokenizers` (viết bằng Rust) nhanh hơn hàng nghìn lần, thêm 2 thứ:

**a) Tách từ trước khi gộp (`ByteLevel` pre-tokenizer).** Cắt câu theo dấu cách / dấu câu trước, rồi mới BPE trong từng mẩu → không bao giờ gộp xuyên qua 2 từ (tránh token vô nghĩa kiểu `"a V"`).

**b) Token đặc biệt (special tokens).** Chuỗi như `<think>` luôn là **1 token** với id cố định, không bị BPE cắt vụn. Phải **chừa sẵn ngay bây giờ** cho mọi chặng sau (chat, tư duy, gọi tool) — thêm sau = train lại tokenizer = mọi model đã train thành vô dụng.

| Token                                                                        | Dùng ở chặng                                   |
| ---------------------------------------------------------------------------- | ------------------------------------------------- |
| `<\|pad\|>`, `<\|bos\|>`, `<\|eos\|>`                                        | Pretrain: đệm, đầu văn bản, cuối văn bản |
| `<\|im_start\|>`, `<\|im_end\|>`                                             | SFT: bắt đầu / kết thúc 1 tin nhắn chat     |
| `<think>`, `</think>`                                                    | Tư duy thích ứng                               |
| `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>` | Tool use                                          |
| `<\|reserved_0\|>` … `<\|reserved_7\|>`                                     | Dự phòng cho nhu cầu chưa nghĩ ra            |

Token đặc biệt nhận id **0, 1, 2, ...** theo đúng thứ tự trong danh sách.

**Chọn `vocab_size`:** vocab to → câu ít token hơn nhưng bảng embedding to (`V·d` tham số). Gợi ý **16000** cho model tí hon.

**Train trên dữ liệu nào:** `dataset/pretrain/part_00.txt` (~19MB) là đủ để có tokenizer tốt, chạy nhanh. **Không** dùng `part_09` (dành làm validation).

## 3. Vào → Ra

|      |                                   |
| ---- | --------------------------------- |
| Vào | Các file`.txt`, `vocab_size` |
| Ra   | `tokenizer/tokenizer.json`      |

Dùng file đã train:

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer/tokenizer.json")
enc = tok.encode("Hà Nội là thủ đô")
enc.ids                 # [..., ..., ...]      list số nguyên
tok.decode(enc.ids)     # "Hà Nội là thủ đô"
tok.token_to_id("<think>")   # 5
tok.get_vocab_size()    # 16000  ← dùng làm vocab_size cho ModelConfig
```

## 4. Khung code

```python
import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

N_RESERVED = 8
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
] + [f"<|reserved_{i}|>" for i in range(N_RESERVED)]


def build_tokenizer(files: list[str], vocab_size: int) -> Tokenizer:
    """Train byte-level BPE tokenizer trên các file văn bản.

    Args:
        files: Đường dẫn các file .txt.
        vocab_size: Kích thước vocab mong muốn (đã tính cả token đặc biệt).

    Returns:
        Tokenizer đã train. Token đặc biệt có id 0..len(SPECIAL_TOKENS)-1.
    """
    # TODO 1: tokenizer = Tokenizer(models.BPE())
    # TODO 2: tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # TODO 3: tokenizer.decoder = decoders.ByteLevel()
    # TODO 4: trainer = trainers.BpeTrainer(
    #             vocab_size=vocab_size,
    #             special_tokens=SPECIAL_TOKENS,
    #             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),   ← đủ 256 byte, không bao giờ gặp chữ lạ
    #             show_progress=True,
    #         )
    # TODO 5: tokenizer.train(files, trainer)
    # TODO 6: return tokenizer


def main() -> None:
    """Chạy từ dòng lệnh: train rồi lưu tokenizer.json."""
    # TODO 7: parser = argparse.ArgumentParser(description="Train byte-level BPE tokenizer")
    #         parser.add_argument("--files", nargs="+", required=True)
    #         parser.add_argument("--vocab-size", type=int, default=16000)
    #         parser.add_argument("--out", default="tokenizer/tokenizer.json")
    #         args = parser.parse_args()
    # TODO 8: tokenizer = build_tokenizer(args.files, args.vocab_size)
    # TODO 9: Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    #         tokenizer.save(args.out)
    #         print(f"Đã lưu {args.out}, vocab_size={tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
```

<details>
<summary><b>Lời giải</b> (mở khi kẹt)</summary>

```python
import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

N_RESERVED = 8
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
] + [f"<|reserved_{i}|>" for i in range(N_RESERVED)]


def build_tokenizer(files: list[str], vocab_size: int) -> Tokenizer:
    """(docstring như khung)"""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train(files, trainer)
    return tokenizer


def main() -> None:
    """(docstring như khung)"""
    parser = argparse.ArgumentParser(description="Train byte-level BPE tokenizer")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--out", default="tokenizer/tokenizer.json")
    args = parser.parse_args()

    tokenizer = build_tokenizer(args.files, args.vocab_size)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.out)
    print(f"Đã lưu {args.out}, vocab_size={tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
```

</details>

## 5. Test — `tests/tokenizer/test_train_tokenizer.py`

Test train tokenizer **tí hon** trên file tạm (nhanh), không dùng dữ liệu thật. Riêng test cuối kiểm tra `tokenizer.json` thật — tự bỏ qua (skip) nếu chưa train.

| Test                                  | Chứng minh                                                                              |
| ------------------------------------- | ---------------------------------------------------------------------------------------- |
| `test_vocab_size_is_respected`      | Không vượt vocab_size                                                                 |
| `test_special_tokens_get_first_ids` | Token đặc biệt có id 0, 1, 2, ... đúng thứ tự                                    |
| `test_special_token_is_not_split`   | Mỗi token đặc biệt nằm giữa câu vẫn là**1 id** (tiêu chí xong chặng 0) |
| `test_decode_encode_round_trip`     | decode(encode(x)) == x (tiêu chí xong chặng 0)                                        |
| `test_real_tokenizer_round_trip`    | `tokenizer.json` thật hoạt động                                                    |

<details>
<summary><b>Code test</b></summary>

```python
from pathlib import Path

import pytest

from tokenizer.train_tokenizer import SPECIAL_TOKENS, build_tokenizer

SAMPLE_TEXT = (
    "Hà Nội là thủ đô của Việt Nam.\n\n"
    "Người Việt nói tiếng Việt. Tiếng Việt có dấu thanh.\n\n"
) * 50
VOCAB_SIZE = 400
REAL_TOKENIZER = Path("tokenizer/tokenizer.json")


@pytest.fixture
def small_tokenizer(tmp_path):
    """Train tokenizer tí hon trên file tạm. tmp_path là thư mục tạm pytest tự tạo và tự xoá."""
    data_file = tmp_path / "sample.txt"
    data_file.write_text(SAMPLE_TEXT, encoding="utf-8")
    return build_tokenizer([str(data_file)], VOCAB_SIZE)


def test_vocab_size_is_respected(small_tokenizer):
    assert small_tokenizer.get_vocab_size() <= VOCAB_SIZE


def test_special_tokens_get_first_ids(small_tokenizer):
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        assert small_tokenizer.token_to_id(token) == expected_id


@pytest.mark.parametrize("token", SPECIAL_TOKENS)
def test_special_token_is_not_split(small_tokenizer, token):
    ids = small_tokenizer.encode(f"xin chào {token} tạm biệt").ids

    assert small_tokenizer.token_to_id(token) in ids


@pytest.mark.parametrize("text", ["Hà Nội là thủ đô của Việt Nam.", "Chữ lạ chưa gặp: ỹ ữ 😀"])
def test_decode_encode_round_trip(small_tokenizer, text):
    ids = small_tokenizer.encode(text).ids

    assert small_tokenizer.decode(ids) == text


@pytest.mark.skipif(not REAL_TOKENIZER.exists(), reason="chưa train tokenizer thật")
def test_real_tokenizer_round_trip():
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(REAL_TOKENIZER))
    text = "Hà Nội là thủ đô của Việt Nam."

    assert tokenizer.decode(tokenizer.encode(text).ids) == text
    assert tokenizer.token_to_id("<think>") == SPECIAL_TOKENS.index("<think>")
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/tokenizer/test_train_tokenizer.py -v` → toàn bộ passed, 1 skipped
- [ ] Train tokenizer thật:
  ```powershell
  python -m tokenizer.train_tokenizer --files dataset/pretrain/part_00.txt --vocab-size 16000
  ```
- [ ] Chạy lại test → test cuối hết skip, **passed**
- [ ] Gửi Claude review

**Xong A1 + A2 = xong phần tokenizer của chặng 0.**

---

---

# PHẦN B — MODEL

---

# B1 — `core/embedding.py`

## 1. Vị trí

```
"Hà Nội..." → Tokenizer → input_ids → ▶TokenEmbedding◀ → [Block] × L → RMSNorm → lm_head → logits
```

## 2. Làm gì

Mỗi token id đổi thành một vector `d` chiều bằng cách **tra bảng**.

Bảng `weight` shape `(V, d)`: **hàng thứ `i` là vector của token `i`**.

```
weight (V=5, d=3):                        input_ids = [[3, 0, 3]]
  hàng 0: [ 0.01, -0.02,  0.03]   ◀─┐
  hàng 1: [ 0.00,  0.01, -0.01]     │     output:
  hàng 2: [-0.03,  0.02,  0.00]     │       [[ [0.02, 0.01, -0.02],   ← hàng 3
  hàng 3: [ 0.02,  0.01, -0.02]   ◀─┼─┐        [0.01, -0.02, 0.03],   ← hàng 0
  hàng 4: [ 0.01,  0.00,  0.01]     │ │        [0.02, 0.01, -0.02] ]] ← hàng 3
```

- Bảng là **tham số được học**. Lúc đầu là số ngẫu nhiên nhỏ; train xong, token có nghĩa gần nhau có vector gần nhau.
- Trong PyTorch: `weight[input_ids]` — đánh chỉ số bằng tensor, làm đúng việc tra bảng. `nn.Embedding` bên trong cũng làm y như vậy; ở đây ta tự viết.
- Cùng bảng này sẽ được **dùng lại làm `lm_head`** ở B7 (tie embedding).
- Chỉ những hàng **có token xuất hiện** mới nhận gradient.

**Kiểm tra đầu vào:** id ngoài khoảng `0..V-1` → báo `ValueError` rõ ràng. Không kiểm tra thì trên CPU báo `IndexError`, trên GPU báo lỗi rất khó hiểu (`device-side assert`). Nguyên nhân hay gặp nhất: **`vocab_size` trong config khác số token của tokenizer**.

## 3. Vào → Ra

|                   | Shape         | dtype          |
| ----------------- | ------------- | -------------- |
| Vào`input_ids` | `(B, T)`    | `torch.long` |
| `weight`        | `(V, d)`    | float32        |
| Ra                | `(B, T, d)` | float32        |

**Số tham số:** `V × d`.

## 4. Khung code

```python
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
        # TODO 1: super().__init__()
        # TODO 2: self.vocab_size = vocab_size
        # TODO 3: self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        #         torch.empty = cấp bộ nhớ, chưa gán giá trị
        # TODO 4: nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # TODO 5: nếu input_ids.dtype != torch.long → raise ValueError có chữ "long"
        # TODO 6: nếu input_ids.numel() > 0 và (input_ids.min() < 0 hoặc input_ids.max() >= self.vocab_size)
        #             → raise ValueError có chữ "input_ids"
        # TODO 7: return self.weight[input_ids]
```

<details>
<summary><b>Lời giải</b> (mở khi kẹt)</summary>

```python
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """(docstring như khung)"""

    def __init__(self, vocab_size: int, d_model: int, init_std: float) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            raise ValueError(f"input_ids phải là torch.long, nhận {input_ids.dtype}")
        if input_ids.numel() > 0 and (input_ids.min() < 0 or input_ids.max() >= self.vocab_size):
            raise ValueError(f"input_ids phải nằm trong [0, {self.vocab_size - 1}]")
        return self.weight[input_ids]
```

</details>

## 5. Test — `tests/core/test_embedding.py`

| Test                                     | Chứng minh                                   |
| ---------------------------------------- | --------------------------------------------- |
| `test_output_shape`                    | `(B, T)` → `(B, T, d)`                   |
| `test_parameter_count`                 | Đúng`V × d`                              |
| `test_output_is_row_of_weight`         | Token 5 → đúng hàng 5 của bảng          |
| `test_same_token_gives_same_vector`    | Cùng id → cùng vector, bất kể vị trí   |
| `test_init_std_is_small`               | Khởi tạo đúng độ lệch 0.02             |
| `test_gradient_only_reaches_used_rows` | Chỉ hàng của token có mặt nhận gradient |
| `test_out_of_range_ids_raise`          | id = V hoặc id = -1 bị từ chối            |
| `test_float_ids_raise`                 | id kiểu float bị từ chối                  |

<details>
<summary><b>Code test</b></summary>

```python
import pytest
import torch

from core.embedding import TokenEmbedding

VOCAB = 100
D_MODEL = 16
INIT_STD = 0.02


def test_output_shape():
    emb = TokenEmbedding(VOCAB, D_MODEL, INIT_STD)
    ids = torch.randint(0, VOCAB, (2, 5))

    assert emb(ids).shape == (2, 5, D_MODEL)


def test_parameter_count():
    emb = TokenEmbedding(VOCAB, D_MODEL, INIT_STD)

    assert sum(p.numel() for p in emb.parameters()) == VOCAB * D_MODEL


def test_output_is_row_of_weight():
    emb = TokenEmbedding(VOCAB, D_MODEL, INIT_STD)
    ids = torch.tensor([[5, 42]])

    out = emb(ids)

    torch.testing.assert_close(out[0, 0], emb.weight[5])
    torch.testing.assert_close(out[0, 1], emb.weight[42])


def test_same_token_gives_same_vector():
    emb = TokenEmbedding(VOCAB, D_MODEL, INIT_STD)
    out = emb(torch.tensor([[7, 3, 7]]))

    torch.testing.assert_close(out[0, 0], out[0, 2])


def test_init_std_is_small():
    emb = TokenEmbedding(1000, 64, INIT_STD)

    assert abs(emb.weight.std().item() - INIT_STD) < 0.002


def test_gradient_only_reaches_used_rows():
    emb = TokenEmbedding(VOCAB, D_MODEL, INIT_STD)
    emb(torch.tensor([[1, 2]])).sum().backward()

    used = emb.weight.grad[[1, 2]]
    unused = emb.weight.grad[3:]

    assert used.abs().sum() > 0
    assert unused.abs().sum() == 0


@pytest.mark.parametrize("bad_ids", [torch.tensor([[VOCAB]]), torch.tensor([[-1]])])
def test_out_of_range_ids_raise(bad_ids):
    emb = TokenEmbedding(VOCAB, D_MODEL, INIT_STD)

    with pytest.raises(ValueError, match="input_ids"):
        emb(bad_ids)


def test_float_ids_raise():
    emb = TokenEmbedding(VOCAB, D_MODEL, INIT_STD)

    with pytest.raises(ValueError, match="long"):
        emb(torch.tensor([[1.0, 2.0]]))
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/core/test_embedding.py -v` → **9 passed**
- [ ] Gửi Claude review

---

---

# B2 — `core/rmsnorm.py`

## 1. Vị trí

```
input_ids → Embedding → [Block: ▶RMSNorm◀ → Attention → + → ▶RMSNorm◀ → FeedForward → +] × L → ▶RMSNorm◀ → lm_head
```

Dùng 3 chỗ: trước Attention, trước FeedForward, và 1 lần cuối model.

## 2. Làm gì

Chia mỗi vector cho "độ lớn trung bình" của nó → vector ra có độ lớn ≈ 1, rồi nhân trọng số học được `weight`. Giữ các con số không phình to / co nhỏ qua nhiều tầng → train ổn định.

```
RMS(x) = sqrt( trung bình của x² trên chiều cuối  +  eps )
y      = x / RMS(x) * weight
```

## 3. Vào → Ra

|           | Shape                       | Ví dụ                |
| --------- | --------------------------- | ---------------------- |
| Vào`x` | `(..., d)`                | `[[3.0, 4.0]]`       |
| Ra`y`   | `(..., d)` — giống vào | `[[0.8485, 1.1314]]` |

Tính tay: trung bình `3², 4²` = `(9 + 16) / 2 = 12.5` → `RMS = √12.5 ≈ 3.5355` → `3 / 3.5355 = 0.8485`, `4 / 3.5355 = 1.1314`.

## 4. Khung code

Bạn đã viết xong docstring và `__init__`. Còn `forward`:

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Chuẩn hoá x theo chiều cuối. Đầu ra cùng shape và dtype với x."""
        # TODO 4: input_dtype = x.dtype
        # TODO 5: x_float = x.float()                                       ← ép float32
        # TODO 6: mean_square = x_float.pow(2).mean(dim=-1, keepdim=True)    ← shape (..., 1)
        # TODO 7: x_normed = x_float * torch.rsqrt(mean_square + self.eps)   ← rsqrt(a) = 1/sqrt(a)
        # TODO 8: nhân x_normed với self.weight
        # TODO 9: ép về input_dtype rồi return                               ← .to(input_dtype)
```

- **`dim=-1`**: trung bình theo chiều cuối → mỗi token chuẩn hoá riêng. Quên → trung bình cả tensor, sai mà không báo lỗi.
- **`keepdim=True`**: giữ shape `(..., 1)` để chia được cho `(..., d)`.
- **Ép float32**: fp16 tối đa 65504; `x = 1000` → `x² = 1 000 000` tràn thành `inf`.

<details>
<summary><b>Lời giải forward</b> (mở khi kẹt)</summary>

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Chuẩn hoá x theo chiều cuối. Đầu ra cùng shape và dtype với x."""
        input_dtype = x.dtype
        x_float = x.float()
        mean_square = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(mean_square + self.eps)
        return (x_normed * self.weight).to(input_dtype)
```

</details>

## 5. Test — `tests/core/test_rmsnorm.py`

| Test                                            | Chứng minh                                                                  |
| ----------------------------------------------- | ---------------------------------------------------------------------------- |
| `test_output_shape_is_unchanged`              | Shape ra = shape vào                                                        |
| `test_matches_hand_calculation`               | `[3, 4]` → đúng số tính tay                                           |
| `test_output_rms_is_one`                      | Độ lớn đầu ra ≈ 1                                                      |
| `test_scaling_input_does_not_change_output`   | Nhân x với 0.5 / 10 / 1000 → đầu ra không đổi                        |
| `test_weight_scales_output`                   | weight = 2 → đầu ra gấp đôi                                            |
| `test_each_token_is_normalized_independently` | Đổi token này không ảnh hưởng token khác (bắt lỗi quên`dim=-1`) |
| `test_output_keeps_input_dtype`               | Vào fp16/bf16 → ra fp16/bf16                                               |
| `test_large_fp16_input_does_not_overflow`     | `x = 1000` kiểu fp16 → không ra `inf` (bắt lỗi quên ép float32)   |
| `test_weight_receives_gradient`               | `weight` học được                                                      |

<details>
<summary><b>Code test</b></summary>

```python
import math

import pytest
import torch

from core.rmsnorm import RMSNorm

DIM = 16
EPS = 1e-6


def test_output_shape_is_unchanged():
    norm = RMSNorm(DIM, EPS)
    x = torch.randn(2, 5, DIM)

    assert norm(x).shape == (2, 5, DIM)


def test_matches_hand_calculation():
    norm = RMSNorm(dim=2, eps=EPS)
    x = torch.tensor([[3.0, 4.0]])

    expected = x / math.sqrt(12.5)  # mean(9, 16) = 12.5

    torch.testing.assert_close(norm(x), expected)


def test_output_rms_is_one():
    norm = RMSNorm(DIM, EPS)
    x = torch.randn(2, 5, DIM) * 7.0

    rms = norm(x).pow(2).mean(dim=-1).sqrt()

    torch.testing.assert_close(rms, torch.ones(2, 5))


@pytest.mark.parametrize("scale", [0.5, 10.0, 1000.0])
def test_scaling_input_does_not_change_output(scale):
    norm = RMSNorm(DIM, EPS)
    x = torch.randn(2, 5, DIM)

    torch.testing.assert_close(norm(x * scale), norm(x))


def test_weight_scales_output():
    norm = RMSNorm(DIM, EPS)
    x = torch.randn(2, 5, DIM)
    base = norm(x)

    with torch.no_grad():
        norm.weight.fill_(2.0)

    torch.testing.assert_close(norm(x), 2.0 * base)


def test_each_token_is_normalized_independently():
    norm = RMSNorm(DIM, EPS)
    x = torch.randn(2, 5, DIM)
    x_changed = x.clone()
    x_changed[0, 0] += 100.0

    torch.testing.assert_close(norm(x_changed)[0, 1:], norm(x)[0, 1:])
    torch.testing.assert_close(norm(x_changed)[1], norm(x)[1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_output_keeps_input_dtype(dtype):
    norm = RMSNorm(DIM, EPS)
    x = torch.randn(2, 5, DIM).to(dtype)

    assert norm(x).dtype == dtype


def test_large_fp16_input_does_not_overflow():
    norm = RMSNorm(4, EPS)
    x = torch.full((1, 4), 1000.0, dtype=torch.float16)  # 1000² = 1e6 > giới hạn fp16 (65504)

    y = norm(x)

    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.float(), torch.ones(1, 4))


def test_weight_receives_gradient():
    norm = RMSNorm(DIM, EPS)
    norm(torch.randn(2, 5, DIM)).sum().backward()

    assert norm.weight.grad is not None
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/core/test_rmsnorm.py -v` → **12 passed**
- [ ] Gửi Claude review

---

---

# B3 — `core/rope.py`

## 1. Vị trí

```
input_ids → Embedding → [Block: RMSNorm → Attention(▶RoPE◀) → + → RMSNorm → FeedForward → +] × L → ...
```

Không phải tầng riêng: Attention (B4) **gọi** RoPE để xoay q và k.

## 2. Làm gì

Attention bản thân không biết token nào đứng trước, token nào đứng sau. RoPE **xoay** vector q và k một góc tỉ lệ với vị trí token. Nhờ vậy `q · k` phụ thuộc **khoảng cách** giữa 2 token.

Chia vector `hd` chiều làm 2 nửa, ghép phần tử thứ `i` nửa đầu với phần tử thứ `i` nửa sau thành 1 cặp, xoay mỗi cặp như xoay điểm trên mặt phẳng:

```
(a, b)  →  (a·cos − b·sin,  a·sin + b·cos)
```

Góc của cặp `i` ở vị trí `m` = `m × tần_số_i`. Cặp đầu quay nhanh, cặp sau quay chậm.

| Hàm                | Làm gì                                    | Gọi khi nào           |
| ------------------- | ------------------------------------------- | ----------------------- |
| `precompute_rope` | Tính sẵn bảng cos, sin cho mọi vị trí | 1 lần, lúc tạo model |
| `apply_rope`      | Xoay q hoặc k bằng bảng                  | Mỗi lần forward       |

## 3. Vào → Ra

**`precompute_rope(head_dim, max_seq_len, theta)`** → `cos`, `sin` mỗi cái `(max_seq_len, head_dim // 2)`.

**`apply_rope(x, cos, sin)`**

|                      | Shape                                           | Ví dụ           |
| -------------------- | ----------------------------------------------- | ----------------- |
| Vào`x`            | `(B, h, T, hd)`                               | `(2, 4, 5, 16)` |
| Vào`cos`, `sin` | `(T, hd // 2)` — đã cắt đúng T vị trí | `(5, 8)`        |
| Ra                   | `(B, h, T, hd)`                               | `(2, 4, 5, 16)` |

`cos (T, hd/2)` nhân được với `x1 (B, h, T, hd/2)` vì PyTorch khớp shape từ phải sang trái.

## 4. Khung code

```python
import torch


def precompute_rope(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Tính sẵn bảng cos, sin cho RoPE.

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
    # TODO 1: nếu head_dim lẻ → raise ValueError có chữ "head_dim"
    # TODO 2: exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    # TODO 3: inv_freq = 1.0 / (theta ** exponents)                  ← (head_dim // 2,)
    # TODO 4: positions = torch.arange(max_seq_len, dtype=torch.float32)
    # TODO 5: angles = torch.outer(positions, inv_freq)              ← (max_seq_len, head_dim // 2)
    # TODO 6: return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Xoay q hoặc k theo vị trí.

    Shape:
        x: (B, h, T, hd), cos/sin: (T, hd // 2)  ->  (B, h, T, hd)
    Công thức:
        x1 = nửa đầu, x2 = nửa sau
        ra = [x1·cos − x2·sin,  x1·sin + x2·cos]
    """
    # TODO 7: half = x.shape[-1] // 2
    # TODO 8: x1 = x[..., :half] ; x2 = x[..., half:]
    # TODO 9: rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    # TODO 10: return rotated.to(x.dtype)
```

<details>
<summary><b>Lời giải</b> (mở khi kẹt)</summary>

```python
import torch


def precompute_rope(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(docstring như khung)"""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim={head_dim} phải là số chẵn")

    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inv_freq = 1.0 / (theta ** exponents)
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """(docstring như khung)"""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.to(x.dtype)
```

</details>

## 5. Test — `tests/core/test_rope.py`

| Test                                                   | Chứng minh                                                                                 |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `test_table_shapes`                                  | Bảng cos, sin đúng shape                                                                 |
| `test_odd_head_dim_raises`                           | `head_dim` lẻ bị từ chối                                                              |
| `test_output_shape_is_unchanged`                     | Xoay không đổi shape                                                                     |
| `test_position_zero_is_not_rotated`                  | Vị trí 0 → góc 0 → vector giữ nguyên                                                 |
| `test_rotation_preserves_vector_length`              | Xoay không làm dài/ngắn vector                                                          |
| `test_dot_product_depends_only_on_relative_position` | **Tính chất chính:** vị trí (1,3) và (5,7) cho cùng `q·k`; (1,4) thì khác |

<details>
<summary><b>Code test</b></summary>

```python
import pytest
import torch

from core.rope import apply_rope, precompute_rope

HEAD_DIM = 8
MAX_LEN = 16
THETA = 10000.0


def test_table_shapes():
    cos, sin = precompute_rope(HEAD_DIM, MAX_LEN, THETA)

    assert cos.shape == (MAX_LEN, HEAD_DIM // 2)
    assert sin.shape == (MAX_LEN, HEAD_DIM // 2)


def test_odd_head_dim_raises():
    with pytest.raises(ValueError, match="head_dim"):
        precompute_rope(7, MAX_LEN, THETA)


def test_output_shape_is_unchanged():
    cos, sin = precompute_rope(HEAD_DIM, MAX_LEN, THETA)
    x = torch.randn(2, 4, MAX_LEN, HEAD_DIM)  # (B, h, T, hd)

    assert apply_rope(x, cos, sin).shape == x.shape


def test_position_zero_is_not_rotated():
    cos, sin = precompute_rope(HEAD_DIM, MAX_LEN, THETA)
    x = torch.randn(1, 1, 1, HEAD_DIM)

    torch.testing.assert_close(apply_rope(x, cos[:1], sin[:1]), x)


def test_rotation_preserves_vector_length():
    cos, sin = precompute_rope(HEAD_DIM, MAX_LEN, THETA)
    x = torch.randn(2, 4, MAX_LEN, HEAD_DIM)

    y = apply_rope(x, cos, sin)

    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))


def test_dot_product_depends_only_on_relative_position():
    cos, sin = precompute_rope(HEAD_DIM, MAX_LEN, THETA)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)

    def rotated_dot(m: int, n: int) -> torch.Tensor:
        q_m = apply_rope(q, cos[m], sin[m])
        k_n = apply_rope(k, cos[n], sin[n])
        return (q_m * k_n).sum()

    # Cùng khoảng cách 2 → bằng nhau
    torch.testing.assert_close(rotated_dot(1, 3), rotated_dot(5, 7))
    # Khoảng cách khác → khác nhau
    assert not torch.isclose(rotated_dot(1, 3), rotated_dot(1, 4))
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/core/test_rope.py -v` → **6 passed**
- [ ] Gửi Claude review

---

---

# B4 — `core/attention.py`

## 1. Vị trí

```
input_ids → Embedding → [Block: RMSNorm → ▶Attention◀ → + → RMSNorm → FeedForward → +] × L → ...
```

Dùng: `config.py`, `rope.py` (B3).

## 2. Làm gì

Mỗi token **nhìn các token đứng trước nó** (và chính nó) để lấy thông tin. Từ mỗi token tạo 3 vector:

- **q** (query) — "tôi đang tìm gì"
- **k** (key) — "tôi có gì"
- **v** (value) — "thông tin tôi đưa ra"

Điểm token `i` nhìn token `j` = `q_i · k_j`. Softmax điểm → trọng số → trung bình có trọng số các `v`.

**Causal mask:** không được nhìn **tương lai** (nhiệm vụ là đoán token kế tiếp — nhìn trước là gian lận).

Làm song song `h` lần với vector nhỏ `hd = d / h` — mỗi lần là 1 **head**.

## 3. Vào → Ra (từng bước trong forward)

Ví dụ `B=2, T=5, d=64, h=4, hd=16`:

| Bước | Phép                                              | Shape                                 |
| ------ | -------------------------------------------------- | ------------------------------------- |
| vào   | `x`                                              | `(B, T, d)` = `(2, 5, 64)`        |
| 1      | `q_proj(x)`, `k_proj(x)`, `v_proj(x)`        | `(2, 5, 64)`                        |
| 2      | `.view(B, T, h, hd)` — cắt thành h head       | `(2, 5, 4, 16)`                     |
| 3      | `.transpose(1, 2)` — đưa head lên trước    | `(B, h, T, hd)` = `(2, 4, 5, 16)` |
| 4      | `apply_rope(q)`, `apply_rope(k)`               | `(2, 4, 5, 16)`                     |
| 5      | `scores = q @ k.transpose(-2, -1) / sqrt(hd)`    | `(B, h, T, T)` = `(2, 4, 5, 5)`   |
| 6      | điền`-inf` vào ô tương lai                 | `(2, 4, 5, 5)`                      |
| 7      | `softmax(dim=-1)` → mỗi hàng cộng = 1        | `(2, 4, 5, 5)`                      |
| 8      | `weights @ v`                                    | `(2, 4, 5, 16)`                     |
| 9      | `.transpose(1, 2).reshape(B, T, d)` — gộp head | `(2, 5, 64)`                        |
| 10     | `o_proj(...)`                                    | ra`(2, 5, 64)`                      |

**Mask** với `T = 5` (`True` = cấm nhìn):

```
         j=0    j=1    j=2    j=3    j=4
i=0   [ False  True   True   True   True  ]   token 0 chỉ nhìn chính nó
i=1   [ False  False  True   True   True  ]
i=2   [ False  False  False  True   True  ]
i=3   [ False  False  False  False  True  ]
i=4   [ False  False  False  False  False ]   token cuối nhìn tất cả
```

= `torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)`.

## 4. Khung code

```python
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
        x: (B, T, d), cos/sin: (T, hd // 2)  ->  (B, T, d)
    Công thức:
        Attention(q, k, v) = softmax(q·kᵀ / √hd + mask) · v
    Độ phức tạp:
        O(B · h · T² · hd) — bậc 2 theo T.
    Ổn định số học:
        - Chia √hd để điểm không quá lớn.
        - Softmax tính bằng float32.
        - Mask dùng -inf; đường chéo không bị mask nên không hàng nào toàn -inf (tránh NaN).
    """

    def __init__(self, config: ModelConfig) -> None:
        # TODO 1: super().__init__()
        # TODO 2: self.n_heads = config.n_heads ; self.head_dim = config.head_dim
        # TODO 3: 4 lớp nn.Linear(d_model, d_model, bias=False): q_proj, k_proj, v_proj, o_proj

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # TODO 4: batch, seq_len, d_model = x.shape
        # TODO 5: q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        #         tương tự cho k, v
        # TODO 6: q = apply_rope(q, cos, sin) ; k = apply_rope(k, cos, sin)     (v KHÔNG xoay)
        # TODO 7: scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        # TODO 8: future = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        #         scores = scores.masked_fill(future, float("-inf"))
        # TODO 9: weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        # TODO 10: out = weights @ v
        #          out = out.transpose(1, 2).reshape(batch, seq_len, d_model)
        # TODO 11: return self.o_proj(out)
```

**TODO 10:** sau `transpose` phải dùng `reshape`, **không** dùng `view` (lỗi `view size is not compatible...`).

<details>
<summary><b>Lời giải</b> (mở khi kẹt)</summary>

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import ModelConfig
from core.rope import apply_rope


class CausalSelfAttention(nn.Module):
    """(docstring như khung)"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        d_model = config.d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        future = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)

        out = weights @ v
        out = out.transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.o_proj(out)
```

</details>

## 5. Test — `tests/core/test_attention.py`

| Test                                               | Chứng minh                                                                                 |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `test_output_shape_is_unchanged`                 | `(B, T, d)` → `(B, T, d)`                                                              |
| `test_parameter_count`                           | Đúng`4·d²` (q, k, v, o)                                                               |
| `test_future_token_does_not_change_past_outputs` | **Tính chất chính:** đổi token cuối → đầu ra các token trước giữ nguyên |
| `test_single_token_sequence_works`               | `T = 1` vẫn chạy                                                                        |

<details>
<summary><b>Code test</b></summary>

```python
import torch

from core.attention import CausalSelfAttention
from core.rope import precompute_rope


def rope_for(config, seq_len):
    cos, sin = precompute_rope(config.head_dim, config.max_seq_len, config.rope_theta)
    return cos[:seq_len], sin[:seq_len]


def test_output_shape_is_unchanged(tiny_config):
    attn = CausalSelfAttention(tiny_config)
    cos, sin = rope_for(tiny_config, 6)
    x = torch.randn(2, 6, tiny_config.d_model)

    assert attn(x, cos, sin).shape == (2, 6, tiny_config.d_model)


def test_parameter_count(tiny_config):
    attn = CausalSelfAttention(tiny_config)
    d = tiny_config.d_model

    assert sum(p.numel() for p in attn.parameters()) == 4 * d * d  # q, k, v, o


def test_future_token_does_not_change_past_outputs(tiny_config):
    attn = CausalSelfAttention(tiny_config)
    cos, sin = rope_for(tiny_config, 6)
    x = torch.randn(2, 6, tiny_config.d_model)
    x_changed = x.clone()
    x_changed[:, -1] = torch.randn(2, tiny_config.d_model)  # chỉ đổi token cuối

    out = attn(x, cos, sin)
    out_changed = attn(x_changed, cos, sin)

    torch.testing.assert_close(out_changed[:, :-1], out[:, :-1])  # quá khứ không đổi
    assert not torch.allclose(out_changed[:, -1], out[:, -1])     # token cuối thì đổi


def test_single_token_sequence_works(tiny_config):
    attn = CausalSelfAttention(tiny_config)
    cos, sin = rope_for(tiny_config, 1)
    x = torch.randn(2, 1, tiny_config.d_model)

    assert attn(x, cos, sin).shape == (2, 1, tiny_config.d_model)
```

</details>

**Thử cho biết:** tạm xoá dòng `masked_fill` → chạy lại → `test_future_token_does_not_change_past_outputs` phải **FAILED**. Thấy rồi thì trả dòng đó lại.

## 6. Xong khi

- [ ] `python -m pytest tests/core/test_attention.py -v` → **4 passed**
- [ ] Đã thử xoá mask và thấy test fail
- [ ] Gửi Claude review

---

---

# B5 — `core/feedforward.py`

## 1. Vị trí

```
input_ids → Embedding → [Block: RMSNorm → Attention → + → RMSNorm → ▶FeedForward◀ → +] × L → ...
```

## 2. Làm gì

Attention trộn thông tin **giữa các token**. FeedForward xử lý **từng token riêng lẻ**: nở vector ra `ffn_hidden` chiều, lọc bằng "cổng", rồi thu về `d` chiều.

Kiểu SwiGLU (như Llama, minimind), 3 lớp Linear không bias:

```
gate = SiLU( gate_proj(x) )      d → ffn     "cổng": chiều nào được đi qua
up   = up_proj(x)                d → ffn     nội dung
ra   = down_proj( gate * up )    ffn → d     (* là nhân từng phần tử)
```

`SiLU(z) = z · sigmoid(z)` — có sẵn: `F.silu`.

## 3. Vào → Ra

|                               | Shape           | Ví dụ`d=64, ffn=128` |
| ----------------------------- | --------------- | ------------------------ |
| Vào`x`                     | `(B, T, d)`   | `(2, 5, 64)`           |
| sau`gate_proj`, `up_proj` | `(B, T, ffn)` | `(2, 5, 128)`          |
| Ra                            | `(B, T, d)`   | `(2, 5, 64)`           |

**Số tham số:** `3 × d × ffn`.

## 4. Khung code

```python
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
    Ổn định số học:
        Không có điểm đặc biệt.
    """

    def __init__(self, d_model: int, ffn_hidden: int) -> None:
        # TODO 1: super().__init__()
        # TODO 2: self.gate_proj = nn.Linear(d_model, ffn_hidden, bias=False)
        # TODO 3: self.up_proj   = nn.Linear(d_model, ffn_hidden, bias=False)
        # TODO 4: self.down_proj = nn.Linear(ffn_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 5: gate = F.silu(self.gate_proj(x))
        # TODO 6: up = self.up_proj(x)
        # TODO 7: return self.down_proj(gate * up)
```

`nn.Linear(in, out, bias=False)` = ma trận `(out, in)`; `linear(x)` tương đương `x @ weight.T`.

<details>
<summary><b>Lời giải</b> (mở khi kẹt)</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """(docstring như khung)"""

    def __init__(self, d_model: int, ffn_hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_hidden, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_hidden, bias=False)
        self.down_proj = nn.Linear(ffn_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
```

</details>

## 5. Test — `tests/core/test_feedforward.py`

| Test                                           | Chứng minh                               |
| ---------------------------------------------- | ----------------------------------------- |
| `test_output_shape_is_unchanged`             | `(B, T, d)` → `(B, T, d)`            |
| `test_parameter_count`                       | Đúng`3·d·ffn`, không bias          |
| `test_each_token_is_processed_independently` | Đổi token 0 → token khác không đổi |

<details>
<summary><b>Code test</b></summary>

```python
import torch

from core.feedforward import FeedForward

D_MODEL = 64
FFN_HIDDEN = 128


def test_output_shape_is_unchanged():
    ffn = FeedForward(D_MODEL, FFN_HIDDEN)
    x = torch.randn(2, 5, D_MODEL)

    assert ffn(x).shape == (2, 5, D_MODEL)


def test_parameter_count():
    ffn = FeedForward(D_MODEL, FFN_HIDDEN)

    n_params = sum(p.numel() for p in ffn.parameters())

    assert n_params == 3 * D_MODEL * FFN_HIDDEN  # gate + up + down, không bias


def test_each_token_is_processed_independently():
    ffn = FeedForward(D_MODEL, FFN_HIDDEN)
    x = torch.randn(2, 5, D_MODEL)
    x_changed = x.clone()
    x_changed[:, 0] += 1.0

    torch.testing.assert_close(ffn(x_changed)[:, 1:], ffn(x)[:, 1:])
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/core/test_feedforward.py -v` → **3 passed**
- [ ] Gửi Claude review

---

---

# B6 — `core/block.py`

## 1. Vị trí

```
input_ids → Embedding → [▶Block◀] × L → RMSNorm → lm_head → logits
```

Dùng: `rmsnorm.py`, `attention.py`, `feedforward.py`.

## 2. Làm gì

Ghép 4 thành phần thành **1 tầng**. Không có toán mới — chỉ nối dây:

```
x = x + Attention( RMSNorm1(x) )       ← residual: cộng lại chính đầu vào
x = x + FeedForward( RMSNorm2(x) )
```

- **Pre-norm:** norm đặt **trước** Attention / FeedForward.
- **Residual** (`x + ...`): gradient đi thẳng qua nhiều tầng → train sâu được.
- Viết `x = x + ...`, **không** `x += ...` (cộng tại chỗ dễ lỗi khi tính gradient).

## 3. Vào → Ra

|                      | Shape                                            |
| -------------------- | ------------------------------------------------ |
| Vào`x`            | `(B, T, d)`                                    |
| Vào`cos`, `sin` | `(T, hd // 2)` — chuyển thẳng cho attention |
| Ra                   | `(B, T, d)`                                    |

**Số tham số:** `4·d²` (attention) `+ 3·d·ffn` (feedforward) `+ 2·d` (2 RMSNorm).

## 4. Khung code

```python
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
        # TODO 1: super().__init__()
        # TODO 2: self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        # TODO 3: self.attn = CausalSelfAttention(config)
        # TODO 4: self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        # TODO 5: self.ffn = FeedForward(config.d_model, config.ffn_hidden)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # TODO 6: x = x + self.attn(self.attn_norm(x), cos, sin)
        # TODO 7: x = x + self.ffn(self.ffn_norm(x))
        # TODO 8: return x
```

<details>
<summary><b>Lời giải</b> (mở khi kẹt)</summary>

```python
import torch
import torch.nn as nn

from core.attention import CausalSelfAttention
from core.config import ModelConfig
from core.feedforward import FeedForward
from core.rmsnorm import RMSNorm


class Block(nn.Module):
    """(docstring như khung)"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = FeedForward(config.d_model, config.ffn_hidden)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

</details>

## 5. Test — `tests/core/test_block.py`

| Test                               | Chứng minh                              |
| ---------------------------------- | ---------------------------------------- |
| `test_output_shape_is_unchanged` | `(B, T, d)` → `(B, T, d)`           |
| `test_parameter_count`           | Đúng`4d² + 3·d·ffn + 2d`          |
| `test_block_is_causal`           | Ghép xong vẫn không nhìn tương lai |

<details>
<summary><b>Code test</b></summary>

```python
import torch

from core.block import Block
from core.rope import precompute_rope


def rope_for(config, seq_len):
    cos, sin = precompute_rope(config.head_dim, config.max_seq_len, config.rope_theta)
    return cos[:seq_len], sin[:seq_len]


def test_output_shape_is_unchanged(tiny_config):
    block = Block(tiny_config)
    cos, sin = rope_for(tiny_config, 6)
    x = torch.randn(2, 6, tiny_config.d_model)

    assert block(x, cos, sin).shape == (2, 6, tiny_config.d_model)


def test_parameter_count(tiny_config):
    block = Block(tiny_config)
    d, ffn = tiny_config.d_model, tiny_config.ffn_hidden

    expected = 4 * d * d + 3 * d * ffn + 2 * d  # attention + ffn + 2 RMSNorm

    assert sum(p.numel() for p in block.parameters()) == expected


def test_block_is_causal(tiny_config):
    block = Block(tiny_config)
    cos, sin = rope_for(tiny_config, 6)
    x = torch.randn(2, 6, tiny_config.d_model)
    x_changed = x.clone()
    x_changed[:, -1] = torch.randn(2, tiny_config.d_model)

    torch.testing.assert_close(block(x_changed, cos, sin)[:, :-1], block(x, cos, sin)[:, :-1])
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/core/test_block.py -v` → **3 passed**
- [ ] Gửi Claude review

---

---

# B7 — `core/model.py`

## 1. Vị trí

```
input_ids → ▶TokenEmbedding → [Block] × L → RMSNorm cuối → lm_head◀ → logits
```

Dùng: tất cả file B1–B6.

## 2. Làm gì

File có **2 class** (file hiện có `class Model` rỗng — thay bằng khung dưới):

| Class        | Là                     | Làm gì                                   | Ra                   |
| ------------ | ----------------------- | ------------------------------------------ | -------------------- |
| `Model`    | **thân**         | TokenEmbedding → L Block → RMSNorm cuối | `hidden (B, T, d)` |
| `CausalLM` | **thân + đầu** | `Model` → `lm_head`                   | `logits (B, T, V)` |

Tách thân / đầu vì chặng PPO sau này cần `hidden` (Critic dùng).

Thêm 4 việc:

1. **Bảng RoPE tính 1 lần** trong `Model.__init__`, lưu bằng `register_buffer` (buffer tự chuyển theo `.to("cuda")` nhưng không phải tham số học). Mỗi forward cắt `[:seq_len]`.
2. **Kiểm tra độ dài:** `seq_len > max_seq_len` → `ValueError`.
3. **Khởi tạo Linear nhỏ:** mọi `nn.Linear` lấy từ phân phối chuẩn độ lệch `INIT_STD = 0.02`. (TokenEmbedding tự khởi tạo ở B1.)
4. **Dùng chung trọng số** (`tie_embedding`): `lm_head.weight` trỏ vào **cùng tensor** `embed.weight` (cả hai shape `(V, d)`) → bớt `V·d` tham số.

`lm_head` = "tra bảng ngược": `logits[token i] = hidden · vector_của_token_i`. Token có vector giống `hidden` → điểm cao.

Loss **không** tính ở đây — sẽ nằm ở `trainer/losses.py` (chặng 2). Test và C1 tự tính bằng `F.cross_entropy`.

## 3. Vào → Ra

|                   | Shape                                 |
| ----------------- | ------------------------------------- |
| Vào`input_ids` | `(B, T)` long, giá trị `0..V-1` |
| sau`embed`      | `(B, T, d)`                         |
| sau mỗi Block    | `(B, T, d)`                         |
| `Model` ra      | `(B, T, d)`                         |
| `CausalLM` ra   | `(B, T, V)`                         |

**Số tham số** (có tie):

```
V·d  +  L · (4·d² + 3·d·ffn + 2·d)  +  d
│        └─ mỗi Block                  └─ RMSNorm cuối
└─ embedding (lm_head dùng chung, không đếm thêm)
```

Với `tiny_config`: `100·64 + 2·(4·4096 + 3·64·128 + 128) + 64` = **88 640**.

**Loss ban đầu ≈ ln V:** trọng số nhỏ → logits gần 0 → đoán gần như đều → cross-entropy ≈ `ln V`.

## 4. Khung code

```python
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
        # TODO 1: super().__init__() ; self.config = config
        # TODO 2: self.embed = TokenEmbedding(config.vocab_size, config.d_model, INIT_STD)
        # TODO 3: self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        # TODO 4: self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        # TODO 5: cos, sin = precompute_rope(config.head_dim, config.max_seq_len, config.rope_theta)
        #         self.register_buffer("rope_cos", cos, persistent=False)
        #         self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # TODO 6: batch, seq_len = input_ids.shape
        #         nếu seq_len > self.config.max_seq_len → raise ValueError có chữ "max_seq_len"
        # TODO 7: cos = self.rope_cos[:seq_len] ; sin = self.rope_sin[:seq_len]
        # TODO 8: x = self.embed(input_ids)
        # TODO 9: for layer in self.layers: x = layer(x, cos, sin)
        # TODO 10: return self.final_norm(x)


class CausalLM(nn.Module):
    """Thân + đầu: input_ids -> logits.

    Shape:
        input_ids: (B, T)  ->  logits: (B, T, V)
    """

    def __init__(self, config: ModelConfig) -> None:
        # TODO 11: super().__init__() ; self.config = config
        # TODO 12: self.model = Model(config)
        # TODO 13: self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # TODO 14: self.apply(_init_weights)
        # TODO 15: nếu config.tie_embedding: self.lm_head.weight = self.model.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # TODO 16: hidden = self.model(input_ids)
        # TODO 17: return self.lm_head(hidden)


def _init_weights(module: nn.Module) -> None:
    """Khởi tạo mọi nn.Linear bằng N(0, INIT_STD²)."""
    # TODO 18: nếu isinstance(module, nn.Linear):
    #              nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
```

`self.apply(fn)` gọi `fn` lên **mọi module con**.

<details>
<summary><b>Lời giải</b> (mở khi kẹt)</summary>

```python
import torch
import torch.nn as nn

from core.block import Block
from core.config import ModelConfig
from core.embedding import TokenEmbedding
from core.rmsnorm import RMSNorm
from core.rope import precompute_rope

INIT_STD = 0.02


class Model(nn.Module):
    """(docstring như khung)"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = TokenEmbedding(config.vocab_size, config.d_model, INIT_STD)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)

        cos, sin = precompute_rope(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len={seq_len} vượt max_seq_len={self.config.max_seq_len}")

        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]

        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.final_norm(x)


class CausalLM(nn.Module):
    """(docstring như khung)"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Model(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(_init_weights)
        if config.tie_embedding:
            self.lm_head.weight = self.model.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.model(input_ids)
        return self.lm_head(hidden)


def _init_weights(module: nn.Module) -> None:
    """(docstring như khung)"""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
```

</details>

## 5. Test — `tests/core/test_model.py`

| Test                                             | Chứng minh                                                                |
| ------------------------------------------------ | -------------------------------------------------------------------------- |
| `test_model_returns_hidden_states`             | Thân ra`(B, T, d)`                                                      |
| `test_causal_lm_returns_logits`                | Thân + đầu ra`(B, T, V)`                                              |
| `test_embedding_and_lm_head_share_weights`     | Tie đúng: cùng 1 tensor                                                 |
| `test_parameter_count_matches_formula`         | Đếm tham số khớp công thức (**tiêu chí xong chặng 1**)      |
| `test_untied_model_has_extra_lm_head`          | Tắt tie → thêm đúng`V·d`                                           |
| `test_initial_loss_is_close_to_log_vocab_size` | Loss ban đầu ≈ ln V                                                     |
| `test_model_is_causal`                         | Cả model không nhìn tương lai                                         |
| `test_prefix_gives_same_logits`                | Chạy 4 token đầu = 4 vị trí đầu của cả chuỗi                     |
| `test_sequence_longer_than_max_raises`         | Quá`max_seq_len` bị từ chối                                          |
| `test_can_overfit_one_batch`                   | Train 100 bước trên 1 batch → loss < 0.5 (model**học được**) |

<details>
<summary><b>Code test</b></summary>

```python
import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F

from core.model import CausalLM, Model


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())  # tham số dùng chung chỉ đếm 1 lần


def test_model_returns_hidden_states(tiny_config):
    model = Model(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 6))

    assert model(ids).shape == (2, 6, tiny_config.d_model)


def test_causal_lm_returns_logits(tiny_config):
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 6))

    assert lm(ids).shape == (2, 6, tiny_config.vocab_size)


def test_embedding_and_lm_head_share_weights(tiny_config):
    lm = CausalLM(tiny_config)

    assert lm.lm_head.weight is lm.model.embed.weight


def test_parameter_count_matches_formula(tiny_config):
    lm = CausalLM(tiny_config)
    V, d, L, ffn = tiny_config.vocab_size, tiny_config.d_model, tiny_config.n_layers, tiny_config.ffn_hidden

    expected = V * d + L * (4 * d * d + 3 * d * ffn + 2 * d) + d

    assert count_params(lm) == expected


def test_untied_model_has_extra_lm_head(tiny_config):
    untied_config = dataclasses.replace(tiny_config, tie_embedding=False)

    tied = count_params(CausalLM(tiny_config))
    untied = count_params(CausalLM(untied_config))

    assert untied - tied == tiny_config.vocab_size * tiny_config.d_model


def test_initial_loss_is_close_to_log_vocab_size(tiny_config):
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (4, 16))

    logits = lm(ids)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, tiny_config.vocab_size),
        ids[:, 1:].reshape(-1),
    )

    assert abs(loss.item() - math.log(tiny_config.vocab_size)) < 0.3


def test_model_is_causal(tiny_config):
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
    ids_changed = ids.clone()
    ids_changed[:, -1] = (ids[:, -1] + 1) % tiny_config.vocab_size

    torch.testing.assert_close(lm(ids_changed)[:, :-1], lm(ids)[:, :-1])


def test_prefix_gives_same_logits(tiny_config):
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 10))

    torch.testing.assert_close(lm(ids[:, :4]), lm(ids)[:, :4])


def test_sequence_longer_than_max_raises(tiny_config):
    lm = CausalLM(tiny_config)
    ids = torch.zeros(1, tiny_config.max_seq_len + 1, dtype=torch.long)

    with pytest.raises(ValueError, match="max_seq_len"):
        lm(ids)


def test_can_overfit_one_batch(tiny_config):
    lm = CausalLM(tiny_config)
    optimizer = torch.optim.AdamW(lm.parameters(), lr=3e-3)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 16))

    for _ in range(100):
        logits = lm(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, tiny_config.vocab_size), ids[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.5
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/core/test_model.py -v` → **10 passed**
- [ ] `python -m pytest tests -v` → tất cả passed
- [ ] Gửi Claude review

**Xong B7 = xong chặng 1** (test từng thành phần; đếm tham số khớp công thức).

---

---

# PHẦN C — GHÉP LẠI

---

# C1 — `scripts/text_to_logits.py`

## 1. Vị trí

**Toàn bộ luồng** trong một script:

```
▶"Hà Nội..." → Tokenizer → input_ids → Embedding → [Block] × L → RMSNorm → lm_head → logits → loss◀
```

Dùng: `tokenizer/tokenizer.json` (A2) + `core/model.py` (B7).

## 2. Làm gì

Không có kiến thức mới — chạy thử để **nhìn tận mắt** dữ liệu đi qua từng chặng. Model **chưa train** nên dự đoán là ngẫu nhiên; mục đích là thấy shape và loss đúng.

Loss ở đây: vị trí `t` đoán token ở vị trí `t + 1`:

```
logits[:, :-1]   ← vị trí 0 .. T-2 đưa ra dự đoán   (bỏ vị trí cuối: không có gì để đoán)
input_ids[:, 1:] ← đáp án: token ở vị trí 1 .. T-1   (bỏ vị trí đầu: không ai đoán nó)
```

Câu phải có **ít nhất 2 token**, không thì không có cặp nào để tính loss → ra `nan`.

## 3. Vào → Ra

| Bước                        | Giá trị                                   | Shape          |
| ----------------------------- | ------------------------------------------- | -------------- |
| chữ                          | `"Hà Nội là thủ đô của Việt Nam"` | —             |
| `tokenizer.encode(...).ids` | `[..., ..., ...]`                         | list T số     |
| `input_ids`                 | tensor                                      | `(1, T)`     |
| `lm.model(input_ids)`       | hidden                                      | `(1, T, 64)` |
| `lm.lm_head(hidden)`        | logits                                      | `(1, T, V)`  |
| loss                          | ≈`ln V` (V = 16000 → ≈ 9.68)           | số            |

## 4. Code (script, viết hẳn)

```python
"""Chạy thử toàn bộ luồng: chữ → token ids → model → logits → loss.

Chạy từ thư mục gốc:
    python -m scripts.text_to_logits
"""

import math

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from core.config import ModelConfig
from core.model import CausalLM

TOKENIZER_PATH = "tokenizer/tokenizer.json"
TEXT = "Hà Nội là thủ đô của Việt Nam"
TOP_K = 5


def main() -> None:
    torch.manual_seed(0)

    # ---- 1. Chữ → token ids ----
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    ids = tokenizer.encode(TEXT).ids
    print(f"Chữ: {TEXT!r}")
    print(f"Số byte UTF-8: {len(TEXT.encode('utf-8'))}  →  số token: {len(ids)}")
    print(f"Token ids: {ids}")
    print(f"Từng token: {[tokenizer.decode([i]) for i in ids]}")

    # ---- 2. Tạo model (vocab_size PHẢI bằng số token của tokenizer) ----
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=64,
        n_layers=2,
        n_heads=4,
        ffn_hidden=128,
        max_seq_len=64,
        rope_theta=10000.0,
        norm_eps=1e-6,
        tie_embedding=True,
    )
    lm = CausalLM(config)
    lm.eval()  # chế độ suy luận

    # ---- 3. ids → tensor (B=1, T) ----
    input_ids = torch.tensor([ids], dtype=torch.long)
    print(f"\ninput_ids: {tuple(input_ids.shape)}")

    # ---- 4. Forward ----
    with torch.no_grad():  # không cần gradient khi chỉ chạy thử
        hidden = lm.model(input_ids)
        logits = lm.lm_head(hidden)
    print(f"hidden:    {tuple(hidden.shape)}")
    print(f"logits:    {tuple(logits.shape)}")

    # ---- 5. Loss: vị trí t đoán token t+1 ----
    vocab = config.vocab_size
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), input_ids[:, 1:].reshape(-1))
    print(f"\nloss = {loss.item():.3f}   (ln V = {math.log(vocab):.3f})")

    # ---- 6. Model đoán token tiếp theo sau câu ----
    top_ids = logits[0, -1].topk(TOP_K).indices.tolist()
    print(f"{TOP_K} token model đoán tiếp theo (chưa train → ngẫu nhiên): "
          f"{[tokenizer.decode([i]) for i in top_ids]}")


if __name__ == "__main__":
    main()
```

## 5. Chạy

```powershell
python -m scripts.text_to_logits
```

Dùng `python -m scripts.text_to_logits` (không phải `python scripts/text_to_logits.py`) để import được `core`.

**Kết quả mong đợi** — dạng output như dưới; số token, token ids và tên từng token **tuỳ tokenizer bạn train**, không cần khớp:

```
Chữ: 'Hà Nội là thủ đô của Việt Nam'
Số byte UTF-8: 41  →  số token: T        ← T nhỏ hơn 41 nhiều
Token ids: [...]
Từng token: [...]

input_ids: (1, T)
hidden:    (1, T, 64)
logits:    (1, T, 16000)

loss = ...   (ln V = 9.680)             ← loss phải gần 9.68
5 token model đoán tiếp theo (chưa train → ngẫu nhiên): [...]
```

Những gì **phải đúng**: 3 dòng shape theo đúng mẫu `(1, T)`, `(1, T, 64)`, `(1, T, 16000)` với cùng một `T`; loss gần `ln V`.

- Một vài token decode riêng lẻ có thể hiện `�`: token đó là **nửa chữ có dấu** (vài byte của một ký tự UTF-8). Ghép cả dãy thì decode vẫn đúng.
- Luồng này đã được Claude chạy kiểm chứng với tokenizer tự viết ở A1 (thay cho thư viện): shape đúng, loss 5.699 so với ln V = 5.704.

## 6. Xong khi

- [ ] Script chạy không lỗi
- [ ] Shape `input_ids`, `hidden`, `logits` khớp bảng mục 3
- [ ] Loss gần `ln V`
- [ ] Thử đổi `TEXT` sang câu khác, xem số token thay đổi thế nào

**Xong C1 = đã đi hết đường từ chữ đến logits.** Chặng tiếp theo (pretrain) sẽ thêm: đọc cả file dữ liệu, cắt thành batch, và vòng lặp train để loss giảm.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

---

# PHẦN D — PRETRAIN

Model đã chạy được, giờ dạy nó tiếng Việt: cho đọc 174MB văn bản và học đoán token kế tiếp.

**Luồng của chặng này:**

```
dataset/pretrain/train/*.txt          174 MB văn bản
   │
   ▼  [D1] scripts/prepare_pretrain.py          chạy 1 lần, vài phút
train.bin (uint16)  +  train_meta.json          ~35 triệu token, ~71 MB
   │
   ▼  [D2] trainer/pretrain_dataset.py          cắt cửa sổ T+1, đọc bằng memmap
(x, y) mỗi cái (B, T)
   │
   ▼  [B7] core/model.py                        logits (B, T, V)
   │
   ▼  [D3] trainer/losses.py                    loss
   │
   ▼  backward
   │
   ▼  [D4] optim.py  +  [D5] lr_scheduler.py    cập nhật trọng số
   │
   ▼  [D6] checkpoint.py  +  [D7] metrics.py    lưu lại, in log
   
   tất cả ghép trong [D9] trainer/train_pretrain.py, tham số ở [D8] train_config.py
```

**Số đo thật** (Claude chạy thử trên 400KB dữ liệu của bạn với tokenizer 16000):

|                                 | Giá trị                               |
| ------------------------------- | --------------------------------------- |
| Tỉ lệ nén                    | 4.9 byte văn bản / 1 token            |
| Ước tính tập train (174 MB) | ~35 triệu token → file`.bin` ~71 MB |
| Ước tính tập eval (19 MB)   | ~3.9 triệu token → ~8 MB              |

35 triệu token đủ cho model 2–10 triệu tham số theo tỉ lệ Chinchilla (20 token/tham số).

---

---

# D1 — `scripts/prepare_pretrain.py`

## 1. Vị trí

```
▶.txt → .bin◀ → Dataset → Model → loss → backward → cập nhật
```

Chạy **một lần**, kết quả dùng cho mọi lần train sau.

## 2. Làm gì

Tokenize toàn bộ văn bản **trước**, lưu ra file nhị phân. Nếu tokenize trong lúc train thì mỗi epoch phải làm lại từ đầu, rất chậm.

**Mỗi văn bản được bọc:** `<|bos|>` + token ids + `<|eos|>`. Nhờ `<|eos|>`, model học được "chỗ này hết chuyện, sang chuyện khác".

**Kiểu số `uint16`:** 2 byte/token, chứa được 65536 giá trị, đủ cho vocab 16000. Dùng `uint32` là phí gấp đôi dung lượng.

**Đọc theo luồng:** đọc từng văn bản, gom thành lô 1000 cái rồi tokenize và ghi xuống đĩa. Không bao giờ nạp cả 174MB vào RAM.

**`meta.json`** ghi kèm số token, vocab_size, tokenizer đã dùng. Lúc train đối chiếu để bắt sớm lỗi lệch tokenizer.

## 3. Vào → Ra

|      |                                                           |
| ---- | --------------------------------------------------------- |
| Vào | Thư mục chứa`.txt`, đường dẫn `tokenizer.json` |
| Ra   | `<prefix>.bin` (uint16) và `<prefix>_meta.json`      |

```
dataset/pretrain/train/*.txt  →  dataset/pretrain/tokens/train.bin + train_meta.json
dataset/pretrain/eval/*.txt   →  dataset/pretrain/tokens/eval.bin  + eval_meta.json
```

## 4. Khung code

```python
"""Chuyển file .txt thành file nhị phân token ids (.bin) + meta.json."""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

DTYPE = np.uint16
MAX_VOCAB_FOR_DTYPE = int(np.iinfo(DTYPE).max) + 1
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
DOCS_PER_BATCH = 1000


def iter_documents(paths: list[Path]) -> Iterator[str]:
    """Đọc từng văn bản; các văn bản cách nhau bằng dòng trống."""
    # TODO 1: với mỗi path: mở file, gom các dòng không trống vào list `lines`
    # TODO 2: gặp dòng trống mà `lines` không rỗng → yield "\n".join(lines), rồi lines = []
    # TODO 3: hết file mà `lines` còn dữ liệu → yield nốt


def encode_documents(tokenizer, documents: list[str], bos_id: int, eos_id: int) -> list[int]:
    """Tokenize một lô văn bản, mỗi văn bản bọc bos ... eos."""
    # TODO 4: với mỗi encoding trong tokenizer.encode_batch(documents):
    #             thêm bos_id, rồi encoding.ids, rồi eos_id vào list kết quả


def write_bin(paths, tokenizer, out_bin: Path, docs_per_batch: int = DOCS_PER_BATCH) -> int:
    """Ghi token ids ra file nhị phân uint16. Trả về tổng số token đã ghi."""
    # TODO 5: vocab_size > MAX_VOCAB_FOR_DTYPE → raise ValueError
    # TODO 6: lấy bos_id, eos_id bằng tokenizer.token_to_id(...); thiếu → raise ValueError
    # TODO 7: tạo thư mục cha của out_bin
    # TODO 8: mở file "wb"; gom document thành lô docs_per_batch rồi gọi _flush;
    #         nhớ flush phần lẻ cuối cùng
    # TODO 9: trả về tổng số token


def _flush(out_file, tokenizer, batch: list[str], bos_id: int, eos_id: int) -> int:
    """Tokenize lô hiện tại rồi ghi xuống đĩa. Trả về số token vừa ghi."""
    # TODO 10: ids = encode_documents(...)
    # TODO 11: np.asarray(ids, dtype=DTYPE).tofile(out_file)
    # TODO 12: return len(ids)


def write_meta(out_meta, tokenizer_path, n_tokens: int, vocab_size: int, sources) -> None:
    """Ghi thông tin đi kèm file .bin để lúc train kiểm tra lại."""
    # TODO 13: dict gồm dtype, n_tokens, vocab_size, tokenizer, sources → ghi JSON


def main() -> None:
    """Chạy từ dòng lệnh."""
    # TODO 14: argparse: --tokenizer, --in-dir, --out-prefix
    # TODO 15: Tokenizer.from_file(...); sorted(Path(in_dir).glob("*.txt"))
    # TODO 16: write_bin → write_meta → in số token và dung lượng


if __name__ == "__main__":
    main()
```

<details>
<summary><b>Lời giải</b></summary>

```python
"""Chuyển file .txt thành file nhị phân token ids (.bin) + meta.json."""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

DTYPE = np.uint16
MAX_VOCAB_FOR_DTYPE = int(np.iinfo(DTYPE).max) + 1
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
DOCS_PER_BATCH = 1000


def iter_documents(paths: list[Path]) -> Iterator[str]:
    """Đọc từng văn bản; các văn bản cách nhau bằng dòng trống."""
    for path in paths:
        lines: list[str] = []
        with open(path, encoding="utf-8") as text_file:
            for line in text_file:
                if line.strip():
                    lines.append(line.rstrip("\n"))
                elif lines:
                    yield "\n".join(lines)
                    lines = []
        if lines:
            yield "\n".join(lines)


def encode_documents(
    tokenizer: Tokenizer, documents: list[str], bos_id: int, eos_id: int
) -> list[int]:
    """Tokenize một lô văn bản, mỗi văn bản bọc bos ... eos."""
    ids: list[int] = []
    for encoding in tokenizer.encode_batch(documents):
        ids.append(bos_id)
        ids.extend(encoding.ids)
        ids.append(eos_id)
    return ids


def write_bin(
    paths: list[Path],
    tokenizer: Tokenizer,
    out_bin: Path,
    docs_per_batch: int = DOCS_PER_BATCH,
) -> int:
    """Ghi token ids ra file nhị phân uint16. Trả về tổng số token đã ghi."""
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > MAX_VOCAB_FOR_DTYPE:
        raise ValueError(f"vocab_size={vocab_size} vượt {MAX_VOCAB_FOR_DTYPE}, uint16 không đủ")

    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if bos_id is None or eos_id is None:
        raise ValueError(f"tokenizer thiếu {BOS_TOKEN} hoặc {EOS_TOKEN}")

    out_bin = Path(out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    batch: list[str] = []
    with open(out_bin, "wb") as out_file:
        for document in iter_documents(paths):
            batch.append(document)
            if len(batch) >= docs_per_batch:
                total += _flush(out_file, tokenizer, batch, bos_id, eos_id)
                batch = []
        if batch:
            total += _flush(out_file, tokenizer, batch, bos_id, eos_id)
    return total


def _flush(out_file, tokenizer: Tokenizer, batch: list[str], bos_id: int, eos_id: int) -> int:
    """Tokenize lô hiện tại rồi ghi xuống đĩa. Trả về số token vừa ghi."""
    ids = encode_documents(tokenizer, batch, bos_id, eos_id)
    np.asarray(ids, dtype=DTYPE).tofile(out_file)
    return len(ids)


def write_meta(
    out_meta: Path,
    tokenizer_path: Path,
    n_tokens: int,
    vocab_size: int,
    sources: list[Path],
) -> None:
    """Ghi thông tin đi kèm file .bin để lúc train kiểm tra lại."""
    meta = {
        "dtype": np.dtype(DTYPE).name,
        "n_tokens": n_tokens,
        "vocab_size": vocab_size,
        "tokenizer": str(tokenizer_path),
        "sources": [str(path) for path in sources],
    }
    Path(out_meta).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """Chạy từ dòng lệnh: đọc thư mục .txt, ghi <out_prefix>.bin và <out_prefix>_meta.json."""
    parser = argparse.ArgumentParser(description="txt -> token ids (.bin)")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--in-dir", required=True, help="thư mục chứa file .txt")
    parser.add_argument("--out-prefix", required=True, help="ví dụ dataset/pretrain/tokens/train")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    paths = sorted(Path(args.in_dir).glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"không có file .txt trong {args.in_dir}")

    out_bin = Path(f"{args.out_prefix}.bin")
    n_tokens = write_bin(paths, tokenizer, out_bin)
    write_meta(
        Path(f"{args.out_prefix}_meta.json"),
        Path(args.tokenizer),
        n_tokens,
        tokenizer.get_vocab_size(),
        paths,
    )
    print(f"Đã ghi {out_bin}: {n_tokens:,} token ({out_bin.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
```

</details>

## 5. Test — `tests/trainer/test_prepare_pretrain.py`

| Test                                             | Chứng minh                                                  |
| ------------------------------------------------ | ------------------------------------------------------------ |
| `test_iter_documents_splits_on_blank_lines`    | Tách đúng theo dòng trống                               |
| `test_iter_documents_reads_multiple_files`     | Nối nhiều file                                             |
| `test_encode_documents_wraps_each_document`    | Mỗi văn bản có bos ở đầu, eos ở cuối                |
| `test_write_bin_round_trip`                    | Đọc`.bin` rồi decode ra **đúng văn bản gốc** |
| `test_write_bin_respects_batching`             | Lô 1 hay lô 100 đều cho file giống hệt                 |
| `test_write_meta`                              | meta ghi đúng số token và dtype                          |
| `test_tokenizer_without_special_tokens_raises` | Tokenizer thiếu bos/eos bị từ chối                       |

<details>
<summary><b>Code test</b></summary>

```python
import json

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from scripts.prepare_pretrain import (
    BOS_TOKEN,
    DTYPE,
    EOS_TOKEN,
    encode_documents,
    iter_documents,
    write_bin,
    write_meta,
)
from tokenizer.train_tokenizer import build_tokenizer

DOCUMENTS = ["Hà Nội là thủ đô.", "Người Việt nói tiếng Việt.\nDòng thứ hai.", "Văn bản thứ ba."]
SAMPLE_TEXT = "\n\n".join(DOCUMENTS) + "\n"


@pytest.fixture
def tokenizer(tmp_path):
    """Tokenizer tí hon train trên chính văn bản mẫu."""
    data_file = tmp_path / "train.txt"
    data_file.write_text(SAMPLE_TEXT * 30, encoding="utf-8")
    return build_tokenizer([str(data_file)], vocab_size=400)


@pytest.fixture
def text_file(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text(SAMPLE_TEXT, encoding="utf-8")
    return path


def test_iter_documents_splits_on_blank_lines(text_file):
    assert list(iter_documents([text_file])) == DOCUMENTS


def test_iter_documents_reads_multiple_files(text_file):
    assert list(iter_documents([text_file, text_file])) == DOCUMENTS * 2


def test_encode_documents_wraps_each_document(tokenizer):
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    ids = encode_documents(tokenizer, DOCUMENTS[:1], bos_id, eos_id)

    assert ids[0] == bos_id
    assert ids[-1] == eos_id
    assert ids[1:-1] == tokenizer.encode(DOCUMENTS[0]).ids


def test_write_bin_round_trip(tokenizer, text_file, tmp_path):
    out_bin = tmp_path / "tokens" / "train.bin"

    n_tokens = write_bin([text_file], tokenizer, out_bin)
    tokens = np.fromfile(out_bin, dtype=DTYPE)

    assert tokens.size == n_tokens
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    assert int(tokens[0]) == bos_id
    assert int(tokens[-1]) == eos_id
    assert list(tokens).count(bos_id) == len(DOCUMENTS)

    # Cắt theo bos/eos rồi decode lại phải ra đúng văn bản gốc
    decoded = []
    current: list[int] = []
    for token_id in tokens.tolist():
        if token_id == bos_id:
            current = []
        elif token_id == eos_id:
            decoded.append(tokenizer.decode(current))
        else:
            current.append(token_id)
    assert decoded == DOCUMENTS


def test_write_bin_respects_batching(tokenizer, text_file, tmp_path):
    """Chia lô nhỏ hay lớn đều cho cùng kết quả."""
    one_batch = tmp_path / "a.bin"
    many_batches = tmp_path / "b.bin"

    write_bin([text_file], tokenizer, one_batch, docs_per_batch=100)
    write_bin([text_file], tokenizer, many_batches, docs_per_batch=1)

    assert one_batch.read_bytes() == many_batches.read_bytes()


def test_write_meta(tokenizer, text_file, tmp_path):
    out_meta = tmp_path / "train_meta.json"

    write_meta(out_meta, tmp_path / "tokenizer.json", 1234, tokenizer.get_vocab_size(), [text_file])
    meta = json.loads(out_meta.read_text(encoding="utf-8"))

    assert meta["n_tokens"] == 1234
    assert meta["dtype"] == "uint16"
    assert meta["vocab_size"] == tokenizer.get_vocab_size()


def test_tokenizer_without_special_tokens_raises(tmp_path, text_file):
    plain = Tokenizer(models.BPE())
    plain.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    plain.train([str(text_file)], trainers.BpeTrainer(vocab_size=300))

    with pytest.raises(ValueError, match="bos"):
        write_bin([text_file], plain, tmp_path / "out.bin")
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/trainer/test_prepare_pretrain.py -v` → **7 passed**
- [ ] Chạy thật:
  ```powershell
  python -m scripts.prepare_pretrain --in-dir dataset/pretrain/train --out-prefix dataset/pretrain/tokens/train
  python -m scripts.prepare_pretrain --in-dir dataset/pretrain/eval  --out-prefix dataset/pretrain/tokens/eval
  ```
- [ ] `train.bin` khoảng 70 MB, `train_meta.json` báo khoảng 35 triệu token
- [ ] Thêm `/dataset` vẫn nằm trong `.gitignore` (file `.bin` không commit)

---

---

# D2 — `trainer/pretrain_dataset.py`

## 1. Vị trí

```
.txt → .bin → ▶Dataset◀ → Model → loss → backward → cập nhật
```

## 2. Làm gì

Cắt dải token dài thành các **cửa sổ `T + 1` token**, rồi tách thành `x` và `y` dịch nhau 1 vị trí:

```
dải token:  [ 12  87  5  91  33  7  64  2 ... ]
cửa sổ T=4: [ 12  87  5  91  33 ]        ← đọc T+1 = 5 token
x         = [ 12  87  5  91 ]
y         = [     87  5  91  33 ]        ← y[t] = x[t+1]
```

**Nối gói (packing):** cửa sổ cắt thẳng trên dải token, không quan tâm ranh giới văn bản. Một cửa sổ có thể chứa cuối văn bản này và đầu văn bản kia, ngăn nhau bằng `<|eos|><|bos|>`. Cách này không lãng phí token nào; minimind pad từng mẫu nên tốn rất nhiều ô trống.

**`np.memmap`:** file 71MB không nạp vào RAM, hệ điều hành tự nạp phần đang đọc.

## 3. Vào → Ra

|                  | Shape / kiểu                                             |
| ---------------- | --------------------------------------------------------- |
| Vào             | `bin_path`, `seq_len`                                 |
| `dataset[i]`   | `(x, y)`, mỗi cái `(seq_len,)` kiểu `torch.long` |
| `len(dataset)` | `(tổng_token − 1) // seq_len`                         |

Trừ 1 vì cửa sổ cuối vẫn phải đọc thêm được 1 token cho `y`.

## 4. Khung code

```python
"""Dataset đọc file .bin token ids, cắt thành các cửa sổ (seq_len + 1)."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DTYPE = np.uint16


class PretrainDataset(Dataset):
    """Cắt dải token thành các cửa sổ không chồng lấn.

    Mục đích:
        Cấp (x, y) cho pretrain, trong đó y là x dịch 1 vị trí.
    Shape:
        x: (seq_len,) long, y: (seq_len,) long
    Độ phức tạp:
        O(seq_len) mỗi mẫu. File đọc bằng memmap nên không nạp hết vào RAM.
    Lưu ý:
        Token được nối gói (packing): một cửa sổ có thể chứa nhiều văn bản.
    """

    def __init__(self, bin_path: str | Path, seq_len: int) -> None:
        # TODO 1: seq_len < 1 → raise ValueError
        # TODO 2: self.tokens = np.memmap(bin_path, dtype=DTYPE, mode="r")
        # TODO 3: file có ít hơn seq_len + 1 token → raise ValueError

    def __len__(self) -> int:
        # TODO 4: (self.tokens.size - 1) // self.seq_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO 5: index âm → cộng len(self) (hỗ trợ dataset[-1])
        # TODO 6: ngoài khoảng → raise IndexError
        # TODO 7: start = index * self.seq_len
        #         window = self.tokens[start : start + self.seq_len + 1].astype(np.int64)
        #         (astype để đổi uint16 → int64, vì embedding cần torch.long)
        # TODO 8: chunk = torch.from_numpy(window); return chunk[:-1], chunk[1:]
```

<details>
<summary><b>Lời giải</b></summary>

```python
    def __init__(self, bin_path: str | Path, seq_len: int) -> None:
        """
        Args:
            bin_path: File .bin chứa token ids kiểu uint16.
            seq_len: Số token mỗi mẫu (T). Mỗi cửa sổ đọc T + 1 token.
        """
        if seq_len < 1:
            raise ValueError(f"seq_len={seq_len} phải >= 1")

        self.seq_len = seq_len
        self.tokens = np.memmap(bin_path, dtype=DTYPE, mode="r")
        if self.tokens.size < seq_len + 1:
            raise ValueError(f"file chỉ có {self.tokens.size} token, cần ít nhất {seq_len + 1}")

    def __len__(self) -> int:
        return (self.tokens.size - 1) // self.seq_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"index={index} ngoài khoảng [0, {len(self) - 1}]")

        start = index * self.seq_len
        window = self.tokens[start : start + self.seq_len + 1].astype(np.int64)
        chunk = torch.from_numpy(window)
        return chunk[:-1], chunk[1:]
```

</details>

## 5. Test — `tests/trainer/test_pretrain_dataset.py`

| Test                                     | Chứng minh                                             |
| ---------------------------------------- | ------------------------------------------------------- |
| `test_length_counts_full_windows`      | `len` đúng công thức                              |
| `test_x_and_y_are_shifted_by_one`      | **y[t] = x[t+1]**                                 |
| `test_windows_do_not_overlap`          | Cửa sổ kế tiếp bắt đầu ngay sau cửa sổ trước |
| `test_last_window_stays_inside_file`   | Không đọc quá cuối file                            |
| `test_output_is_long_tensor`           | dtype`torch.long`, shape `(seq_len,)`               |
| `test_same_index_gives_same_sample`    | Lấy lại cùng chỉ số ra cùng dữ liệu             |
| `test_negative_index_counts_from_end`  | `dataset[-1]` hoạt động                            |
| `test_index_out_of_range_raises`       | Chỉ số sai bị từ chối                              |
| `test_file_shorter_than_window_raises` | File quá ngắn bị từ chối                           |
| `test_invalid_seq_len_raises`          | `seq_len = 0` bị từ chối                           |

<details>
<summary><b>Code test</b></summary>

```python
import numpy as np
import pytest
import torch

from trainer.pretrain_dataset import DTYPE, PretrainDataset

N_TOKENS = 101
SEQ_LEN = 10


@pytest.fixture
def bin_path(tmp_path):
    """File .bin chứa 0, 1, 2, ... để dễ kiểm tra vị trí."""
    path = tmp_path / "tokens.bin"
    np.arange(N_TOKENS, dtype=DTYPE).tofile(path)
    return path


def test_length_counts_full_windows(bin_path):
    dataset = PretrainDataset(bin_path, SEQ_LEN)

    assert len(dataset) == (N_TOKENS - 1) // SEQ_LEN


def test_x_and_y_are_shifted_by_one(bin_path):
    dataset = PretrainDataset(bin_path, SEQ_LEN)

    x, y = dataset[0]

    torch.testing.assert_close(y[:-1], x[1:])
    assert x.tolist() == list(range(SEQ_LEN))
    assert y.tolist() == list(range(1, SEQ_LEN + 1))


def test_windows_do_not_overlap(bin_path):
    dataset = PretrainDataset(bin_path, SEQ_LEN)

    x0, _ = dataset[0]
    x1, _ = dataset[1]

    assert x1[0].item() == x0[-1].item() + 1


def test_last_window_stays_inside_file(bin_path):
    dataset = PretrainDataset(bin_path, SEQ_LEN)

    _, y = dataset[len(dataset) - 1]

    assert y[-1].item() <= N_TOKENS - 1


def test_output_is_long_tensor(bin_path):
    dataset = PretrainDataset(bin_path, SEQ_LEN)

    x, y = dataset[0]

    assert x.dtype == torch.long and y.dtype == torch.long
    assert x.shape == (SEQ_LEN,) and y.shape == (SEQ_LEN,)


def test_same_index_gives_same_sample(bin_path):
    dataset = PretrainDataset(bin_path, SEQ_LEN)

    torch.testing.assert_close(dataset[3][0], dataset[3][0])


def test_negative_index_counts_from_end(bin_path):
    dataset = PretrainDataset(bin_path, SEQ_LEN)

    torch.testing.assert_close(dataset[-1][0], dataset[len(dataset) - 1][0])


@pytest.mark.parametrize("index", [-100, 1000])
def test_index_out_of_range_raises(bin_path, index):
    dataset = PretrainDataset(bin_path, SEQ_LEN)

    with pytest.raises(IndexError):
        dataset[index]


def test_file_shorter_than_window_raises(tmp_path):
    path = tmp_path / "tiny.bin"
    np.arange(5, dtype=DTYPE).tofile(path)

    with pytest.raises(ValueError, match="token"):
        PretrainDataset(path, SEQ_LEN)


def test_invalid_seq_len_raises(bin_path):
    with pytest.raises(ValueError, match="seq_len"):
        PretrainDataset(bin_path, 0)
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/trainer/test_pretrain_dataset.py -v` → **11 passed**

---

---

# D3 — `trainer/losses.py`

## 1. Vị trí

```
.txt → .bin → Dataset → Model → ▶loss◀ → backward → cập nhật
```

## 2. Làm gì

Cross-entropy giữa `logits` và `y`. Dataset đã dịch nhãn sẵn, nên ở đây **không dịch thêm lần nữa**.

```
L = -(1/N) Σ log p(y[b, t])
```

Model chưa train → `L ≈ ln V`.

**Đặt loss ngoài model** (khác minimind, tính trong model): DPO, GRPO, PPO sau này cần log-prob từng token chứ không cần trung bình, nên tách ra sẽ linh hoạt hơn.

**`IGNORE_INDEX = -100`:** vị trí có nhãn này bị bỏ qua. Pretrain chưa dùng, nhưng SFT (chặng 4) cần để chỉ học phần assistant.

## 3. Vào → Ra

|             | Shape           |
| ----------- | --------------- |
| `logits`  | `(B, T, V)`   |
| `targets` | `(B, T)` long |
| Ra          | scalar          |

## 4. Khung code

```python
"""Loss cho pretrain: đoán token kế tiếp."""

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy trung bình trên mọi vị trí.

    Mục đích:
        Đo model đoán token kế tiếp tốt đến đâu.
    Shape:
        logits: (B, T, V), targets: (B, T) long  ->  scalar
    Công thức:
        L = -(1/N) Σ log p(targets[b, t]); model chưa train: L ≈ ln V.
    Độ phức tạp:
        O(B·T·V)
    Ổn định số học:
        Ép logits sang float32 vì log-softmax trên V lớn dễ tràn ở fp16.
    """
    # TODO 1: vocab_size = logits.shape[-1]
    # TODO 2: F.cross_entropy(
    #             logits.reshape(-1, vocab_size).float(),   ← gộp B và T, ép fp32
    #             targets.reshape(-1),
    #             ignore_index=IGNORE_INDEX,
    #         )
```

<details>
<summary><b>Lời giải</b></summary>

```python
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size).float(),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
```

</details>

## 5. Test — `tests/trainer/test_losses.py`

| Test                                                       | Chứng minh                                  |
| ---------------------------------------------------------- | -------------------------------------------- |
| `test_uniform_logits_give_log_vocab`                     | Logits toàn 0 → loss đúng bằng`ln V`  |
| `test_confident_correct_prediction_gives_near_zero_loss` | Đoán đúng chắc chắn → loss ≈ 0       |
| `test_ignored_positions_do_not_count`                    | Nhãn −100 bị bỏ qua                      |
| `test_fp16_logits_do_not_overflow`                       | Logits fp16 rất lớn → loss vẫn hữu hạn |
| `test_gradient_flows_to_logits`                          | Có gradient                                 |

<details>
<summary><b>Code test</b></summary>

```python
import math

import torch

from trainer.losses import IGNORE_INDEX, next_token_loss

VOCAB = 50
BATCH, SEQ = 2, 6


def test_uniform_logits_give_log_vocab():
    """Logits toàn 0 = đoán đều mọi token → loss = ln V."""
    logits = torch.zeros(BATCH, SEQ, VOCAB)
    targets = torch.randint(0, VOCAB, (BATCH, SEQ))

    loss = next_token_loss(logits, targets)

    assert abs(loss.item() - math.log(VOCAB)) < 1e-5


def test_confident_correct_prediction_gives_near_zero_loss():
    targets = torch.randint(0, VOCAB, (BATCH, SEQ))
    logits = torch.zeros(BATCH, SEQ, VOCAB)
    logits.scatter_(-1, targets.unsqueeze(-1), 50.0)

    assert next_token_loss(logits, targets).item() < 1e-5


def test_ignored_positions_do_not_count():
    """Vị trí có nhãn -100 bị bỏ qua, dùng cho SFT sau này."""
    logits = torch.zeros(BATCH, SEQ, VOCAB)
    targets = torch.randint(0, VOCAB, (BATCH, SEQ))
    logits.scatter_(-1, targets.unsqueeze(-1), 50.0)

    targets_with_ignore = targets.clone()
    targets_with_ignore[:, 0] = IGNORE_INDEX
    logits[:, 0] = 0.0  # vị trí này đoán rất tệ nhưng phải bị bỏ qua

    assert next_token_loss(logits, targets_with_ignore).item() < 1e-5


def test_fp16_logits_do_not_overflow():
    """Logits fp16 lớn: cross-entropy phải tính trong fp32 nên vẫn hữu hạn."""
    logits = torch.full((1, 2, VOCAB), 3000.0, dtype=torch.float16)
    targets = torch.zeros(1, 2, dtype=torch.long)

    loss = next_token_loss(logits, targets)

    assert torch.isfinite(loss)


def test_gradient_flows_to_logits():
    logits = torch.zeros(BATCH, SEQ, VOCAB, requires_grad=True)
    targets = torch.randint(0, VOCAB, (BATCH, SEQ))

    next_token_loss(logits, targets).backward()

    assert logits.grad is not None and logits.grad.abs().sum() > 0
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/trainer/test_losses.py -v` → **5 passed**

---

---

# D4 — `trainer/optim.py`

## 1. Vị trí

```
... → loss → backward → ▶optimizer◀ + lr schedule → cập nhật
```

## 2. Làm gì

Tạo AdamW với **2 nhóm tham số**:

| Nhóm                   | Gồm                          | Weight decay      |
| ----------------------- | ----------------------------- | ----------------- |
| Ma trận (`dim >= 2`) | Linear, embedding             | Có (ví dụ 0.1) |
| Vector (`dim == 1`)   | `weight` của RMSNorm, bias | **Không**  |

Phạt trọng số RMSNorm sẽ kéo nó về 0, làm hỏng khả năng chuẩn hoá. minimind dùng một nhóm duy nhất — đây là lỗi cần tránh.

**Bỏ qua tham số đã đóng băng** (`requires_grad=False`): nhờ vậy LoRA (chặng 5) dùng lại được hàm này mà không phải sửa.

## 3. Vào → Ra

|      |                                                |
| ---- | ---------------------------------------------- |
| Vào | `model`, `lr`, `weight_decay`            |
| Ra   | `torch.optim.AdamW` có đúng 2 param group |

`betas=(0.9, 0.95)` là mặc định phổ biến cho LLM (khác mặc định 0.999 của PyTorch): momentum bậc hai ngắn hơn, phản ứng nhanh hơn với thay đổi.

## 4. Khung code

```python
"""Tạo optimizer AdamW với 2 nhóm weight decay."""

import torch
import torch.nn as nn

DEFAULT_BETAS = (0.9, 0.95)
DEFAULT_EPS = 1e-8


def split_decay_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Tách tham số thành nhóm có và không có weight decay."""
    # TODO 1: duyệt model.parameters()
    # TODO 2: bỏ qua param.requires_grad == False
    # TODO 3: param.dim() >= 2 → nhóm decay; ngược lại → nhóm no_decay
    # TODO 4: return decay, no_decay


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = DEFAULT_BETAS,
    eps: float = DEFAULT_EPS,
) -> torch.optim.AdamW:
    """AdamW với weight decay chỉ áp cho ma trận."""
    # TODO 5: decay, no_decay = split_decay_params(model)
    # TODO 6: groups = [{"params": decay, "weight_decay": weight_decay},
    #                   {"params": no_decay, "weight_decay": 0.0}]
    # TODO 7: return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
```

<details>
<summary><b>Lời giải</b></summary>

```python
def split_decay_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Tách tham số thành nhóm có và không có weight decay.

    Quy tắc: tensor từ 2 chiều trở lên (ma trận Linear, bảng embedding) thì decay;
    tensor 1 chiều (weight của RMSNorm, bias) thì không. Tham số đã đóng băng
    (requires_grad=False, ví dụ khi train LoRA) bị bỏ qua.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return decay, no_decay


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = DEFAULT_BETAS,
    eps: float = DEFAULT_EPS,
) -> torch.optim.AdamW:
    """AdamW với weight decay chỉ áp cho ma trận."""
    decay, no_decay = split_decay_params(model)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
```

</details>

## 5. Test — `tests/trainer/test_optim.py`

| Test                                          | Chứng minh                            |
| --------------------------------------------- | -------------------------------------- |
| `test_two_groups_with_and_without_decay`    | Đúng 2 nhóm, decay đúng giá trị |
| `test_matrices_decay_vectors_do_not`        | Phân loại theo số chiều            |
| `test_every_trainable_parameter_is_covered` | Không sót tham số nào              |
| `test_norm_weights_are_in_no_decay_group`   | RMSNorm nằm đúng nhóm không decay |
| `test_frozen_parameters_are_skipped`        | Chuẩn bị cho LoRA                    |
| `test_optimizer_updates_weights`            | `step()` thật sự đổi trọng số  |

<details>
<summary><b>Code test</b></summary>

```python
import torch

from core.model import CausalLM
from trainer.optim import build_optimizer, split_decay_params

LR = 1e-3
WEIGHT_DECAY = 0.1


def test_two_groups_with_and_without_decay(tiny_config):
    optimizer = build_optimizer(CausalLM(tiny_config), lr=LR, weight_decay=WEIGHT_DECAY)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == WEIGHT_DECAY
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_matrices_decay_vectors_do_not(tiny_config):
    decay, no_decay = split_decay_params(CausalLM(tiny_config))

    assert all(param.dim() >= 2 for param in decay)
    assert all(param.dim() == 1 for param in no_decay)


def test_every_trainable_parameter_is_covered(tiny_config):
    model = CausalLM(tiny_config)
    decay, no_decay = split_decay_params(model)

    assert sum(p.numel() for p in decay) + sum(p.numel() for p in no_decay) == sum(
        p.numel() for p in model.parameters()
    )


def test_norm_weights_are_in_no_decay_group(tiny_config):
    model = CausalLM(tiny_config)
    _, no_decay = split_decay_params(model)

    norm_weight = model.model.final_norm.weight
    assert any(param is norm_weight for param in no_decay)


def test_frozen_parameters_are_skipped(tiny_config):
    """Chuẩn bị cho LoRA: tham số đóng băng không vào optimizer."""
    model = CausalLM(tiny_config)
    model.model.embed.weight.requires_grad_(False)

    decay, no_decay = split_decay_params(model)

    assert all(param.requires_grad for param in decay + no_decay)


def test_optimizer_updates_weights(tiny_config):
    model = CausalLM(tiny_config)
    optimizer = build_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)
    before = model.model.final_norm.weight.detach().clone()

    model(torch.randint(0, tiny_config.vocab_size, (2, 4))).sum().backward()
    optimizer.step()

    assert not torch.allclose(before, model.model.final_norm.weight)
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/trainer/test_optim.py -v` → **6 passed**

---

---

# D5 — `trainer/lr_scheduler.py`

## 1. Vị trí

```
... → backward → optimizer + ▶lr schedule◀ → cập nhật
```

## 2. Làm gì

Learning rate đổi theo step:

```
lr
 │      ╭──╮
 │     ╱    ╰──╮
 │    ╱          ╰───╮
 │   ╱                  ╰────╮
 │  ╱                          ╰──────  min_lr
 └─┴────────────────────────────────────  step
   warmup          cosine
```

- **Warmup:** tăng tuyến tính từ gần 0 lên `max_lr` trong `warmup_steps` bước đầu. Lúc này trạng thái Adam còn nhiễu; LR lớn ngay dễ làm loss nổ. minimind không có warmup — lỗi cần tránh.
- **Cosine:** giảm mượt về `min_lr`. Giảm mượt cho kết quả tốt hơn giảm đột ngột.

## 3. Vào → Ra

| Step                                       | Giá trị                 |
| ------------------------------------------ | ------------------------- |
| `0`                                      | `max_lr / warmup_steps` |
| `warmup_steps − 1` và `warmup_steps` | `max_lr`                |
| `max_steps − 1`                         | ≈`min_lr`              |
| `>= max_steps`                           | `min_lr`                |

Là hàm thuần, không giữ trạng thái: gọi lại với cùng `step` luôn ra cùng giá trị. Nhờ vậy resume không cần lưu gì thêm.

## 4. Khung code

```python
"""Lịch learning rate: warmup tuyến tính rồi cosine."""

import math


def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Learning rate tại `step` (đếm từ 0).

    Công thức:
        step < warmup        : max_lr · (step + 1) / warmup
        warmup <= step < max : min_lr + 0.5·(max_lr − min_lr)·(1 + cos(π · tiến_độ))
        step >= max          : min_lr
        với tiến_độ = (step − warmup) / (max − warmup)
    """
    # TODO 1: step < 0 → ValueError; max_lr < min_lr → ValueError;
    #         warmup_steps ngoài [0, max_steps] → ValueError
    # TODO 2: step < warmup_steps → max_lr * (step + 1) / warmup_steps
    # TODO 3: step >= max_steps → min_lr
    # TODO 4: progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    # TODO 5: min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
```

<details>
<summary><b>Lời giải</b></summary>

```python
    if step < 0:
        raise ValueError(f"step={step} phải >= 0")
    if max_lr < min_lr:
        raise ValueError(f"max_lr={max_lr} phải >= min_lr={min_lr}")
    if not 0 <= warmup_steps <= max_steps:
        raise ValueError(f"warmup_steps={warmup_steps} phải trong [0, max_steps={max_steps}]")

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
```

</details>

## 5. Test — `tests/trainer/test_lr_scheduler.py`

| Test                                     | Chứng minh                                |
| ---------------------------------------- | ------------------------------------------ |
| `test_first_step_is_small`             | Bước đầu chỉ bằng`max_lr / warmup` |
| `test_warmup_ends_at_max_lr`           | Hết warmup đúng bằng`max_lr`         |
| `test_last_step_is_close_to_min_lr`    | Cuối lịch ≈`min_lr`                   |
| `test_after_max_steps_stays_at_min_lr` | Vượt lịch vẫn là`min_lr`            |
| `test_lr_increases_during_warmup`      | Tăng dần trong warmup                    |
| `test_lr_decreases_after_warmup`       | Giảm dần sau warmup                      |
| `test_lr_always_within_bounds`         | Luôn trong khoảng`(0, max_lr]`         |
| `test_no_warmup_starts_at_max_lr`      | `warmup_steps = 0` vẫn chạy            |
| `test_invalid_arguments_raise`         | Tham số vô lý bị từ chối             |

<details>
<summary><b>Code test</b></summary>

```python
import pytest

from trainer.lr_scheduler import get_lr

MAX_LR = 1e-3
MIN_LR = 1e-4
WARMUP = 10
MAX_STEPS = 100
SCHEDULE = dict(max_lr=MAX_LR, min_lr=MIN_LR, warmup_steps=WARMUP, max_steps=MAX_STEPS)


def test_first_step_is_small():
    assert get_lr(0, **SCHEDULE) == pytest.approx(MAX_LR / WARMUP)


def test_warmup_ends_at_max_lr():
    assert get_lr(WARMUP - 1, **SCHEDULE) == pytest.approx(MAX_LR)
    assert get_lr(WARMUP, **SCHEDULE) == pytest.approx(MAX_LR)


def test_last_step_is_close_to_min_lr():
    assert get_lr(MAX_STEPS - 1, **SCHEDULE) == pytest.approx(MIN_LR, rel=0.01)


def test_after_max_steps_stays_at_min_lr():
    assert get_lr(MAX_STEPS, **SCHEDULE) == MIN_LR
    assert get_lr(MAX_STEPS * 10, **SCHEDULE) == MIN_LR


def test_lr_increases_during_warmup():
    values = [get_lr(step, **SCHEDULE) for step in range(WARMUP)]

    assert values == sorted(values)
    assert values[0] < values[-1]


def test_lr_decreases_after_warmup():
    values = [get_lr(step, **SCHEDULE) for step in range(WARMUP, MAX_STEPS)]

    assert values == sorted(values, reverse=True)


def test_lr_always_within_bounds():
    for step in range(MAX_STEPS + 20):
        assert 0 < get_lr(step, **SCHEDULE) <= MAX_LR


def test_no_warmup_starts_at_max_lr():
    assert get_lr(0, max_lr=MAX_LR, min_lr=MIN_LR, warmup_steps=0, max_steps=MAX_STEPS) == MAX_LR


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(step=-1, max_lr=MAX_LR, min_lr=MIN_LR, warmup_steps=WARMUP, max_steps=MAX_STEPS),
        dict(step=0, max_lr=MIN_LR, min_lr=MAX_LR, warmup_steps=WARMUP, max_steps=MAX_STEPS),
        dict(step=0, max_lr=MAX_LR, min_lr=MIN_LR, warmup_steps=MAX_STEPS + 1, max_steps=MAX_STEPS),
    ],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        get_lr(**kwargs)
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/trainer/test_lr_scheduler.py -v` → **11 passed**

---

---

# D6 — `trainer/checkpoint.py`

## 1. Vị trí

```
... → cập nhật → ▶lưu checkpoint◀ (định kỳ)
```

## 2. Làm gì

Phân biệt **hai việc khác hẳn nhau**:

|           | `save_checkpoint` / `load_checkpoint`           | `load_weights`                       |
| --------- | --------------------------------------------------- | -------------------------------------- |
| Dùng khi | Máy tắt giữa chừng, chạy tiếp đúng run này | Bắt đầu giai đoạn mới (SFT, DPO) |
| Nạp lại | Trọng số + optimizer + scaler + RNG + step        | Chỉ trọng số                        |

**Ba điểm quan trọng:**

1. **Lưu float32.** minimind lưu `half()` nên resume xong trọng số không còn khớp — mỗi lần lưu lại mất thêm số lẻ.
2. **Lưu trạng thái RNG** của Python, NumPy, PyTorch. Không có thì thứ tự dữ liệu và dropout sau khi resume sẽ khác.
3. **Ghi ra file tạm rồi đổi tên.** Mất điện giữa lúc ghi thì file cũ vẫn nguyên; nếu ghi đè trực tiếp sẽ mất cả hai.

## 3. Vào → Ra

File `.pt` chứa dict:

```python
{
  "step": 1200,
  "model": {...},          # float32
  "optimizer": {...},
  "scaler": {...} | None,
  "rng": {"python": ..., "numpy": ..., "torch": ..., "cuda": ...},
  "model_config": {...},   # để biết checkpoint này thuộc kiến trúc nào
  "train_config": {...},
}
```

## 4. Khung code

```python
"""Lưu và nạp checkpoint."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _rng_state() -> dict[str, Any]:
    """Trạng thái sinh số ngẫu nhiên của Python, NumPy, PyTorch."""
    # TODO 1: dict với random.getstate(), np.random.get_state(), torch.get_rng_state()
    # TODO 2: nếu có CUDA thì thêm torch.cuda.get_rng_state_all()


def _restore_rng(state: dict[str, Any]) -> None:
    # TODO 3: random.setstate / np.random.set_state / torch.set_rng_state
    # TODO 4: nếu có "cuda" và máy có CUDA → torch.cuda.set_rng_state_all


def save_checkpoint(path, model, optimizer, step, model_config, train_config, scaler=None) -> None:
    """Lưu đủ mọi thứ để chạy tiếp đúng run này."""
    # TODO 5: tạo thư mục cha
    # TODO 6: payload = {step, model (ÉP .float()), optimizer, scaler, rng, 2 config .to_dict()}
    # TODO 7: torch.save vào path + ".tmp" rồi tmp_path.replace(path)


def load_checkpoint(path, model, optimizer, scaler=None, map_location="cpu") -> int:
    """Nạp lại toàn bộ trạng thái. Trả về step đã chạy xong."""
    # TODO 8: torch.load(..., weights_only=False)   ← cần False vì payload có state RNG
    # TODO 9: model.load_state_dict, optimizer.load_state_dict, scaler (nếu có), _restore_rng
    # TODO 10: return int(payload["step"])


def load_weights(path, model, map_location="cpu") -> None:
    """Chỉ nạp trọng số, bỏ optimizer và RNG."""
    # TODO 11: torch.load rồi model.load_state_dict(payload["model"])
```

<details>
<summary><b>Lời giải</b></summary>

```python
def _rng_state() -> dict[str, Any]:
    """Trạng thái sinh số ngẫu nhiên của Python, NumPy, PyTorch."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_config: Any,
    train_config: Any,
    scaler: Any = None,
) -> None:
    """Lưu đủ mọi thứ để chạy tiếp đúng run này."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "model": {key: value.detach().cpu().float() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": _rng_state(),
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any = None,
    map_location: Any = "cpu",
) -> int:
    """Nạp lại toàn bộ trạng thái. Trả về step đã chạy xong."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    _restore_rng(payload["rng"])
    return int(payload["step"])


def load_weights(path: str | Path, model: nn.Module, map_location: Any = "cpu") -> None:
    """Chỉ nạp trọng số, bỏ optimizer và RNG — dùng khi bắt đầu giai đoạn mới."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
```

</details>

## 5. Test — `tests/trainer/test_checkpoint.py`

| Test                                              | Chứng minh                                                                           |
| ------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `test_resume_matches_uninterrupted_run`         | **Tiêu chí xong chặng 2:** 4 bước liền mạch = 2 bước + resume 2 bước |
| `test_saved_weights_are_float32`                | Model fp16 vẫn lưu ra fp32                                                          |
| `test_checkpoint_stores_step_and_configs`       | Có step và 2 config                                                                 |
| `test_load_checkpoint_restores_optimizer_state` | Momentum của Adam được khôi phục                                                |
| `test_random_state_is_restored`                 | Số ngẫu nhiên sau khi load giống hệt lúc lưu                                   |
| `test_load_weights_ignores_optimizer`           | Nạp trọng số không kéo theo optimizer                                            |
| `test_atomic_write_leaves_no_temp_file`         | Không để lại file`.tmp`                                                         |

Test đầu tiên gọi `train(...)` của D9, nên viết sau khi có D9 (hoặc tạm bỏ qua rồi thêm lại).

<details>
<summary><b>Code test</b></summary>

```python
import torch

from core.model import CausalLM
from trainer.checkpoint import load_checkpoint, load_weights, save_checkpoint
from trainer.optim import build_optimizer
from trainer.train_pretrain import CHECKPOINT_NAME, train


def make_model_and_optimizer(model_config):
    model = CausalLM(model_config)
    return model, build_optimizer(model, lr=1e-3, weight_decay=0.1)


def test_resume_matches_uninterrupted_run(tiny_model_config, make_train_config, tmp_path):
    """4 bước liền mạch phải ra đúng trọng số như 2 bước + resume 2 bước."""
    straight = make_train_config(max_steps=4, out_dir=str(tmp_path / "straight"))
    train(tiny_model_config, straight)

    first_half = make_train_config(max_steps=2, out_dir=str(tmp_path / "resumed"))
    train(tiny_model_config, first_half)
    second_half = make_train_config(max_steps=4, out_dir=str(tmp_path / "resumed"))
    train(tiny_model_config, second_half, resume=True)

    straight_state = torch.load(
        tmp_path / "straight" / CHECKPOINT_NAME, map_location="cpu", weights_only=False
    )["model"]
    resumed_state = torch.load(
        tmp_path / "resumed" / CHECKPOINT_NAME, map_location="cpu", weights_only=False
    )["model"]

    for key, value in straight_state.items():
        torch.testing.assert_close(resumed_state[key], value, msg=f"khác nhau ở {key}")


def test_saved_weights_are_float32(tiny_model_config, make_train_config, tmp_path):
    """Lưu fp16 như minimind sẽ làm resume không khớp."""
    model, optimizer = make_model_and_optimizer(tiny_model_config)
    model.half()
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, optimizer, 1, tiny_model_config, make_train_config())
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert all(tensor.dtype == torch.float32 for tensor in payload["model"].values())


def test_checkpoint_stores_step_and_configs(tiny_model_config, make_train_config, tmp_path):
    model, optimizer = make_model_and_optimizer(tiny_model_config)
    train_config = make_train_config()
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, optimizer, 7, tiny_model_config, train_config)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["step"] == 7
    assert payload["model_config"] == tiny_model_config.to_dict()
    assert payload["train_config"] == train_config.to_dict()


def test_load_checkpoint_restores_optimizer_state(tiny_model_config, make_train_config, tmp_path):
    model, optimizer = make_model_and_optimizer(tiny_model_config)
    model(torch.randint(0, tiny_model_config.vocab_size, (2, 4))).sum().backward()
    optimizer.step()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, 3, tiny_model_config, make_train_config())

    fresh_model, fresh_optimizer = make_model_and_optimizer(tiny_model_config)
    step = load_checkpoint(path, fresh_model, fresh_optimizer)

    assert step == 3

    saved_state = optimizer.state_dict()["state"]
    loaded_state = fresh_optimizer.state_dict()["state"]
    assert set(loaded_state) == set(saved_state)
    for index, entry in saved_state.items():
        torch.testing.assert_close(loaded_state[index]["exp_avg"], entry["exp_avg"])
        torch.testing.assert_close(loaded_state[index]["exp_avg_sq"], entry["exp_avg_sq"])


def test_random_state_is_restored(tiny_model_config, make_train_config, tmp_path):
    model, optimizer = make_model_and_optimizer(tiny_model_config)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, 1, tiny_model_config, make_train_config())

    expected = torch.rand(3)
    load_checkpoint(path, model, optimizer)

    torch.testing.assert_close(torch.rand(3), expected)


def test_load_weights_ignores_optimizer(tiny_model_config, make_train_config, tmp_path):
    """Bắt đầu giai đoạn mới: lấy trọng số, bỏ trạng thái optimizer."""
    model, optimizer = make_model_and_optimizer(tiny_model_config)
    model(torch.randint(0, tiny_model_config.vocab_size, (2, 4))).sum().backward()
    optimizer.step()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, 5, tiny_model_config, make_train_config())

    fresh_model, fresh_optimizer = make_model_and_optimizer(tiny_model_config)
    load_weights(path, fresh_model)

    for key, value in model.state_dict().items():
        torch.testing.assert_close(fresh_model.state_dict()[key], value.float())
    assert fresh_optimizer.state_dict()["state"] == {}


def test_atomic_write_leaves_no_temp_file(tiny_model_config, make_train_config, tmp_path):
    model, optimizer = make_model_and_optimizer(tiny_model_config)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, optimizer, 1, tiny_model_config, make_train_config())

    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/trainer/test_checkpoint.py -v` → **7 passed** (sau khi có D9)

---

---

# D7 — `trainer/metrics.py`

## 1. Vị trí

```
... → cập nhật → ▶log số liệu◀
```

## 2. Làm gì

Bốn số cần theo dõi mỗi lần log:

| Số                 | Ý nghĩa khi bất thường                                                    |
| ------------------- | ------------------------------------------------------------------------------ |
| **loss**      | Không giảm → sai dữ liệu, sai LR, hoặc sai code                          |
| **grad norm** | Vọt lên → loss sắp nổ; luôn bằng đúng`grad_clip` → clip quá chặt |
| **tokens/s**  | Tụt → nghẽn ở đọc dữ liệu                                              |
| **VRAM**      | Gần giới hạn → giảm batch hoặc seq_len                                   |

## 3. Vào → Ra

```
step    120/2000 | loss  7.412 | lr 5.83e-04 | gnorm  1.204 |    12,431 tok/s | vram   2103 MB
```

## 4. Khung code

```python
"""Đo và in số liệu khi train."""

import time
from collections.abc import Iterable

import torch

BYTES_PER_MB = 1e6


def global_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Chuẩn L2 của toàn bộ gradient, gộp mọi tham số: sqrt( Σ ||grad||² )."""
    # TODO 1: cộng dồn grad.detach().float().pow(2).sum() cho param có grad
    # TODO 2: return căn bậc hai


def max_vram_mb() -> float:
    """VRAM cao nhất đã dùng (MB). Trả 0 nếu chạy CPU."""
    # TODO 3: không có CUDA → 0.0; có → torch.cuda.max_memory_allocated() / BYTES_PER_MB


class Throughput:
    """Đếm tốc độ xử lý token."""

    # TODO 4: __init__ gọi reset()
    # TODO 5: reset(): lưu time.perf_counter() và đặt bộ đếm token = 0
    # TODO 6: update(n_tokens): cộng dồn
    # TODO 7: tokens_per_second(): token / thời gian trôi qua (0 nếu chưa trôi)


def format_log(step, max_steps, loss, lr, grad_norm, tokens_per_second, vram_mb) -> str:
    """Một dòng log gọn cho mỗi lần in."""
    # TODO 8: dùng f-string căn cột như ví dụ ở mục 3
```

<details>
<summary><b>Lời giải</b></summary>

```python
def global_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Chuẩn L2 của toàn bộ gradient, gộp mọi tham số.

    Công thức:
        sqrt( Σ_p ||grad_p||² )
    Lưu ý:
        Grad norm tăng vọt thường báo trước loss sắp nổ. Luôn log giá trị này.
    """
    total = torch.zeros((), dtype=torch.float32)
    for param in parameters:
        if param.grad is not None:
            total += param.grad.detach().float().pow(2).sum().cpu()
    return total.sqrt().item()


def max_vram_mb() -> float:
    """VRAM cao nhất đã dùng (MB). Trả 0 nếu chạy CPU."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / BYTES_PER_MB


class Throughput:
    """Đếm tốc độ xử lý token."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._start = time.perf_counter()
        self._tokens = 0

    def update(self, n_tokens: int) -> None:
        self._tokens += n_tokens

    def tokens_per_second(self) -> float:
        elapsed = time.perf_counter() - self._start
        return self._tokens / elapsed if elapsed > 0 else 0.0


def format_log(
    step: int,
    max_steps: int,
    loss: float,
    lr: float,
    grad_norm: float,
    tokens_per_second: float,
    vram_mb: float,
) -> str:
    """Một dòng log gọn cho mỗi lần in."""
    return (
        f"step {step:>6}/{max_steps} | loss {loss:6.3f} | lr {lr:.2e} | "
        f"gnorm {grad_norm:6.3f} | {tokens_per_second:9,.0f} tok/s | vram {vram_mb:6.0f} MB"
    )
```

</details>

## 5. Test — `tests/trainer/test_metrics.py`

| Test                                               | Chứng minh                               |
| -------------------------------------------------- | ----------------------------------------- |
| `test_grad_norm_matches_manual_computation`      | Grad toàn 1 trên 3 số → norm = √3    |
| `test_grad_norm_ignores_parameters_without_grad` | Tham số chưa có grad không tính vào |
| `test_throughput_counts_tokens`                  | Đếm được token/s                     |
| `test_throughput_reset_clears_counter`           | `reset()` xoá bộ đếm                |
| `test_max_vram_is_zero_on_cpu`                   | Chạy CPU trả 0                          |
| `test_format_log_contains_every_number`          | Dòng log có đủ mọi số               |

<details>
<summary><b>Code test</b></summary>

```python
import time

import torch

from trainer.metrics import Throughput, format_log, global_grad_norm, max_vram_mb


def test_grad_norm_matches_manual_computation():
    first = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    second = torch.nn.Parameter(torch.tensor([[12.0]]))
    (first.sum() + second.sum()).backward()  # grad toàn 1 → norm = sqrt(3)

    assert global_grad_norm([first, second]) == torch.tensor(3.0).sqrt().item()


def test_grad_norm_ignores_parameters_without_grad():
    with_grad = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    without_grad = torch.nn.Parameter(torch.tensor([10.0]))
    (with_grad * torch.tensor([1.0, 2.0])).sum().backward()  # grad = [1, 2]

    expected = torch.tensor([1.0, 2.0]).norm().item()

    assert global_grad_norm([with_grad, without_grad]) == expected
    assert without_grad.grad is None


def test_throughput_counts_tokens():
    meter = Throughput()
    meter.update(1000)
    time.sleep(0.01)

    assert meter.tokens_per_second() > 0


def test_throughput_reset_clears_counter():
    meter = Throughput()
    meter.update(1000)
    meter.reset()
    time.sleep(0.01)

    assert meter.tokens_per_second() == 0.0


def test_max_vram_is_zero_on_cpu():
    if not torch.cuda.is_available():
        assert max_vram_mb() == 0.0


def test_format_log_contains_every_number():
    line = format_log(
        step=7, max_steps=100, loss=1.234, lr=0.001, grad_norm=0.5, tokens_per_second=1234, vram_mb=42
    )

    for text in ("7", "100", "1.234", "loss", "lr", "gnorm", "tok/s", "vram"):
        assert text in line
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/trainer/test_metrics.py -v` → **6 passed**

---

---

# D8 — `trainer/train_config.py`

## 1. Vị trí

Tham số cho **một lần chạy**, tách khỏi `ModelConfig` (kiến trúc).

|                      | `ModelConfig`                                        | `TrainConfig`                     |
| -------------------- | ------------------------------------------------------ | ----------------------------------- |
| Nội dung            | d_model, n_layers, vocab_size...                       | lr, batch_size, max_steps...        |
| Đổi được không | **Không** — đổi là trọng số cũ vô dụng | Được, mỗi lần chạy một khác |
| Dùng ở chặng nào | Mọi chặng                                            | Riêng từng chặng                 |

## 2. Làm gì

Dataclass frozen giống `ModelConfig`, thêm `from_json`. Kiểm tra ràng buộc: số nguyên dương, `0 <= warmup_steps <= max_steps`, `0 <= min_lr <= lr`, `dtype` thuộc `("fp32", "fp16", "bf16")`.

**Property `tokens_per_step`** = `batch_size × accum_steps × seq_len` — số token model học mỗi bước cập nhật. Đây là con số quyết định tốc độ học, không phải `batch_size` đơn lẻ.

## 3. Vào → Ra

`configs/pretrain_tiny.json` mẫu (chạy được trên CPU/GPU 4GB, model ~5 triệu tham số):

```json
{
  "train_bin": "dataset/pretrain/tokens/train.bin",
  "eval_bin": "dataset/pretrain/tokens/eval.bin",
  "out_dir": "checkpoints",
  "seq_len": 128,
  "batch_size": 8,
  "accum_steps": 4,
  "max_steps": 2000,
  "warmup_steps": 100,
  "lr": 0.0006,
  "min_lr": 0.00006,
  "weight_decay": 0.1,
  "grad_clip": 1.0,
  "log_interval": 10,
  "eval_interval": 200,
  "eval_batches": 20,
  "save_interval": 500,
  "dtype": "fp32",
  "device": "cpu",
  "seed": 0
}
```

Và `configs/model_tiny.json`:

```json
{
  "vocab_size": 16000,
  "d_model": 128,
  "n_layers": 2,
  "n_heads": 4,
  "ffn_hidden": 320,
  "max_seq_len": 256,
  "rope_theta": 10000.0,
  "norm_eps": 1e-6,
  "tie_embedding": true
}
```

Khi có GPU: đổi `device` thành `"cuda"`, `dtype` thành `"bf16"`, tăng `d_model`, `n_layers`, `batch_size`.

## 4. Khung code

Cấu trúc giống hệt `core/config.py`, chỉ khác danh sách trường.

```python
"""Tham số cho một lần chạy pretrain (khác ModelConfig: kiến trúc)."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

ALLOWED_DTYPES = ("fp32", "fp16", "bf16")
POSITIVE_INT_FIELDS = (
    "seq_len", "batch_size", "accum_steps", "max_steps",
    "log_interval", "eval_interval", "eval_batches", "save_interval",
)
POSITIVE_FLOAT_FIELDS = ("lr", "grad_clip")


@dataclass(frozen=True)
class TrainConfig:
    # TODO 1: khai báo 19 trường: train_bin, eval_bin, out_dir (str);
    #         seq_len, batch_size, accum_steps, max_steps, warmup_steps (int);
    #         lr, min_lr, weight_decay, grad_clip (float);
    #         log_interval, eval_interval, eval_batches, save_interval (int);
    #         dtype, device (str); seed (int)

    def __post_init__(self) -> None:
        # TODO 2: POSITIVE_INT_FIELDS + seed phải là int (chặn bool); các trường trừ seed phải > 0
        # TODO 3: 0 <= warmup_steps <= max_steps
        # TODO 4: POSITIVE_FLOAT_FIELDS phải là số hữu hạn > 0
        # TODO 5: 0 <= min_lr <= lr; weight_decay >= 0; dtype thuộc ALLOWED_DTYPES

    @property
    def tokens_per_step(self) -> int:
        # TODO 6: batch_size * accum_steps * seq_len

    @classmethod
    def from_dict(cls, data): ...   # TODO 7: sao chép cách làm của ModelConfig.from_dict

    @classmethod
    def from_json(cls, path):       # TODO 8: đọc file JSON rồi gọi from_dict
        ...

    def to_dict(self):              # TODO 9: asdict(self)
        ...
```

<details>
<summary><b>Lời giải</b> (phần thân, bỏ docstring cho gọn)</summary>

```python
@dataclass(frozen=True)
class TrainConfig:
    train_bin: str
    eval_bin: str
    out_dir: str
    seq_len: int
    batch_size: int
    accum_steps: int
    max_steps: int
    warmup_steps: int
    lr: float
    min_lr: float
    weight_decay: float
    grad_clip: float
    log_interval: int
    eval_interval: int
    eval_batches: int
    save_interval: int
    dtype: str
    device: str
    seed: int

    def __post_init__(self) -> None:
        for name in POSITIVE_INT_FIELDS + ("seed",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name}={value!r} phải là int")
            if name != "seed" and value <= 0:
                raise ValueError(f"{name}={value} phải > 0")

        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int):
            raise ValueError(f"warmup_steps={self.warmup_steps!r} phải là int")
        if not 0 <= self.warmup_steps <= self.max_steps:
            raise ValueError(
                f"warmup_steps={self.warmup_steps} phải trong [0, max_steps={self.max_steps}]"
            )

        for name in POSITIVE_FLOAT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name}={value!r} phải là số thực")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}={value} phải hữu hạn và > 0")

        if not 0 <= self.min_lr <= self.lr:
            raise ValueError(f"min_lr={self.min_lr} phải trong [0, lr={self.lr}]")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay={self.weight_decay} phải >= 0")
        if self.dtype not in ALLOWED_DTYPES:
            raise ValueError(f"dtype={self.dtype!r} phải thuộc {ALLOWED_DTYPES}")

    @property
    def tokens_per_step(self) -> int:
        """Số token model học trong 1 bước cập nhật."""
        return self.batch_size * self.accum_steps * self.seq_len

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        if not isinstance(data, dict):
            raise ValueError(f"from_dict cần dict, nhận được {type(data).__name__}")

        valid_names = {field.name for field in fields(cls)}
        unknown = set(data) - valid_names
        missing = valid_names - set(data)

        errors = []
        if unknown:
            errors.append(f"khóa không hợp lệ: {sorted(unknown)}")
        if missing:
            errors.append(f"khóa bị thiếu: {sorted(missing)}")
        if errors:
            raise ValueError("; ".join(errors))

        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

</details>

## 5. Test — `tests/trainer/test_train_config.py`

Cần thêm `tests/trainer/conftest.py` với 3 fixture dùng chung cho cả phần D:

<details>
<summary><b>Code <code>tests/trainer/conftest.py</code></b></summary>

```python
"""Fixture dùng chung cho test phần trainer."""

import numpy as np
import pytest

from core.config import ModelConfig
from trainer.pretrain_dataset import DTYPE
from trainer.train_config import TrainConfig

VOCAB = 64
N_TOKENS = 2000


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB,
        d_model=32,
        n_layers=1,
        n_heads=2,
        ffn_hidden=64,
        max_seq_len=32,
        rope_theta=10000.0,
        norm_eps=1e-6,
        tie_embedding=True,
    )


@pytest.fixture
def token_bins(tmp_path):
    """train.bin và eval.bin chứa token ngẫu nhiên có seed cố định."""
    rng = np.random.default_rng(0)
    paths = {}
    for name in ("train", "eval"):
        path = tmp_path / f"{name}.bin"
        rng.integers(0, VOCAB, size=N_TOKENS, dtype=DTYPE).tofile(path)
        paths[name] = path
    return paths


@pytest.fixture
def make_train_config(tmp_path, token_bins):
    """Trả về hàm tạo TrainConfig, cho phép ghi đè từng trường."""

    def _make(**overrides) -> TrainConfig:
        defaults = dict(
            train_bin=str(token_bins["train"]),
            eval_bin=str(token_bins["eval"]),
            out_dir=str(tmp_path / "out"),
            seq_len=16,
            batch_size=2,
            accum_steps=2,
            max_steps=3,
            warmup_steps=1,
            lr=1e-3,
            min_lr=1e-4,
            weight_decay=0.1,
            grad_clip=1.0,
            log_interval=1000,
            eval_interval=1000,
            eval_batches=2,
            save_interval=1000,
            dtype="fp32",
            device="cpu",
            seed=0,
        )
        defaults.update(overrides)
        return TrainConfig(**defaults)

    return _make
```

</details>

| Test                                                      | Chứng minh                         |
| --------------------------------------------------------- | ----------------------------------- |
| `test_valid_config_is_created`                          | Config hợp lệ tạo được        |
| `test_tokens_per_step`                                  | `4 × 8 × 256`                   |
| `test_json_round_trip`                                  | Ghi JSON rồi đọc lại bằng nhau |
| `test_unknown_key_raises` / `test_missing_key_raises` | Khóa lạ và khóa thiếu          |
| `test_invalid_seq_len_raises`                           | `0`, `-1`, `True`, `2.5`    |
| `test_warmup_larger_than_max_steps_raises`              | Ràng buộc warmup                  |
| `test_min_lr_above_lr_raises`                           | Ràng buộc LR                      |
| `test_unknown_dtype_raises`                             | `"int8"` bị từ chối            |
| `test_negative_weight_decay_raises`                     | decay âm bị từ chối             |
| `test_config_is_frozen`                                 | Không gán lại được            |

<details>
<summary><b>Code test</b></summary>

```python
import json

import pytest

from trainer.train_config import TrainConfig


def test_valid_config_is_created(make_train_config):
    config = make_train_config()

    assert config.seq_len == 16
    assert config.dtype == "fp32"


def test_tokens_per_step(make_train_config):
    config = make_train_config(batch_size=4, accum_steps=8, seq_len=256)

    assert config.tokens_per_step == 4 * 8 * 256


def test_json_round_trip(make_train_config, tmp_path):
    config = make_train_config()
    path = tmp_path / "train.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

    assert TrainConfig.from_json(path) == config


def test_unknown_key_raises(make_train_config):
    data = {**make_train_config().to_dict(), "foo": 1}

    with pytest.raises(ValueError, match="foo"):
        TrainConfig.from_dict(data)


def test_missing_key_raises(make_train_config):
    data = make_train_config().to_dict()
    del data["seq_len"]

    with pytest.raises(ValueError, match="seq_len"):
        TrainConfig.from_dict(data)


@pytest.mark.parametrize("bad_value", [0, -1, True, 2.5])
def test_invalid_seq_len_raises(make_train_config, bad_value):
    with pytest.raises(ValueError, match="seq_len"):
        make_train_config(seq_len=bad_value)


def test_warmup_larger_than_max_steps_raises(make_train_config):
    with pytest.raises(ValueError, match="warmup_steps"):
        make_train_config(max_steps=10, warmup_steps=11)


def test_min_lr_above_lr_raises(make_train_config):
    with pytest.raises(ValueError, match="min_lr"):
        make_train_config(lr=1e-4, min_lr=1e-3)


def test_unknown_dtype_raises(make_train_config):
    with pytest.raises(ValueError, match="dtype"):
        make_train_config(dtype="int8")


def test_negative_weight_decay_raises(make_train_config):
    with pytest.raises(ValueError, match="weight_decay"):
        make_train_config(weight_decay=-0.1)


def test_config_is_frozen(make_train_config):
    config = make_train_config()

    with pytest.raises(Exception):
        config.seq_len = 32
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/trainer/test_train_config.py -v` → **13 passed**
- [ ] Đã tạo `configs/model_tiny.json` và `configs/pretrain_tiny.json`

---

---

# D9 — `trainer/train_pretrain.py`

## 1. Vị trí

**Ghép tất cả:**

```
Dataset → DataLoader → model → loss → backward ×accum → clip → optimizer.step → log/eval/save
```

## 2. Làm gì

Một bước cập nhật (`step`) gồm:

```
1. lr = get_lr(step)              →  gán vào optimizer.param_groups
2. zero_grad
3. lặp accum_steps lần:
       lấy micro-batch (x, y)
       logits = model(x)
       loss = next_token_loss(logits, y) / accum_steps    ← chia để lấy trung bình
       loss.backward()                                     ← cộng dồn gradient
4. clip_grad_norm_(grad_clip)
5. optimizer.step()
6. định kỳ: log, val loss, lưu checkpoint
```

**Gradient accumulation:** batch hiệu dụng = `batch_size × accum_steps`, nhưng VRAM chỉ tốn bằng `batch_size`. Đây là cách train batch lớn trên GPU 4GB.

**Đếm theo step, không theo epoch:** dữ liệu lặp vô hạn qua `endless(loader)`. Pretrain thường chưa chạy hết 1 epoch.

**Resume phải bỏ qua đúng số batch đã dùng.** Nếu không, sau khi resume model học lại đúng dữ liệu cũ và kết quả khác với chạy liền mạch — test `test_resume_matches_uninterrupted_run` bắt được lỗi này.

**Mixed precision:** `bf16` hoặc `fp16` chỉ bật khi chạy CUDA. `fp16` cần `GradScaler`; `bf16` thì không, nhưng để chung một đường code cho gọn (`GradScaler(enabled=False)` không làm gì cả).

## 3. Vào → Ra

|                         |                                                             |
| ----------------------- | ----------------------------------------------------------- |
| Vào                    | `configs/model_tiny.json`, `configs/pretrain_tiny.json` |
| Ra                      | `checkpoints/pretrain.pt`, log trên màn hình           |
| `train(...)` trả về | `{"step": ..., "train_loss": ..., "eval_loss": ...}`      |

## 4. Khung code

```python
"""Vòng lặp pretrain: ghép dataset, model, optimizer, scheduler, checkpoint."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from core.config import ModelConfig
from core.model import CausalLM
from trainer.checkpoint import load_checkpoint, save_checkpoint
from trainer.losses import next_token_loss
from trainer.lr_scheduler import get_lr
from trainer.metrics import Throughput, format_log, global_grad_norm, max_vram_mb
from trainer.optim import build_optimizer
from trainer.pretrain_dataset import PretrainDataset
from trainer.train_config import TrainConfig

TORCH_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
CHECKPOINT_NAME = "pretrain.pt"


def build_loader(dataset, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    """DataLoader có seed cố định, bỏ batch lẻ cuối để mọi batch cùng kích thước."""
    # TODO 1: generator = torch.Generator().manual_seed(seed)
    # TODO 2: DataLoader(..., drop_last=True, generator=generator nếu shuffle)


def endless(loader: DataLoader):
    """Lặp vô hạn qua loader: pretrain đếm theo step, không theo epoch."""
    # TODO 3: while True: yield from loader


@torch.no_grad()
def evaluate(model, loader, n_batches: int, device, autocast) -> float:
    """Val loss trung bình trên n_batches batch đầu tiên."""
    # TODO 4: model.eval(); cộng dồn loss trên n_batches batch; model.train(); trả trung bình


def train(model_config: ModelConfig, train_config: TrainConfig, resume: bool = False) -> dict:
    """Chạy pretrain. Trả về thống kê cuối cùng."""
    # TODO 5:  seed, device, dtype; use_amp chỉ bật khi CUDA và dtype khác fp32
    # TODO 6:  autocast(): torch.autocast(...) nếu use_amp, ngược lại nullcontext()
    # TODO 7:  scaler = torch.amp.GradScaler(device.type, enabled=(use_amp và dtype là fp16))
    # TODO 8:  train_loader, eval_loader từ PretrainDataset
    # TODO 9:  model = CausalLM(model_config).to(device); optimizer = build_optimizer(...)
    # TODO 10: nếu resume: load_checkpoint → start_step; không có file → FileNotFoundError
    # TODO 11: batches = endless(train_loader); BỎ QUA start_step * accum_steps batch
    # TODO 12: vòng for step in range(start_step, max_steps):
    #              lr = get_lr(...) → gán cho mọi param_group
    #              optimizer.zero_grad(set_to_none=True)
    #              lặp accum_steps: lấy batch, forward trong autocast,
    #                  loss / accum_steps, scaler.scale(loss).backward(), cộng dồn loss
    #              scaler.unscale_(optimizer)
    #              grad_norm = clip_grad_norm_(model.parameters(), grad_clip)
    #              scaler.step(optimizer); scaler.update()
    #              định kỳ: in log, evaluate, save_checkpoint
    # TODO 13: lưu checkpoint lần cuối; trả dict thống kê


def main() -> None:
    """python -m trainer.train_pretrain --model-config ... --train-config ... [--resume]"""
    # TODO 14: argparse; đọc 2 file JSON; gọi train(...)
```

<details>
<summary><b>Lời giải</b></summary>

```python
def build_loader(dataset: PretrainDataset, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    """DataLoader có seed cố định, bỏ batch lẻ cuối để mọi batch cùng kích thước."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        generator=generator if shuffle else None,
    )


def endless(loader: DataLoader):
    """Lặp vô hạn qua loader: pretrain đếm theo step, không theo epoch."""
    while True:
        yield from loader


@torch.no_grad()
def evaluate(model: CausalLM, loader: DataLoader, n_batches: int, device: torch.device, autocast) -> float:
    """Val loss trung bình trên n_batches batch đầu tiên."""
    model.eval()
    total, count = 0.0, 0
    for batch_index, (x, y) in enumerate(loader):
        if batch_index >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast():
            total += next_token_loss(model(x), y).item()
        count += 1
    model.train()
    return total / max(count, 1)


def train(model_config: ModelConfig, train_config: TrainConfig, resume: bool = False) -> dict[str, Any]:
    """Chạy pretrain. Trả về thống kê cuối cùng."""
    torch.manual_seed(train_config.seed)
    device = torch.device(train_config.device)
    dtype = TORCH_DTYPES[train_config.dtype]
    use_amp = dtype is not torch.float32 and device.type == "cuda"

    def autocast():
        return torch.autocast(device_type=device.type, dtype=dtype) if use_amp else nullcontext()

    scaler = torch.amp.GradScaler(device.type, enabled=(use_amp and dtype is torch.float16))

    train_loader = build_loader(
        PretrainDataset(train_config.train_bin, train_config.seq_len),
        train_config.batch_size,
        train_config.seed,
        shuffle=True,
    )
    eval_loader = build_loader(
        PretrainDataset(train_config.eval_bin, train_config.seq_len),
        train_config.batch_size,
        train_config.seed,
        shuffle=False,
    )

    model = CausalLM(model_config).to(device)
    optimizer = build_optimizer(model, lr=train_config.lr, weight_decay=train_config.weight_decay)

    checkpoint_path = Path(train_config.out_dir) / CHECKPOINT_NAME
    start_step = 0
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"không có checkpoint để resume: {checkpoint_path}")
        start_step = load_checkpoint(checkpoint_path, model, optimizer, scaler, map_location=device)

    batches = endless(train_loader)
    # Resume: bỏ qua đúng số batch đã dùng, để chạy tiếp trùng với một lần chạy liền mạch.
    for _ in range(start_step * train_config.accum_steps):
        next(batches)

    throughput = Throughput()
    tokens_per_micro_batch = train_config.batch_size * train_config.seq_len
    model.train()

    train_loss, eval_loss, step = float("nan"), float("nan"), start_step
    for step in range(start_step, train_config.max_steps):
        lr = get_lr(
            step,
            max_lr=train_config.lr,
            min_lr=train_config.min_lr,
            warmup_steps=train_config.warmup_steps,
            max_steps=train_config.max_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(train_config.accum_steps):
            x, y = next(batches)
            x, y = x.to(device), y.to(device)
            with autocast():
                loss = next_token_loss(model(x), y) / train_config.accum_steps
            scaler.scale(loss).backward()
            train_loss += loss.item()
            throughput.update(tokens_per_micro_batch)

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip).item()
        scaler.step(optimizer)
        scaler.update()

        if (step + 1) % train_config.log_interval == 0:
            print(
                format_log(
                    step + 1,
                    train_config.max_steps,
                    train_loss,
                    lr,
                    grad_norm,
                    throughput.tokens_per_second(),
                    max_vram_mb(),
                )
            )
            throughput.reset()

        if (step + 1) % train_config.eval_interval == 0:
            eval_loss = evaluate(model, eval_loader, train_config.eval_batches, device, autocast)
            print(f"step {step + 1:>6} | val loss {eval_loss:6.3f}")

        if (step + 1) % train_config.save_interval == 0:
            save_checkpoint(
                checkpoint_path, model, optimizer, step + 1, model_config, train_config, scaler
            )

    save_checkpoint(checkpoint_path, model, optimizer, step + 1, model_config, train_config, scaler)
    return {"step": step + 1, "train_loss": train_loss, "eval_loss": eval_loss}


def main() -> None:
    """Chạy từ dòng lệnh: python -m trainer.train_pretrain --model-config ... --train-config ..."""
    parser = argparse.ArgumentParser(description="Pretrain một SLM tiếng Việt")
    parser.add_argument("--model-config", default="configs/model_tiny.json")
    parser.add_argument("--train-config", default="configs/pretrain_tiny.json")
    parser.add_argument("--resume", action="store_true", help="chạy tiếp từ checkpoint")
    args = parser.parse_args()

    model_config = ModelConfig.from_dict(
        json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    )
    train_config = TrainConfig.from_json(args.train_config)

    stats = train(model_config, train_config, resume=args.resume)
    print(f"Xong: {stats}")


if __name__ == "__main__":
    main()
```

</details>

## 5. Test — `tests/trainer/test_train_pretrain.py`

| Test                                        | Chứng minh                                                        |
| ------------------------------------------- | ------------------------------------------------------------------ |
| `test_train_runs_and_saves_checkpoint`    | Chạy 3 bước, có file checkpoint, loss hữu hạn                |
| `test_initial_loss_is_close_to_log_vocab` | Bước đầu loss ≈ ln V                                          |
| `test_resume_continues_step_count`        | Resume tiếp đúng số bước                                     |
| `test_resume_without_checkpoint_raises`   | Resume khi chưa có file → lỗi rõ ràng                        |
| `test_evaluation_runs`                    | Val loss tính được                                             |
| `test_loss_decreases_on_repeating_data`   | **Dữ liệu lặp → loss giảm quá nửa**: model học thật |
| `test_batches_differ_between_steps`       | Mỗi step lấy dữ liệu mới                                      |

<details>
<summary><b>Code test</b></summary>

```python
import math

import numpy as np
import pytest
import torch

from core.config import ModelConfig
from trainer.pretrain_dataset import DTYPE
from trainer.train_pretrain import CHECKPOINT_NAME, train

PATTERN = [5, 9, 13, 17, 21, 25, 29, 33]


def test_train_runs_and_saves_checkpoint(tiny_model_config, make_train_config, tmp_path):
    config = make_train_config(max_steps=3)

    stats = train(tiny_model_config, config)

    assert stats["step"] == 3
    assert math.isfinite(stats["train_loss"])
    assert (tmp_path / "out" / CHECKPOINT_NAME).exists()


def test_initial_loss_is_close_to_log_vocab(tiny_model_config, make_train_config):
    """Bước đầu tiên: model chưa học gì nên loss ≈ ln V."""
    stats = train(
        tiny_model_config, make_train_config(max_steps=1, warmup_steps=1, lr=1e-8, min_lr=0.0)
    )

    assert abs(stats["train_loss"] - math.log(tiny_model_config.vocab_size)) < 0.3


def test_resume_continues_step_count(tiny_model_config, make_train_config, tmp_path):
    train(tiny_model_config, make_train_config(max_steps=2))

    stats = train(tiny_model_config, make_train_config(max_steps=5), resume=True)

    assert stats["step"] == 5


def test_resume_without_checkpoint_raises(tiny_model_config, make_train_config):
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        train(tiny_model_config, make_train_config(), resume=True)


def test_evaluation_runs(tiny_model_config, make_train_config):
    stats = train(tiny_model_config, make_train_config(max_steps=2, eval_interval=1))

    assert math.isfinite(stats["eval_loss"])


def test_loss_decreases_on_repeating_data(tmp_path, make_train_config):
    """Dữ liệu lặp lại một mẫu ngắn: model phải học thuộc, loss giảm mạnh."""
    tokens = np.array(PATTERN * 500, dtype=DTYPE)
    bin_path = tmp_path / "repeat.bin"
    tokens.tofile(bin_path)

    model_config = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        ffn_hidden=64,
        max_seq_len=32,
        rope_theta=10000.0,
        norm_eps=1e-6,
        tie_embedding=True,
    )
    common = dict(
        train_bin=str(bin_path),
        eval_bin=str(bin_path),
        seq_len=16,
        batch_size=4,
        accum_steps=1,
        lr=5e-3,
        min_lr=1e-4,
    )
    before = train(model_config, make_train_config(max_steps=1, warmup_steps=1, **common))
    after = train(model_config, make_train_config(max_steps=60, warmup_steps=5, **common))

    assert after["train_loss"] < 0.5 * before["train_loss"]


def test_batches_differ_between_steps(tiny_model_config, make_train_config):
    """Mỗi step phải lấy dữ liệu mới, không lặp lại đúng một batch."""
    config = make_train_config(max_steps=2, accum_steps=1, batch_size=2)
    seen = []

    from trainer import train_pretrain

    original = train_pretrain.next_token_loss

    def spy(logits, targets):
        seen.append(targets.clone())
        return original(logits, targets)

    train_pretrain.next_token_loss = spy
    try:
        train(tiny_model_config, config)
    finally:
        train_pretrain.next_token_loss = original

    assert not torch.equal(seen[0], seen[1])
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/trainer -v` → **74 passed**
- [ ] Chạy thật trên dữ liệu nhỏ trước (sửa `max_steps` xuống 20) để xem log ra đúng dạng
- [ ] Chạy đủ: `python -m trainer.train_pretrain`
- [ ] **Tiêu chí xong chặng 2:** loss giảm, val loss giảm, và `test_resume_matches_uninterrupted_run` xanh

**Số Claude đo được khi chạy thử** trên 400KB dữ liệu thật của bạn, model `d_model=128, n_layers=2`, CPU:

```
step      2/6 | loss  9.704 | lr 6.00e-04 | gnorm  1.414 |     7,438 tok/s | vram      0 MB
step      3 | val loss  9.622
step      4/6 | loss  9.606 | lr 5.21e-04 | gnorm  1.442 |     6,540 tok/s | vram      0 MB
step      6/6 | loss  9.552 | lr 1.39e-04 | gnorm  1.366 |     9,385 tok/s | vram      0 MB
step      6 | val loss  9.539
```

Loss bắt đầu từ 9.70, gần `ln 16000 = 9.68` — đúng như mong đợi.

---

---

## Sau chặng 2

Xong D9 là xong pretrain. Việc tiếp theo:

1. **Cài `torch` bản CUDA** rồi train thật trên GPU (`device: "cuda"`, `dtype: "bf16"`).
2. **Chặng 3:** sinh chữ + KV cache — cho model viết tiếp một câu tiếng Việt.
3. Khi thuê GPU: tăng `d_model`, `n_layers`, `max_steps`, và thêm DDP.

---

---

# PHẦN E — SINH CHỮ (chặng 3)

Model đã train xong biết đoán token kế tiếp. Chặng này biến khả năng đó thành **viết tiếp văn bản**, và thêm **KV cache** để không phải tính lại từ đầu mỗi bước.

```
"Hà Nội là thủ đô"
   │
   ▼  tokenizer.encode
input_ids (1, T)
   │
   ▼  ┌─────────────────────────────────────────┐
      │  lặp max_new_tokens lần:                │
      │     logits = model(token mới, cache)    │  [E1] KV cache trong core/
      │     token  = chọn từ logits[:, -1]      │  [E2] inference/generate.py
      │     nối token vào chuỗi                 │
      └─────────────────────────────────────────┘
   │
   ▼  tokenizer.decode
"Hà Nội là thủ đô thị, trong đó có thể làm cho các bạn."
```

**Kết quả thật** — model 2,4 triệu tham số sau 2000 bước pretrain, Claude chạy thử:

| Cách chọn token             | Kết quả                                                                                                                                                                                                                         |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Greedy (`temperature=0`)    | *Hà Nội là thủ đô thị, trong đó có thể làm cho các bạn.*                                                                                                                                                          |
| `temperature=0.8, top_k=40` | *Hà Nội là thủ đô Bình Định, Quảng Ninh, Bắc Long, TP Hưng Bình.*                                                                                                                                                  |
| `temperature=0.8, top_k=40` | *Người Việt Nam, đã từng nhận nhận ra những vấn đề mà bản thức điều này cũng như hiện tượng đó sẽ giúp công việc không thể hiện những vấn đề về việc để tìm những cuộc sống khác.* |

Ngữ pháp xuôi, cụm từ đúng tiếng Việt, nội dung chưa có nghĩa — đúng mức của model 2,4 triệu tham số học 8 triệu token.

---

---

# E1 — KV cache: sửa `core/attention.py`, `core/block.py`, `core/model.py`

## 1. Vị trí

```
input_ids → Embedding → [Block: RMSNorm → ▶Attention (k, v cũ được giữ lại)◀ → ...] × L → lm_head
```

## 2. Làm gì

Sinh chữ là lặp: có `T` token → đoán token `T+1` → có `T+1` token → đoán token `T+2`…

**Không cache:** mỗi bước đưa cả chuỗi vào model, tính lại `k`, `v` cho mọi token cũ. Sinh `n` token tốn **O(n · T²)**.

**Có cache:** `k`, `v` của token cũ **không bao giờ đổi** (mỗi token chỉ nhìn về quá khứ). Nên chỉ cần lưu lại, mỗi bước tính `k`, `v` cho **đúng 1 token mới** rồi nối vào. Còn **O(n · T)**.

```
Bước 1 (prefill):  [Hà] [Nội] [là] [thủ] [đô]   →  cache k,v của 5 token
Bước 2:            chỉ đưa [thị] vào            →  cache thành 6 token
Bước 3:            chỉ đưa [,] vào              →  cache thành 7 token
```

**Hai chỗ phải sửa theo, rất dễ quên:**

1. **RoPE lấy sai vị trí.** Token mới ở vị trí `past_len`, không phải vị trí 0. Phải cắt `rope_cos[past_len : past_len + seq_len]`.
2. **Mask lệch.** Ma trận điểm giờ là `(seq_len, past_len + seq_len)`. Ô cấm là `j > i + past_len`, tức `torch.triu(..., diagonal=1 + past_len)`.

**Giữ API cũ:** `use_cache=False` (mặc định) thì `forward` vẫn trả **đúng một tensor** như trước, nên toàn bộ code và test của chặng 1–2 không phải sửa. Chỉ khi `use_cache=True` mới trả tuple `(kết quả, cache)`.

## 3. Vào → Ra

| Hàm                            | Không cache               | Có cache                                                         |
| ------------------------------- | -------------------------- | ----------------------------------------------------------------- |
| `CausalSelfAttention.forward` | `(B,T,d)` → `(B,T,d)` | thêm`past_kv`, trả `(out, (k, v))`                          |
| `Block.forward`               | `(B,T,d)` → `(B,T,d)` | thêm`past_kv`, trả `(x, (k, v))`                            |
| `Model.forward`               | `(B,T)` → `(B,T,d)`   | thêm`past_kvs` (list L phần tử), trả `(hidden, presents)` |
| `CausalLM.forward`            | `(B,T)` → `(B,T,V)`   | trả`(logits, presents)`                                        |

`k`, `v` trong cache có shape `(B, h, số_token_đã_có, hd)`.

## 4. Khung code

**`core/attention.py`** — 4 chỗ sửa:

```python
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,   # TODO 1: thêm 2 tham số
        use_cache: bool = False,
    ):
        ...
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # TODO 2: nếu có past_kv: past_len = past_k.shape[-2];
        #         nối past_k vào trước k và past_v vào trước v theo chiều -2
        #         ngược lại past_len = 0
        # TODO 3: present = (k, v) nếu use_cache, ngược lại None

        scores = ...
        future = torch.triu(
            torch.ones(seq_len, past_len + seq_len, ...),   # TODO 4a: chiều thứ 2 đổi
            diagonal=1 + past_len,                          # TODO 4b: lệch theo past_len
        )
        ...
        # TODO 5: return (out, present) nếu use_cache, ngược lại chỉ out
```

**`core/block.py`** — chuyển tiếp cache xuống attention:

```python
    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        # TODO 6: gọi attention với past_kv, use_cache
        # TODO 7: nếu use_cache thì tách (attn_out, present)
        # TODO 8: phần residual + ffn giữ nguyên
        # TODO 9: return (x, present) nếu use_cache, ngược lại x
```

**`core/model.py`** — quản lý danh sách cache của L tầng:

```python
    def forward(self, input_ids, past_kvs=None, use_cache=False):
        # TODO 10: past_len = past_kvs[0][0].shape[-2] nếu có past_kvs, ngược lại 0
        # TODO 11: kiểm tra past_len + seq_len > max_seq_len → ValueError
        # TODO 12: cos/sin cắt từ past_len đến past_len + seq_len
        # TODO 13: lặp zip(self.layers, past_kvs or [None] * L), gom presents
        # TODO 14: return (hidden, presents) nếu use_cache, ngược lại hidden
```

<details>
<summary><b>Lời giải — attention.py</b></summary>

Chỉ chép phần đổi; hai dòng `scores = ...` dưới đây viết gọn một dòng, file của bạn đang tách nhiều dòng thì cứ giữ nguyên.

```python
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        # ... phần chiếu q, k, v và apply_rope giữ nguyên ...

        # KV cache: nối k, v cũ vào trước k, v mới
        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.shape[-2]
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        present = (k, v) if use_cache else None

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        future = torch.triu(
            torch.ones(
                seq_len,
                past_len + seq_len,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1 + past_len,
        )
        scores = scores.masked_fill(future, float("-inf"))

        # ... softmax, nhân v, gộp head giữ nguyên ...

        out = self.o_proj(out)
        return (out, present) if use_cache else out
```

</details>

<details>
<summary><b>Lời giải — block.py</b></summary>

```python
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
```

</details>

<details>
<summary><b>Lời giải — model.py</b></summary>

```python
class Model(nn.Module):
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

        # RoPE phải lấy đúng đoạn vị trí của token mới
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
```

</details>

## 5. Test — `tests/core/test_kv_cache.py`

| Test                                                  | Chứng minh                                                    |
| ----------------------------------------------------- | -------------------------------------------------------------- |
| `test_forward_without_cache_returns_tensor`         | API cũ không đổi                                           |
| `test_cache_shapes`                                 | L phần tử, mỗi cái`(B, h, T, hd)`                        |
| `test_step_by_step_with_cache_matches_full_forward` | **Tiêu chí xong chặng 3**                             |
| `test_prefill_then_one_token_matches_full_forward`  | Cách dùng thật: nạp prompt một lần rồi sinh từng token |
| `test_cache_grows_by_one_each_step`                 | Cache dài thêm đúng 1 sau mỗi bước                      |
| `test_model_body_also_supports_cache`               | Thân model (Critic của PPO dùng sau này) cũng có cache   |

<details>
<summary><b>Code test</b></summary>

```python
import torch

from core.model import CausalLM, Model

SEQ_LEN = 6


def test_forward_without_cache_returns_tensor(tiny_config):
    """API cũ không đổi: không bật cache thì vẫn trả về đúng một tensor."""
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, SEQ_LEN))

    assert isinstance(lm(ids), torch.Tensor)


def test_cache_shapes(tiny_config):
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, SEQ_LEN))

    _, past_kvs = lm(ids, use_cache=True)

    assert len(past_kvs) == tiny_config.n_layers
    for key, value in past_kvs:
        assert key.shape == (2, tiny_config.n_heads, SEQ_LEN, tiny_config.head_dim)
        assert value.shape == key.shape


def test_step_by_step_with_cache_matches_full_forward(tiny_config):
    """Tiêu chí xong chặng 3: đưa từng token một kèm cache = đưa cả chuỗi một lần."""
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, SEQ_LEN))

    full_logits = lm(ids)

    step_logits = []
    past_kvs = None
    for position in range(SEQ_LEN):
        logits, past_kvs = lm(ids[:, position : position + 1], past_kvs=past_kvs, use_cache=True)
        step_logits.append(logits)

    torch.testing.assert_close(torch.cat(step_logits, dim=1), full_logits)


def test_prefill_then_one_token_matches_full_forward(tiny_config):
    """Cách dùng thật: nạp cả prompt một lần (prefill), rồi sinh từng token."""
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, SEQ_LEN))

    full_logits = lm(ids)

    _, past_kvs = lm(ids[:, :-1], use_cache=True)
    last_logits, _ = lm(ids[:, -1:], past_kvs=past_kvs, use_cache=True)

    torch.testing.assert_close(last_logits[:, -1], full_logits[:, -1])


def test_cache_grows_by_one_each_step(tiny_config):
    lm = CausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (1, 3))

    _, past_kvs = lm(ids, use_cache=True)
    assert past_kvs[0][0].shape[-2] == 3

    _, past_kvs = lm(ids[:, :1], past_kvs=past_kvs, use_cache=True)
    assert past_kvs[0][0].shape[-2] == 4


def test_model_body_also_supports_cache(tiny_config):
    """Thân model (dùng cho Critic ở PPO) cũng phải trả cache."""
    model = Model(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (1, 3))

    hidden, past_kvs = model(ids, use_cache=True)

    assert hidden.shape == (1, 3, tiny_config.d_model)
    assert len(past_kvs) == tiny_config.n_layers
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/core/test_kv_cache.py -v` → **6 passed**
- [ ] `python -m pytest tests -v` → **toàn bộ test cũ vẫn xanh** (API không đổi)
- [ ] Thử phá: đổi `diagonal=1 + past_len` thành `diagonal=1` → test phải đỏ

---

---

# E2 — `inference/generate.py`

## 1. Vị trí

```
tokenizer.encode → ▶vòng lặp sinh token◀ → tokenizer.decode
```

## 2. Làm gì

Hai hàm:

| Hàm                                                | Việc                                                    |
| --------------------------------------------------- | -------------------------------------------------------- |
| `sample_next_token(logits, temperature, top_k)`   | Từ logits của**một** vị trí, chọn ra 1 token |
| `generate(model, input_ids, max_new_tokens, ...)` | Lặp: gọi model → chọn token → nối vào chuỗi      |

**Ba cách chọn token:**

| Cách                      | Tham số            | Tính chất                                                                                            |
| -------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------ |
| Tham lam                   | `temperature=0`   | Luôn lấy token điểm cao nhất; chạy lại ra y hệt; dễ lặp từ                                  |
| Lấy mẫu theo nhiệt độ | `temperature=0.8` | Chia logits cho nhiệt độ rồi bốc theo xác suất. Nhiệt độ thấp = an toàn, cao = phiêu lưu |
| Top-k                      | `top_k=40`        | Chỉ giữ 40 token điểm cao nhất, phần còn lại xác suất 0. Chặn những token vô lý          |

Thường dùng chung `temperature=0.8, top_k=40`.

**Dừng sớm:** gặp `<|eos|>` thì coi như hết văn bản. Chuỗi nào trong batch đã xong thì các bước sau chỉ thêm `<|eos|>`, không sinh chữ mới.

## 3. Vào → Ra

|      |                                                     |
| ---- | --------------------------------------------------- |
| Vào | `model`, `input_ids (B, T)`, `max_new_tokens` |
| Ra   | `(B, T + n)`, `n ≤ max_new_tokens`             |

`sample_next_token`: `(B, V)` → `(B, 1)` kiểu long.

## 4. Khung code

```python
"""Sinh chữ: lặp lấy logits ở vị trí cuối, chọn token, nối vào chuỗi."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import ModelConfig
from core.model import CausalLM
from trainer.checkpoint import load_weights

GREEDY_TEMPERATURE = 0.0


def sample_next_token(logits, temperature=1.0, top_k=None, generator=None) -> torch.Tensor:
    """Chọn token kế tiếp từ logits của vị trí cuối. (B, V) -> (B, 1) long."""
    # TODO 1: temperature < 0 → ValueError
    # TODO 2: temperature == 0 → logits.argmax(dim=-1, keepdim=True)
    # TODO 3: scores = logits.float() / temperature
    # TODO 4: nếu có top_k: top_k < 1 → ValueError;
    #         k = min(top_k, V); lấy giá trị thứ k bằng scores.topk(k, dim=-1).values[:, -1:]
    #         scores.masked_fill(scores < giá_trị_thứ_k, -inf)
    # TODO 5: probs = softmax(scores, -1); return torch.multinomial(probs, 1, generator=generator)


@torch.no_grad()
def generate(model, input_ids, max_new_tokens, temperature=1.0, top_k=None,
             eos_id=None, use_cache=True, generator=None) -> torch.Tensor:
    """Sinh tiếp max_new_tokens token sau input_ids."""
    # TODO 6: max_new_tokens < 0 → ValueError; model.eval()
    # TODO 7: output = input_ids; step_input = input_ids; past_kvs = None
    #         finished = tensor bool (B,) toàn False
    # TODO 8: lặp max_new_tokens lần:
    #             use_cache → logits, past_kvs = model(step_input, past_kvs=..., use_cache=True)
    #             ngược lại → logits = model(output)          (tính lại cả chuỗi)
    #             next_token = sample_next_token(logits[:, -1], ...)
    #             nếu có eos_id: chuỗi đã finished thì ép next_token = eos_id;
    #                            cập nhật finished
    #             output = cat([output, next_token], dim=1); step_input = next_token
    #             mọi chuỗi finished → break
    # TODO 9: return output


def load_model(checkpoint_path, device="cpu") -> CausalLM:
    """Dựng lại model từ checkpoint (kiến trúc lấy từ chính checkpoint)."""
    # TODO 10: torch.load lấy model_config → CausalLM(...) → load_weights


def main() -> None:
    """python -m inference.generate --prompt "Hà Nội là" """
    # TODO 11: argparse: --checkpoint, --tokenizer, --prompt, --max-new-tokens,
    #          --temperature, --top-k, --seed, --no-cache
    # TODO 12: nạp tokenizer + model, encode prompt, generate, decode và in ra
```

<details>
<summary><b>Lời giải</b></summary>

```python
def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Chọn token kế tiếp từ logits của vị trí cuối.

    Shape:
        logits: (B, V)  ->  (B, 1) long
    Công thức:
        temperature = 0 : argmax (tham lam)
        temperature > 0 : lấy mẫu từ softmax(logits / temperature)
        top_k = k       : chỉ giữ k token điểm cao nhất
    Ổn định số học:
        Tính bằng float32; chia temperature trước khi softmax.
    """
    if temperature < 0:
        raise ValueError(f"temperature={temperature} phải >= 0")
    if temperature == GREEDY_TEMPERATURE:
        return logits.argmax(dim=-1, keepdim=True)

    scores = logits.float() / temperature
    if top_k is not None:
        if top_k < 1:
            raise ValueError(f"top_k={top_k} phải >= 1")
        k = min(top_k, scores.shape[-1])
        kth_value = scores.topk(k, dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < kth_value, float("-inf"))

    probs = F.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sinh tiếp `max_new_tokens` token sau `input_ids`.

    Shape:
        input_ids: (B, T)  ->  (B, T + n) với n <= max_new_tokens
    Độ phức tạp:
        use_cache=True : O(n · T) — mỗi bước chỉ tính cho 1 token mới.
        use_cache=False: O(n · T²) — mỗi bước tính lại cả chuỗi.
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens={max_new_tokens} phải >= 0")

    model.eval()
    output = input_ids
    step_input = input_ids
    past_kvs = None
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    for _ in range(max_new_tokens):
        if use_cache:
            logits, past_kvs = model(step_input, past_kvs=past_kvs, use_cache=True)
        else:
            logits = model(output)

        next_token = sample_next_token(logits[:, -1], temperature, top_k, generator)
        if eos_id is not None:
            next_token[finished] = eos_id
            finished |= next_token.squeeze(-1) == eos_id

        output = torch.cat([output, next_token], dim=1)
        step_input = next_token

        if eos_id is not None and bool(finished.all()):
            break

    return output


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> CausalLM:
    """Dựng lại model từ checkpoint (kiến trúc lấy từ chính checkpoint)."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CausalLM(ModelConfig.from_dict(payload["model_config"])).to(device)
    load_weights(checkpoint_path, model, map_location=device)
    return model


def main() -> None:
    """python -m inference.generate --prompt "Hà Nội là" """
    parser = argparse.ArgumentParser(description="Cho model viết tiếp một câu")
    parser.add_argument("--checkpoint", default="checkpoints/pretrain.pt")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--prompt", default="Hà Nội là")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true", help="tắt KV cache (chậm hơn)")
    args = parser.parse_args()

    from tokenizers import Tokenizer

    torch.manual_seed(args.seed)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = load_model(args.checkpoint)

    input_ids = torch.tensor([tokenizer.encode(args.prompt).ids], dtype=torch.long)
    output = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.token_to_id("<|eos|>"),
        use_cache=not args.no_cache,
    )

    print(f"Prompt : {args.prompt}")
    print(f"Sinh ra: {tokenizer.decode(output[0].tolist())}")


if __name__ == "__main__":
    main()
```

</details>

## 5. Test — `tests/inference/test_generate.py`

| Test                                              | Chứng minh                                      |
| ------------------------------------------------- | ------------------------------------------------ |
| `test_zero_temperature_is_argmax`               | Tham lam = argmax                                |
| `test_top_k_one_equals_argmax`                  | `top_k=1` trùng tham lam                      |
| `test_top_k_never_picks_outside_top_k`          | Bốc 50 lần không lọt ra ngoài top-k         |
| `test_same_generator_seed_gives_same_token`     | Cùng seed cho cùng kết quả                   |
| `test_invalid_arguments_raise`                  | `temperature < 0`, `top_k = 0` bị từ chối |
| `test_output_grows_by_max_new_tokens`           | Shape ra đúng, prompt được giữ nguyên     |
| `test_cache_and_no_cache_give_same_tokens`      | **Bật/tắt cache ra cùng chữ**          |
| `test_greedy_is_deterministic`                  | Chạy 2 lần ra y hệt                           |
| `test_zero_new_tokens_returns_prompt`           | `max_new_tokens=0` trả lại prompt            |
| `test_stops_early_at_eos`                       | Gặp eos là dừng                               |
| `test_finished_sequences_keep_padding_with_eos` | Chuỗi xong rồi chỉ thêm eos                  |
| `test_cache_makes_model_see_one_token_per_step` | Có cache: mỗi bước chỉ đưa 1 token        |
| `test_generation_beyond_max_seq_len_raises`     | Vượt`max_seq_len` bị chặn                  |

<details>
<summary><b>Code test</b></summary>

```python
import pytest
import torch

from core.model import CausalLM
from inference.generate import generate, sample_next_token

PROMPT_LEN = 4
NEW_TOKENS = 6


class FixedLogitsModel(torch.nn.Module):
    """Model giả luôn trả cùng một bộ logits — để kiểm tra phần điều khiển vòng lặp."""

    def __init__(self, vocab_size: int, favourite: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.favourite = favourite
        self.calls = 0

    def forward(self, input_ids, past_kvs=None, use_cache=False):
        self.calls += 1
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size)
        logits[..., self.favourite] = 10.0
        if use_cache:
            return logits, [(torch.zeros(batch, 1, seq_len, 1), torch.zeros(batch, 1, seq_len, 1))]
        return logits


# ---------------------------------------------------------------- sample_next_token


def test_zero_temperature_is_argmax():
    logits = torch.tensor([[1.0, 5.0, 2.0], [9.0, 0.0, 0.0]])

    assert sample_next_token(logits, temperature=0.0).tolist() == [[1], [0]]


def test_top_k_one_equals_argmax():
    logits = torch.randn(4, 50)

    greedy = sample_next_token(logits, temperature=0.0)
    top1 = sample_next_token(logits, temperature=1.0, top_k=1)

    assert torch.equal(greedy, top1)


def test_top_k_never_picks_outside_top_k():
    logits = torch.randn(1, 100)
    allowed = set(logits.topk(5, dim=-1).indices[0].tolist())

    picks = {sample_next_token(logits, temperature=1.0, top_k=5).item() for _ in range(50)}

    assert picks <= allowed


def test_same_generator_seed_gives_same_token():
    logits = torch.randn(2, 20)

    first = sample_next_token(logits, generator=torch.Generator().manual_seed(0))
    second = sample_next_token(logits, generator=torch.Generator().manual_seed(0))

    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("kwargs", [{"temperature": -1.0}, {"top_k": 0}])
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        sample_next_token(torch.randn(1, 10), **kwargs)


# ---------------------------------------------------------------- generate


def test_output_grows_by_max_new_tokens(tiny_config):
    model = CausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, PROMPT_LEN))

    output = generate(model, input_ids, max_new_tokens=NEW_TOKENS, temperature=0.0)

    assert output.shape == (2, PROMPT_LEN + NEW_TOKENS)
    torch.testing.assert_close(output[:, :PROMPT_LEN], input_ids)


def test_cache_and_no_cache_give_same_tokens(tiny_config):
    """Tiêu chí xong chặng 3: sinh từng bước có cache = forward cả chuỗi."""
    model = CausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, PROMPT_LEN))

    with_cache = generate(model, input_ids, NEW_TOKENS, temperature=0.0, use_cache=True)
    without_cache = generate(model, input_ids, NEW_TOKENS, temperature=0.0, use_cache=False)

    torch.testing.assert_close(with_cache, without_cache)


def test_greedy_is_deterministic(tiny_config):
    model = CausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, PROMPT_LEN))

    first = generate(model, input_ids, NEW_TOKENS, temperature=0.0)
    second = generate(model, input_ids, NEW_TOKENS, temperature=0.0)

    torch.testing.assert_close(first, second)


def test_zero_new_tokens_returns_prompt(tiny_config):
    model = CausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, PROMPT_LEN))

    torch.testing.assert_close(generate(model, input_ids, 0), input_ids)


def test_stops_early_at_eos():
    eos_id = 3
    model = FixedLogitsModel(vocab_size=10, favourite=eos_id)
    input_ids = torch.zeros(1, 2, dtype=torch.long)

    output = generate(model, input_ids, max_new_tokens=20, temperature=0.0, eos_id=eos_id)

    assert output.shape == (1, 3)  # prompt 2 token + đúng 1 token eos
    assert output[0, -1].item() == eos_id


def test_finished_sequences_keep_padding_with_eos():
    """Chuỗi nào ra eos trước thì các bước sau chỉ thêm eos, không sinh chữ khác."""
    eos_id = 3
    model = FixedLogitsModel(vocab_size=10, favourite=eos_id)
    input_ids = torch.zeros(2, 2, dtype=torch.long)

    output = generate(model, input_ids, max_new_tokens=5, temperature=0.0, eos_id=eos_id)

    assert (output[:, 2:] == eos_id).all()


def test_cache_makes_model_see_one_token_per_step(tiny_config):
    """Có cache: mỗi bước chỉ đưa 1 token mới vào model."""
    model = FixedLogitsModel(vocab_size=tiny_config.vocab_size, favourite=1)
    input_ids = torch.zeros(1, PROMPT_LEN, dtype=torch.long)

    generate(model, input_ids, max_new_tokens=NEW_TOKENS, temperature=0.0, use_cache=True)

    assert model.calls == NEW_TOKENS


def test_generation_beyond_max_seq_len_raises(tiny_config):
    """Prompt dài đúng max_seq_len: sinh 1 token vẫn được, token thứ 2 thì vượt giới hạn."""
    model = CausalLM(tiny_config)
    input_ids = torch.zeros(1, tiny_config.max_seq_len, dtype=torch.long)

    generate(model, input_ids, max_new_tokens=1, temperature=0.0)

    with pytest.raises(ValueError, match="max_seq_len"):
        generate(model, input_ids, max_new_tokens=2, temperature=0.0)
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/inference -v` → **14 passed**
- [ ] Chạy thật:
  ```powershell
  python -m inference.generate --prompt "Hà Nội là thủ đô" --temperature 0 --max-new-tokens 40
  python -m inference.generate --prompt "Người Việt Nam" --temperature 0.8 --top-k 40
  ```
- [ ] So sánh `--temperature 0` (lặp từ, an toàn) với `--temperature 1.2` (phiêu, dễ vô nghĩa)

---

---

# E3 — `scripts/windows/generate.bat`

File này chỉ bọc lệnh trên cho tiện. Nhận prompt làm tham số.

```bat
@echo off
rem ================================================================
rem  Cho model viet tiep mot cau.
rem
rem  Cach dung:
rem    scripts\windows\generate.bat
rem    scripts\windows\generate.bat "Nguoi Viet Nam"
rem
rem  Doi tham so sinh chu: sua 3 dong SET ben duoi.
rem  Ma thoat: 0 = xong, 1 = thieu venv, 3 = thieu tokenizer,
rem            6 = chua co checkpoint
rem ================================================================
setlocal

chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"

pushd "%~dp0..\.."

set "PYTHON=.venv\Scripts\python.exe"
set "CHECKPOINT=checkpoints\pretrain.pt"
set "TOKENIZER=tokenizer\tokenizer.json"
set "MAX_NEW_TOKENS=60"
set "TEMPERATURE=0.8"
set "TOP_K=40"
set "EXIT_CODE=0"

if not exist "%PYTHON%" (
    echo [LOI] Khong tim thay %PYTHON%
    set "EXIT_CODE=1"
    goto :finish
)

if not exist "%TOKENIZER%" (
    echo [LOI] Khong tim thay %TOKENIZER%
    set "EXIT_CODE=3"
    goto :finish
)

if not exist "%CHECKPOINT%" (
    echo [LOI] Chua co %CHECKPOINT%
    echo       Train truoc: scripts\windows\train_pretrain.bat
    set "EXIT_CODE=6"
    goto :finish
)

set "PROMPT=%~1"
if "%PROMPT%"=="" set "PROMPT=Ha Noi la thu do"

"%PYTHON%" -m inference.generate --checkpoint "%CHECKPOINT%" --tokenizer "%TOKENIZER%" --prompt "%PROMPT%" --max-new-tokens %MAX_NEW_TOKENS% --temperature %TEMPERATURE% --top-k %TOP_K%
set "EXIT_CODE=%ERRORLEVEL%"

:finish
popd
endlocal & exit /b %EXIT_CODE%
```

**Lưu ý:** file `.bat` lưu dạng ASCII nên prompt gõ trong dòng lệnh có dấu tiếng Việt có thể bị lỗi mã. Muốn prompt có dấu, chạy thẳng bằng Python:

```powershell
python -m inference.generate --prompt "Hà Nội là thủ đô"
```

---

---

## Sau chặng 3

Model giờ **viết tiếp văn bản được**, nhưng vẫn chưa trò chuyện được — hỏi thì nó viết tiếp câu hỏi chứ không trả lời. Đó là việc của **chặng 4 (SFT)**:

1. **Thế giới đồ chơi:** bộ hỏi–đáp tự kiểm tra được, dùng chung cho SFT, DPO, tool use, RL.
2. **Chat template:** ghép tin nhắn bằng `<|im_start|>`, `<|im_end|>` — tokenizer đã chừa sẵn.
3. **`SFTDataset`:** nhãn chỉ phủ phần assistant, phần prompt đặt `-100` (đã có sẵn `IGNORE_INDEX` trong `trainer/losses.py`).
4. **`train_sft.py`:** dùng lại nguyên D4–D7, chỉ đổi dataset và nạp trọng số bằng `load_weights` thay vì resume.

---

---

# PHẦN F — SFT (chặng 4)

Model pretrain chỉ biết **viết tiếp văn bản**. SFT (Supervised Fine-Tuning) dạy nó **trả lời theo định dạng hội thoại**: nhận câu hỏi của người dùng, sinh câu trả lời, rồi **tự dừng**.

```
dataset/sft/train.jsonl   ← [F1] thế giới đồ chơi + [F3] prepare_sft.py
   │  {"messages": [{"role": "user", "content": "Bảy cộng năm bằng bao nhiêu?"},
   │                {"role": "assistant", "content": "12"}]}
   ▼  [F2] chat_template.py
<|bos|><|im_start|>user\nBảy cộng năm bằng bao nhiêu?<|im_end|>\n<|im_start|>assistant\n12<|im_end|>\n
   ▼  nhãn: CHỈ phủ "12<|im_end|>", phần còn lại là -100
   ▼  [F4] sft_dataset.py → (x, y) đã dịch nhãn, pad cố định
   ▼  [F6] train_sft.py: nạp trọng số pretrain → train → checkpoints/sft.pt
   ▼  [F7] inference/chat.py: hỏi – đáp, đo độ chính xác
```

## Kết quả thật

Claude chạy trên checkpoint pretrain 2000 bước của bạn, `configs/sft_tiny.json`, CPU, **3 phút 34 giây**:

|                               | Trả lời đúng (tập train) | Trả lời đúng (tập eval — bài chưa từng thấy) |
| ----------------------------- | ----------------------------- | ------------------------------------------------------ |
| Model pretrain                | 0,0%                          | 0,0%                                                   |
| **Sau SFT 2000 bước** | **97,8%**               | **80,1%**                                        |

| Câu hỏi                                                                  | Model pretrain                                                        | Model SFT    |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------ |
| Bảy cộng năm bằng bao nhiêu?                                          | *(tải bài hát) cho rằng, ca sĩ đã được trình bày.*      | **12** |
| Tính mười lăm trừ sáu.                                               | *(B) cho rằng, nhưng không có thể làm cho các bạn có thể* | **9**  |
| Mười một cộng mười sáu bằng bao nhiêu?*(bài thuộc tập eval)* | *(B) là một trong những sự kiện khác nhau.- Hiện*            | **27** |
| Hà Nội là thủ đô của nước nào?*(ngoài thế giới đồ chơi)* | *, có thể làm cho các bạn có thể làm cho các bạn.*        | **8**  |

Dòng cuối là **giới hạn cần hiểu**: SFT chỉ dạy đúng những gì có trong dữ liệu. Model chỉ từng thấy câu hỏi phép tính, nên hỏi gì nó cũng trả lời bằng một con số. Muốn trò chuyện rộng hơn phải có dữ liệu hội thoại đa dạng hơn.

**Tiêu chí xong chặng 4** trong CLAUDE.md: *"nhãn chỉ phủ token assistant; học thuộc tập đồ chơi"* — cả hai có test riêng (F2 và F6).

---

---

# F1 — `toyworld/arithmetic.py`

## 1. Vị trí

```
▶thế giới đồ chơi◀ → jsonl → chat template → dataset → train → chat / đo độ chính xác
```

Thư mục mới `toyworld/` (thêm `toyworld/__init__.py` rỗng). Dùng lại ở DPO, tool use, RL.

## 2. Làm gì

Sinh bài toán **cộng / trừ với số 0–20, hỏi bằng chữ tiếng Việt**, và **chấm đúng/sai bằng code**.

Vì sao chọn phép tính:

- **Đáp án kiểm tra được tự động** — không cần người chấm, không cần model khác chấm.
- **Đủ nhỏ** để model 2,4 triệu tham số học được trên CPU.
- **Dùng lại được** cho mọi chặng sau: DPO (cặp đúng/sai), tool use (gọi máy tính), RL (reward = đúng/sai).

|                                                | Số lượng         |
| ---------------------------------------------- | ------------------- |
| Phép cộng                                    | 21 × 21 = 441 bài |
| Phép trừ (chỉ`a ≥ b`, không ra số âm) | 231 bài            |
| Cách hỏi                                     | 3 mẫu câu         |
| Tổng mẫu                                     | 672 × 3 = 2016     |

**Chia train/eval theo BÀI TOÁN**, không theo câu hỏi: bài nào ở tập eval thì không xuất hiện ở train dưới bất kỳ cách hỏi nào. Nhờ vậy độ chính xác trên eval đo được model có **tổng quát hoá** hay chỉ học vẹt.

## 3. Vào → Ra

| Hàm                                     | Ví dụ                                                                                                  |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `number_to_words(15)`                  | `"mười lăm"`                                                                                        |
| `question_text(Problem(7, 5, "+"), 0)` | `"Bảy cộng năm bằng bao nhiêu?"`                                                                  |
| `to_messages(Problem(15, 6, "-"), 1)`  | `[{"role": "user", "content": "Tính mười lăm trừ sáu."}, {"role": "assistant", "content": "9"}]` |
| `parse_answer("Đáp án là 27.")`    | `27`                                                                                                   |
| `is_correct("12", Problem(7, 5, "+"))` | `True`                                                                                                 |

## 4. Khung code

```python
MAX_OPERAND = 20
NUMBER_WORDS = ("không", "một", ..., "mười lăm", ..., "hai mươi")   # 21 phần tử
OPERATION_WORDS = {"+": "cộng", "-": "trừ"}
QUESTION_TEMPLATES = ("{a} {op} {b} bằng bao nhiêu?", "Tính {a} {op} {b}.", "{a} {op} {b} bằng mấy?")


@dataclass(frozen=True)
class Problem:
    a: int
    b: int
    op: str
    # TODO 1: property answer: a + b hoặc a - b


def number_to_words(number: int) -> str:
    # TODO 2: ngoài [0, 20] → ValueError; ngược lại tra NUMBER_WORDS


def all_problems() -> list[Problem]:
    # TODO 3: 2 vòng lặp a, b trong 0..20: luôn thêm phép cộng; thêm phép trừ chỉ khi a >= b


def question_text(problem: Problem, template_index: int) -> str:
    # TODO 4: format mẫu câu bằng chữ, viết hoa chữ đầu


def to_messages(problem: Problem, template_index: int) -> list[dict[str, str]]:
    # TODO 5: [user: câu hỏi, assistant: str(đáp án)]


def parse_answer(text: str) -> int | None:
    # TODO 6: re.search(r"-?\d+") → int, không thấy → None


def is_correct(text: str, problem: Problem) -> bool:
    # TODO 7: parse_answer(text) == problem.answer


def split_problems(problems, eval_fraction: float, seed: int) -> tuple[list[Problem], list[Problem]]:
    # TODO 8: eval_fraction ngoài (0, 1) → ValueError
    # TODO 9: xáo trộn bằng random.Random(seed); n_eval = max(1, round(len * eval_fraction))
    #         trả (phần sau, phần đầu)
```

<details>
<summary><b>Lời giải</b></summary>

```python
"""Thế giới đồ chơi: phép cộng/trừ 0–20 hỏi bằng tiếng Việt, chấm điểm tự động.

Dùng xuyên suốt các chặng sau (SFT, DPO, tool use, RL): đáp án luôn kiểm tra được bằng code.
"""

import random
import re
from dataclasses import dataclass

MAX_OPERAND = 20
NUMBER_WORDS = (
    "không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
    "mười một", "mười hai", "mười ba", "mười bốn", "mười lăm",
    "mười sáu", "mười bảy", "mười tám", "mười chín", "hai mươi",
)
OPERATION_WORDS = {"+": "cộng", "-": "trừ"}
QUESTION_TEMPLATES = (
    "{a} {op} {b} bằng bao nhiêu?",
    "Tính {a} {op} {b}.",
    "{a} {op} {b} bằng mấy?",
)
INTEGER_PATTERN = re.compile(r"-?\d+")


@dataclass(frozen=True)
class Problem:
    """Một bài toán: a (op) b."""

    a: int
    b: int
    op: str

    @property
    def answer(self) -> int:
        return self.a + self.b if self.op == "+" else self.a - self.b


def number_to_words(number: int) -> str:
    """0..20 -> chữ tiếng Việt. Ví dụ 15 -> "mười lăm"."""
    if not 0 <= number <= MAX_OPERAND:
        raise ValueError(f"number={number} phải trong [0, {MAX_OPERAND}]")
    return NUMBER_WORDS[number]


def all_problems() -> list[Problem]:
    """Mọi bài cộng và trừ với toán hạng 0..20; phép trừ chỉ lấy a >= b (không ra số âm)."""
    problems = []
    for a in range(MAX_OPERAND + 1):
        for b in range(MAX_OPERAND + 1):
            problems.append(Problem(a, b, "+"))
            if a >= b:
                problems.append(Problem(a, b, "-"))
    return problems


def question_text(problem: Problem, template_index: int) -> str:
    """Câu hỏi tiếng Việt, viết hoa chữ đầu."""
    template = QUESTION_TEMPLATES[template_index]
    text = template.format(
        a=number_to_words(problem.a),
        op=OPERATION_WORDS[problem.op],
        b=number_to_words(problem.b),
    )
    return text[0].upper() + text[1:]


def to_messages(problem: Problem, template_index: int) -> list[dict[str, str]]:
    """Một lượt hỏi–đáp: người dùng hỏi bằng chữ, trợ lý trả lời bằng số."""
    return [
        {"role": "user", "content": question_text(problem, template_index)},
        {"role": "assistant", "content": str(problem.answer)},
    ]


def parse_answer(text: str) -> int | None:
    """Lấy số nguyên đầu tiên trong câu trả lời; không có thì None."""
    match = INTEGER_PATTERN.search(text)
    return int(match.group()) if match else None


def is_correct(text: str, problem: Problem) -> bool:
    return parse_answer(text) == problem.answer


def split_problems(
    problems: list[Problem], eval_fraction: float, seed: int
) -> tuple[list[Problem], list[Problem]]:
    """Chia theo BÀI TOÁN (không theo câu hỏi): bài ở tập eval không xuất hiện ở train dưới bất kỳ cách hỏi nào."""
    if not 0 < eval_fraction < 1:
        raise ValueError(f"eval_fraction={eval_fraction} phải trong (0, 1)")
    shuffled = list(problems)
    random.Random(seed).shuffle(shuffled)
    n_eval = max(1, round(len(shuffled) * eval_fraction))
    return shuffled[n_eval:], shuffled[:n_eval]
```

</details>

## 5. Test — `tests/toyworld/test_arithmetic.py`

| Test                                                  | Chứng minh                                                      |
| ----------------------------------------------------- | ---------------------------------------------------------------- |
| `test_number_to_words`                              | 0, 5, 10, 15 (*mười lăm*), 20 đúng chữ                   |
| `test_number_out_of_range_raises`                   | −1 và 21 bị từ chối                                         |
| `test_answer`                                       | Tính đúng                                                     |
| `test_all_problems_count_and_no_negative_answer`    | Đúng 672 bài, không trùng, không ra số âm                |
| `test_question_is_written_in_words`                 | Câu hỏi không chứa chữ số                                  |
| `test_to_messages_has_user_then_assistant`          | Đúng vai và nội dung                                         |
| `test_parse_answer` / `test_is_correct`           | Chấm điểm đúng                                              |
| `test_split_is_disjoint_complete_and_deterministic` | Train/eval không giao nhau, đủ, cùng seed ra cùng kết quả |
| `test_every_template_is_usable`                     | 3 mẫu câu khác nhau                                           |
| `test_invalid_eval_fraction_raises`                 | 0 và 1 bị từ chối                                            |

<details>
<summary><b>Code test</b></summary>

```python
import pytest

from toyworld.arithmetic import (
    MAX_OPERAND,
    QUESTION_TEMPLATES,
    Problem,
    all_problems,
    is_correct,
    number_to_words,
    parse_answer,
    question_text,
    split_problems,
    to_messages,
)


@pytest.mark.parametrize(
    "number, words", [(0, "không"), (5, "năm"), (10, "mười"), (15, "mười lăm"), (20, "hai mươi")]
)
def test_number_to_words(number, words):
    assert number_to_words(number) == words


@pytest.mark.parametrize("number", [-1, MAX_OPERAND + 1])
def test_number_out_of_range_raises(number):
    with pytest.raises(ValueError, match="number"):
        number_to_words(number)


def test_answer():
    assert Problem(7, 5, "+").answer == 12
    assert Problem(15, 6, "-").answer == 9


def test_all_problems_count_and_no_negative_answer():
    problems = all_problems()
    n_add = (MAX_OPERAND + 1) ** 2
    n_sub = (MAX_OPERAND + 1) * (MAX_OPERAND + 2) // 2  # cặp a >= b

    assert len(problems) == n_add + n_sub
    assert len(set(problems)) == len(problems)
    assert all(problem.answer >= 0 for problem in problems)


def test_question_is_written_in_words():
    text = question_text(Problem(7, 5, "+"), 0)

    assert text == "Bảy cộng năm bằng bao nhiêu?"
    assert not any(char.isdigit() for char in text)


def test_to_messages_has_user_then_assistant():
    messages = to_messages(Problem(15, 6, "-"), 1)

    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "Tính mười lăm trừ sáu."
    assert messages[1]["content"] == "9"


@pytest.mark.parametrize(
    "text, expected", [("12", 12), (" Đáp án là 27.", 27), ("không biết", None), ("-3", -3)]
)
def test_parse_answer(text, expected):
    assert parse_answer(text) == expected


def test_is_correct():
    problem = Problem(7, 5, "+")

    assert is_correct("12", problem)
    assert not is_correct("13", problem)
    assert not is_correct("không biết", problem)


def test_split_is_disjoint_complete_and_deterministic():
    problems = all_problems()

    train, eval_ = split_problems(problems, eval_fraction=0.1, seed=0)
    train_again, eval_again = split_problems(problems, eval_fraction=0.1, seed=0)

    assert set(train).isdisjoint(eval_)
    assert set(train) | set(eval_) == set(problems)
    assert (train, eval_) == (train_again, eval_again)


def test_every_template_is_usable():
    problem = Problem(3, 2, "-")

    texts = {question_text(problem, index) for index in range(len(QUESTION_TEMPLATES))}

    assert len(texts) == len(QUESTION_TEMPLATES)


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_invalid_eval_fraction_raises(fraction):
    with pytest.raises(ValueError, match="eval_fraction"):
        split_problems(all_problems(), fraction, seed=0)
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/toyworld -v` → **20 passed**

---

---

# F2 — `trainer/chat_template.py`

## 1. Vị trí

```
jsonl → ▶chat template: tin nhắn → token ids + nhãn◀ → dataset → train
                                  ▶prompt khi chat◀ → generate
```

## 2. Làm gì

**Định dạng** (giống ChatML mà Qwen, minimind dùng):

```
<|bos|>
<|im_start|>user
Bảy cộng năm bằng bao nhiêu?<|im_end|>
<|im_start|>assistant
12<|im_end|>
```

**Nhãn — ý quan trọng nhất của SFT:** chỉ học **nội dung assistant + `<|im_end|>`**. Mọi thứ khác nhãn `-100` (bị `next_token_loss` bỏ qua).

```
token:  <|bos|> <|im_start|> user \n Bảy cộng năm ... <|im_end|> \n <|im_start|> assistant \n  12  <|im_end|> \n
nhãn:    -100      -100      -100 ...     -100          -100    -100    -100       -100   -100  12  <|im_end|> -100
```

- Không học câu hỏi: model không cần biết cách **hỏi**, chỉ cần biết cách **trả lời**.
- **Học `<|im_end|>`**: đây là cách model biết lúc nào **dừng**. Quên token này, model trả lời xong sẽ viết tiếp mãi.

**Dựng nhãn theo TỪNG TIN NHẮN**, không dò chuỗi token như minimind (minimind tìm dãy token `<|im_start|>assistant` trong cả chuỗi — dễ sai khi tokenizer gộp token khác đi). Ở đây mỗi tin nhắn được encode riêng nên biết chắc token nào thuộc assistant.

**`build_prompt`** dùng **chung** `encode_conversation` rồi nối thêm `<|im_start|>assistant\n`. Nhờ vậy prompt lúc chat trùng từng token với lúc train — có test chứng minh.

## 3. Vào → Ra

| Hàm                                         | Vào                                      | Ra                                                                   |
| -------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------- |
| `encode_conversation(tokenizer, messages)` | list tin nhắn                            | `(input_ids, labels)` cùng độ dài, nhãn **chưa dịch** |
| `build_prompt(tokenizer, messages)`        | list tin nhắn (chưa có câu trả lời) | `input_ids` kết thúc bằng header assistant                      |

## 4. Khung code

```python
from tokenizers import Tokenizer

from trainer.losses import IGNORE_INDEX

BOS_TOKEN = "<|bos|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
ROLES = ("system", "user", "assistant")
ASSISTANT_ROLE = "assistant"


def _special_id(tokenizer, token: str) -> int:
    # TODO 1: token_to_id; None → ValueError "tokenizer thiếu token đặc biệt ..."


def encode_conversation(tokenizer, messages) -> tuple[list[int], list[int]]:
    # TODO 2: lấy bos_id, im_start_id, im_end_id; newline_ids = tokenizer.encode("\n").ids
    # TODO 3: input_ids = [bos_id]; labels = [IGNORE_INDEX]
    # TODO 4: với mỗi message:
    #             role không thuộc ROLES → ValueError
    #             header = [im_start_id] + tokenizer.encode(f"{role}\n").ids
    #             body   = tokenizer.encode(content).ids + [im_end_id]
    #             input_ids += header + body + newline_ids
    #             role là assistant → labels += [-100]*len(header) + body + [-100]*len(newline_ids)
    #             ngược lại         → labels += [-100] * (tổng độ dài 3 phần)
    # TODO 5: return input_ids, labels


def build_prompt(tokenizer, messages) -> list[int]:
    # TODO 6: encode_conversation(...)[0] + [im_start_id] + tokenizer.encode("assistant\n").ids
```

<details>
<summary><b>Lời giải</b></summary>

```python
"""Chat template: biến danh sách tin nhắn thành token ids + nhãn.

Định dạng (mỗi tin nhắn):
    <|im_start|>{role}\\n{content}<|im_end|>\\n
Cả hội thoại bắt đầu bằng <|bos|>.
"""

from tokenizers import Tokenizer

from trainer.losses import IGNORE_INDEX

BOS_TOKEN = "<|bos|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
ROLES = ("system", "user", "assistant")
ASSISTANT_ROLE = "assistant"


def _special_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer thiếu token đặc biệt {token}")
    return token_id


def encode_conversation(
    tokenizer: Tokenizer, messages: list[dict[str, str]]
) -> tuple[list[int], list[int]]:
    """Token ids + nhãn cho một hội thoại.

    Nhãn dựng THEO TỪNG TIN NHẮN (không dò chuỗi token như minimind):
        - tin nhắn assistant: nhãn = nội dung + <|im_end|>  (model học trả lời và học lúc dừng)
        - mọi phần khác (bos, header, user, system, dấu xuống dòng): nhãn = -100

    Returns:
        (input_ids, labels) cùng độ dài. Nhãn CHƯA dịch — dataset sẽ dịch 1 vị trí.
    """
    bos_id = _special_id(tokenizer, BOS_TOKEN)
    im_start_id = _special_id(tokenizer, IM_START_TOKEN)
    im_end_id = _special_id(tokenizer, IM_END_TOKEN)
    newline_ids = tokenizer.encode("\n").ids

    input_ids = [bos_id]
    labels = [IGNORE_INDEX]
    for message in messages:
        role = message["role"]
        if role not in ROLES:
            raise ValueError(f"role={role!r} phải thuộc {ROLES}")

        header = [im_start_id] + tokenizer.encode(f"{role}\n").ids
        body = tokenizer.encode(message["content"]).ids + [im_end_id]

        input_ids += header + body + newline_ids
        if role == ASSISTANT_ROLE:
            labels += [IGNORE_INDEX] * len(header) + body + [IGNORE_INDEX] * len(newline_ids)
        else:
            labels += [IGNORE_INDEX] * (len(header) + len(body) + len(newline_ids))

    return input_ids, labels


def build_prompt(tokenizer: Tokenizer, messages: list[dict[str, str]]) -> list[int]:
    """Token ids để model sinh câu trả lời tiếp theo: hội thoại + header của assistant.

    Dùng CHUNG hàm encode với lúc train để định dạng lúc train và lúc chạy luôn khớp nhau.
    """
    input_ids, _ = encode_conversation(tokenizer, messages)
    im_start_id = _special_id(tokenizer, IM_START_TOKEN)
    return input_ids + [im_start_id] + tokenizer.encode(f"{ASSISTANT_ROLE}\n").ids
```

</details>

## 5. Test — `tests/trainer/test_chat_template.py`

Cần thêm 2 fixture vào `tests/trainer/conftest.py` (xem mục **Fixture cho phần F** ở cuối F2).

| Test                                                     | Chứng minh                                                                         |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `test_ids_and_labels_have_same_length`                 | Hai list cùng độ dài                                                            |
| `test_full_text_follows_template`                      | Decode ra đúng từng ký tự của định dạng                                    |
| `test_only_assistant_answer_and_im_end_are_supervised` | **Tiêu chí xong chặng 4:** phần có nhãn đúng bằng `"12<\|im_end\|>"` |
| `test_user_only_conversation_has_no_supervision`       | Chỉ có câu hỏi → không học gì                                               |
| `test_every_assistant_turn_is_supervised`              | Hội thoại nhiều lượt: mọi câu trả lời đều được học                   |
| `test_prompt_matches_training_format`                  | **Prompt lúc chat trùng từng token với lúc train**                       |
| `test_unknown_role_raises`                             | Vai lạ bị từ chối                                                               |

<details>
<summary><b>Code test</b></summary>

```python
import pytest

from trainer.chat_template import IM_END_TOKEN, build_prompt, encode_conversation
from trainer.losses import IGNORE_INDEX

CONVERSATION = [
    {"role": "user", "content": "Bảy cộng năm bằng bao nhiêu?"},
    {"role": "assistant", "content": "12"},
]


def supervised_text(tokenizer, input_ids, labels):
    """Giải mã riêng những token có nhãn (được học)."""
    kept = [token for token, label in zip(input_ids, labels) if label != IGNORE_INDEX]
    return tokenizer.decode(kept, skip_special_tokens=False)


def test_ids_and_labels_have_same_length(chat_tokenizer):
    input_ids, labels = encode_conversation(chat_tokenizer, CONVERSATION)

    assert len(input_ids) == len(labels)


def test_full_text_follows_template(chat_tokenizer):
    input_ids, _ = encode_conversation(chat_tokenizer, CONVERSATION)

    text = chat_tokenizer.decode(input_ids, skip_special_tokens=False)

    assert text == (
        "<|bos|><|im_start|>user\nBảy cộng năm bằng bao nhiêu?<|im_end|>\n"
        "<|im_start|>assistant\n12<|im_end|>\n"
    )


def test_only_assistant_answer_and_im_end_are_supervised(chat_tokenizer):
    """Tiêu chí xong chặng 4: nhãn chỉ phủ token assistant."""
    input_ids, labels = encode_conversation(chat_tokenizer, CONVERSATION)

    assert supervised_text(chat_tokenizer, input_ids, labels) == "12" + IM_END_TOKEN


def test_user_only_conversation_has_no_supervision(chat_tokenizer):
    _, labels = encode_conversation(chat_tokenizer, CONVERSATION[:1])

    assert all(label == IGNORE_INDEX for label in labels)


def test_every_assistant_turn_is_supervised(chat_tokenizer):
    messages = CONVERSATION + [
        {"role": "user", "content": "Tính mười lăm trừ sáu."},
        {"role": "assistant", "content": "9"},
    ]

    input_ids, labels = encode_conversation(chat_tokenizer, messages)

    assert supervised_text(chat_tokenizer, input_ids, labels) == "12<|im_end|>9<|im_end|>"


def test_prompt_matches_training_format(chat_tokenizer):
    """Lúc chạy, prompt + câu trả lời phải trùng đúng chuỗi token đã dùng lúc train."""
    train_ids, _ = encode_conversation(chat_tokenizer, CONVERSATION)
    prompt_ids = build_prompt(chat_tokenizer, CONVERSATION[:1])
    answer_ids = chat_tokenizer.encode("12").ids + [chat_tokenizer.token_to_id(IM_END_TOKEN)]

    assert train_ids[: len(prompt_ids) + len(answer_ids)] == prompt_ids + answer_ids


def test_unknown_role_raises(chat_tokenizer):
    with pytest.raises(ValueError, match="role"):
        encode_conversation(chat_tokenizer, [{"role": "robot", "content": "xin chào"}])
```

</details>

### Fixture cho phần F

Thêm vào **cuối** `tests/trainer/conftest.py`. Tokenizer tí hon train trên chính câu hỏi của thế giới đồ chơi, có đủ token đặc biệt, chạy dưới 1 giây.

<details>
<summary><b>Code thêm vào <code>tests/trainer/conftest.py</code></b></summary>

```python
@pytest.fixture
def tokenizer_file(tmp_path):
    """Tokenizer tí hon (đủ token đặc biệt) train trên câu hỏi của thế giới đồ chơi, lưu ra file."""
    from tokenizer.train_tokenizer import build_tokenizer
    from toyworld.arithmetic import QUESTION_TEMPLATES, all_problems, question_text

    lines = [
        f"{question_text(problem, index)} {problem.answer}"
        for problem in all_problems()
        for index in range(len(QUESTION_TEMPLATES))
    ]
    data_path = tmp_path / "tokenizer_data.txt"
    data_path.write_text("\n".join(lines), encoding="utf-8")

    tokenizer = build_tokenizer([str(data_path)], vocab_size=500)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    return tokenizer_path


@pytest.fixture
def chat_tokenizer(tokenizer_file):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_file))
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/trainer/test_chat_template.py -v` → **7 passed**

---

---

# F3 — `scripts/prepare_sft.py`

## 1. Vị trí

```
thế giới đồ chơi → ▶dataset/sft/train.jsonl, eval.jsonl◀ → dataset → train
```

## 2. Làm gì

Chia bài toán thành train (90%) / eval (10%), mỗi bài sinh 3 mẫu (3 cách hỏi), ghi ra jsonl. Mỗi dòng giữ thêm trường `problem` để lúc chấm điểm biết đáp án đúng.

```json
{"messages": [{"role": "user", "content": "Mười tám trừ bốn bằng bao nhiêu?"}, {"role": "assistant", "content": "14"}], "problem": {"a": 18, "b": 4, "op": "-"}}
```

## 3. Vào → Ra

```powershell
python -m scripts.prepare_sft
```

```
train: 605 bài × 3 cách hỏi | eval: 67 bài × 3 cách hỏi -> dataset\sft
```

Mỗi hội thoại dài **20–25 token**, trong đó chỉ **2 token** được học (con số + `<|im_end|>`).

## 4. Lời giải

Script ngắn, không chia TODO:

<details>
<summary><b>Lời giải</b></summary>

```python
"""Sinh dữ liệu SFT từ thế giới đồ chơi: dataset/sft/train.jsonl và eval.jsonl."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from toyworld.arithmetic import QUESTION_TEMPLATES, Problem, all_problems, split_problems, to_messages

DEFAULT_EVAL_FRACTION = 0.1


def build_samples(problems: list[Problem]) -> list[dict]:
    """Mỗi bài toán sinh ra một mẫu cho mỗi cách hỏi."""
    samples = []
    for problem in problems:
        for template_index in range(len(QUESTION_TEMPLATES)):
            samples.append({"messages": to_messages(problem, template_index), "problem": asdict(problem)})
    return samples


def write_jsonl(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out_file:
        for sample in samples:
            out_file.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh dữ liệu SFT từ thế giới đồ chơi")
    parser.add_argument("--out-dir", default="dataset/sft")
    parser.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_problems, eval_problems = split_problems(all_problems(), args.eval_fraction, args.seed)
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", build_samples(train_problems))
    write_jsonl(out_dir / "eval.jsonl", build_samples(eval_problems))
    print(
        f"train: {len(train_problems)} bài × {len(QUESTION_TEMPLATES)} cách hỏi | "
        f"eval: {len(eval_problems)} bài × {len(QUESTION_TEMPLATES)} cách hỏi -> {out_dir}"
    )


if __name__ == "__main__":
    main()
```

</details>

## 5. Test

Phần logic đã được test ở F1. Kiểm tra bằng cách chạy lệnh ở mục 3 và mở `dataset/sft/train.jsonl`.

## 6. Xong khi

- [ ] Có `dataset/sft/train.jsonl` (1815 dòng) và `eval.jsonl` (201 dòng)

---

---

# F4 — `trainer/sft_dataset.py`

## 1. Vị trí

```
jsonl → chat template → ▶(x, y) cho DataLoader◀ → train
```

## 2. Làm gì

Giống `PretrainDataset` ở chỗ trả `(x, y)` với `y` dịch 1 vị trí. Khác ở 2 điểm:

**1. Nhãn lấy từ chat template, không phải chính token:**

```
x = input_ids[:-1]
y = labels[1:]          ← labels đã có -100 ở phần không học
```

**2. Pad cho đủ `seq_len`** vì các hội thoại dài ngắn khác nhau:

- `x` pad bằng `<|pad|>`
- `y` pad bằng `-100` → vị trí pad không bị học

**Không cần attention mask:** pad nằm ở **cuối** chuỗi, mà model là causal — token thật chỉ nhìn về trước nên không bao giờ nhìn thấy pad.

**Không tự cắt ngắn:** hội thoại dài hơn `seq_len + 1` thì báo lỗi ngay, thay vì âm thầm cắt mất phần câu trả lời.

## 3. Vào → Ra

|                | Shape                                     |
| -------------- | ----------------------------------------- |
| `dataset[i]` | `(x, y)`, mỗi cái `(seq_len,)` long |

## 4. Khung code

```python
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer: Tokenizer, seq_len: int) -> None:
        # TODO 1: seq_len < 1 → ValueError; pad_id = token_to_id("<|pad|>"), thiếu → ValueError
        # TODO 2: đọc từng dòng jsonl (bỏ dòng trống) → encode_conversation(tokenizer, messages)
        # TODO 3: len(input_ids) > seq_len + 1 → ValueError có số dòng
        # TODO 4: lưu (input_ids, labels) vào self.examples

    def __len__(self) -> int:
        # TODO 5

    def __getitem__(self, index):
        # TODO 6: x = input_ids[:-1]; y = labels[1:]
        # TODO 7: n_pad = seq_len - len(x); x += [pad_id]*n_pad; y += [-100]*n_pad
        # TODO 8: trả 2 tensor long
```

<details>
<summary><b>Lời giải</b></summary>

```python
"""Dataset SFT: đọc file jsonl hội thoại, trả (x, y) đã dịch nhãn và pad."""

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from trainer.chat_template import encode_conversation
from trainer.losses import IGNORE_INDEX

PAD_TOKEN = "<|pad|>"


class SFTDataset(Dataset):
    """Mỗi dòng jsonl là một hội thoại {"messages": [...]}.

    Shape:
        x: (seq_len,) long — token vào, pad bằng <|pad|>
        y: (seq_len,) long — nhãn đã dịch 1 vị trí, pad bằng -100
    Lưu ý:
        Pad nằm ở CUỐI chuỗi. Model là causal nên token thật không bao giờ nhìn thấy pad,
        không cần attention mask. Nhãn của vị trí pad là -100 nên pad không bị học.
    """

    def __init__(self, jsonl_path: str | Path, tokenizer: Tokenizer, seq_len: int) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len={seq_len} phải >= 1")
        pad_id = tokenizer.token_to_id(PAD_TOKEN)
        if pad_id is None:
            raise ValueError(f"tokenizer thiếu {PAD_TOKEN}")

        self.seq_len = seq_len
        self.pad_id = pad_id
        self.examples: list[tuple[list[int], list[int]]] = []
        with open(jsonl_path, encoding="utf-8") as jsonl_file:
            for line_number, line in enumerate(jsonl_file, start=1):
                if not line.strip():
                    continue
                input_ids, labels = encode_conversation(tokenizer, json.loads(line)["messages"])
                if len(input_ids) > seq_len + 1:
                    raise ValueError(
                        f"dòng {line_number}: {len(input_ids)} token, vượt seq_len + 1 = {seq_len + 1}"
                    )
                self.examples.append((input_ids, labels))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = self.examples[index]
        x = input_ids[:-1]
        y = labels[1:]  # dịch 1 vị trí: vị trí t đoán token t+1

        n_pad = self.seq_len - len(x)
        x = x + [self.pad_id] * n_pad
        y = y + [IGNORE_INDEX] * n_pad
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
```

</details>

## 5. Test — `tests/trainer/test_sft_dataset.py`

| Test                                             | Chứng minh                                                |
| ------------------------------------------------ | ---------------------------------------------------------- |
| `test_length_is_number_of_conversations`       | `len` đúng                                             |
| `test_fixed_shape_and_dtype`                   | `(seq_len,)` long                                        |
| `test_labels_are_shifted_by_one`               | **y = labels dịch 1 vị trí**                      |
| `test_padding_uses_pad_token_and_ignore_label` | Pad đúng cho cả x và y                                 |
| `test_supervised_positions_predict_answer`     | Các vị trí có nhãn đoán ra đúng`"12<\|im_end\|>"` |
| `test_conversation_longer_than_seq_len_raises` | Quá dài bị từ chối, không cắt âm thầm             |

<details>
<summary><b>Code test</b></summary>

```python
import json

import pytest
import torch

from trainer.chat_template import encode_conversation
from trainer.losses import IGNORE_INDEX
from trainer.sft_dataset import PAD_TOKEN, SFTDataset

SEQ_LEN = 40
CONVERSATIONS = [
    [{"role": "user", "content": "Bảy cộng năm bằng bao nhiêu?"}, {"role": "assistant", "content": "12"}],
    [{"role": "user", "content": "Tính mười lăm trừ sáu."}, {"role": "assistant", "content": "9"}],
]


@pytest.fixture
def jsonl_path(tmp_path):
    path = tmp_path / "sft.jsonl"
    path.write_text("\n".join(json.dumps({"messages": m}, ensure_ascii=False) for m in CONVERSATIONS) + "\n",
                    encoding="utf-8")
    return path


def test_length_is_number_of_conversations(jsonl_path, chat_tokenizer):
    assert len(SFTDataset(jsonl_path, chat_tokenizer, SEQ_LEN)) == len(CONVERSATIONS)


def test_fixed_shape_and_dtype(jsonl_path, chat_tokenizer):
    x, y = SFTDataset(jsonl_path, chat_tokenizer, SEQ_LEN)[0]

    assert x.shape == (SEQ_LEN,) and y.shape == (SEQ_LEN,)
    assert x.dtype == torch.long and y.dtype == torch.long


def test_labels_are_shifted_by_one(jsonl_path, chat_tokenizer):
    input_ids, labels = encode_conversation(chat_tokenizer, CONVERSATIONS[0])
    n = len(input_ids) - 1

    x, y = SFTDataset(jsonl_path, chat_tokenizer, SEQ_LEN)[0]

    assert x[:n].tolist() == input_ids[:-1]
    assert y[:n].tolist() == labels[1:]


def test_padding_uses_pad_token_and_ignore_label(jsonl_path, chat_tokenizer):
    input_ids, _ = encode_conversation(chat_tokenizer, CONVERSATIONS[0])
    n = len(input_ids) - 1
    pad_id = chat_tokenizer.token_to_id(PAD_TOKEN)

    x, y = SFTDataset(jsonl_path, chat_tokenizer, SEQ_LEN)[0]

    assert (x[n:] == pad_id).all()
    assert (y[n:] == IGNORE_INDEX).all()


def test_supervised_positions_predict_answer(jsonl_path, chat_tokenizer):
    """Các vị trí có nhãn phải đoán ra đúng câu trả lời + <|im_end|>."""
    _, y = SFTDataset(jsonl_path, chat_tokenizer, SEQ_LEN)[0]

    targets = y[y != IGNORE_INDEX].tolist()

    assert chat_tokenizer.decode(targets, skip_special_tokens=False) == "12<|im_end|>"


def test_conversation_longer_than_seq_len_raises(jsonl_path, chat_tokenizer):
    with pytest.raises(ValueError, match="seq_len"):
        SFTDataset(jsonl_path, chat_tokenizer, seq_len=5)
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/trainer/test_sft_dataset.py -v` → **6 passed**

---

---

# F5 — `trainer/sft_config.py` + `configs/sft_tiny.json`

## 1. Vị trí

Tham số cho một lần chạy SFT — mỗi chặng một config riêng, đúng quy ước trong CLAUDE.md.

## 2. Làm gì

Gần giống `TrainConfig` của pretrain, khác 4 trường:

| Trường                      | Ý nghĩa                                          |
| ----------------------------- | -------------------------------------------------- |
| `train_file`, `eval_file` | File`.jsonl` (thay cho `.bin`)                 |
| `init_checkpoint`           | Trọng số pretrain để**bắt đầu**       |
| `tokenizer`                 | Đường dẫn tokenizer (dataset cần để encode) |

**Không có kiến trúc model** trong config này: `train_sft` đọc `model_config` từ chính `init_checkpoint`, nên không thể lệch với trọng số.

**Đọc JSON bằng `utf-8-sig`**: Notepad và PowerShell 5.1 hay lưu JSON kèm BOM ở đầu file, `json.loads` với `utf-8` thường sẽ báo `Unexpected UTF-8 BOM`. Claude gặp đúng lỗi này khi chạy thử. `TrainConfig.from_json` của pretrain nên sửa giống vậy.

## 3. Vào → Ra

`configs/sft_tiny.json` — cấu hình đã chạy ra 97,8% / 80,1%:

```json
{
  "train_file": "dataset/sft/train.jsonl",
  "eval_file": "dataset/sft/eval.jsonl",
  "init_checkpoint": "checkpoints/pretrain.pt",
  "tokenizer": "tokenizer/tokenizer.json",
  "out_dir": "checkpoints",
  "seq_len": 32,
  "batch_size": 32,
  "accum_steps": 1,
  "max_steps": 2000,
  "warmup_steps": 100,
  "lr": 0.001,
  "min_lr": 0.0001,
  "weight_decay": 0.1,
  "grad_clip": 1.0,
  "log_interval": 250,
  "eval_interval": 500,
  "eval_batches": 6,
  "save_interval": 1000,
  "dtype": "fp32",
  "device": "cpu",
  "seed": 0
}
```

- `seq_len: 32` — hội thoại dài nhất 25 token.
- `lr: 0.001` — cao hơn pretrain (0.0006) vì dữ liệu nhỏ, cần học nhanh.
- `max_steps: 2000` × `batch_size: 32` ≈ 35 lần duyệt qua 1815 mẫu. Với SFT trên tập nhỏ, lặp nhiều epoch là bình thường.

## 4. Lời giải

<details>
<summary><b>Lời giải</b></summary>

```python
"""Tham số cho một lần chạy SFT."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

ALLOWED_DTYPES = ("fp32", "fp16", "bf16")
POSITIVE_INT_FIELDS = (
    "seq_len",
    "batch_size",
    "accum_steps",
    "max_steps",
    "log_interval",
    "eval_interval",
    "eval_batches",
    "save_interval",
)


@dataclass(frozen=True)
class SFTConfig:
    """Cấu hình SFT. Khác TrainConfig của pretrain ở 3 trường:
    train_file / eval_file là .jsonl, và init_checkpoint là trọng số pretrain để bắt đầu.
    """

    train_file: str
    eval_file: str
    init_checkpoint: str
    tokenizer: str
    out_dir: str
    seq_len: int
    batch_size: int
    accum_steps: int
    max_steps: int
    warmup_steps: int
    lr: float
    min_lr: float
    weight_decay: float
    grad_clip: float
    log_interval: int
    eval_interval: int
    eval_batches: int
    save_interval: int
    dtype: str
    device: str
    seed: int

    def __post_init__(self) -> None:
        for name in POSITIVE_INT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name}={value!r} phải là int > 0")
        if not 0 <= self.warmup_steps <= self.max_steps:
            raise ValueError(f"warmup_steps={self.warmup_steps} phải trong [0, max_steps={self.max_steps}]")
        if not (math.isfinite(self.lr) and self.lr > 0):
            raise ValueError(f"lr={self.lr} phải hữu hạn và > 0")
        if not 0 <= self.min_lr <= self.lr:
            raise ValueError(f"min_lr={self.min_lr} phải trong [0, lr={self.lr}]")
        if self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("weight_decay phải >= 0 và grad_clip phải > 0")
        if self.dtype not in ALLOWED_DTYPES:
            raise ValueError(f"dtype={self.dtype!r} phải thuộc {ALLOWED_DTYPES}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SFTConfig":
        valid_names = {field.name for field in fields(cls)}
        unknown = set(data) - valid_names
        missing = valid_names - set(data)
        if unknown or missing:
            raise ValueError(f"khóa không hợp lệ: {sorted(unknown)}; khóa bị thiếu: {sorted(missing)}")
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "SFTConfig":
        # utf-8-sig: đọc được cả file có BOM (Notepad, PowerShell 5.1 hay tự thêm)
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

</details>

## 5. Test

Có 2 test trong `test_train_sft.py` (F6): khóa lạ bị từ chối, và đọc được JSON có BOM.

## 6. Xong khi

- [ ] Tạo `trainer/sft_config.py` và `configs/sft_tiny.json`

---

---

# F6 — `trainer/train_sft.py`

## 1. Vị trí

```
checkpoints/pretrain.pt → ▶nạp trọng số → train trên jsonl◀ → checkpoints/sft.pt
```

## 2. Làm gì

Vòng lặp **giống hệt pretrain**, dùng lại nguyên các module của PHẦN D:

| Dùng lại                                                    | Từ                        |
| ------------------------------------------------------------- | -------------------------- |
| `build_optimizer`                                           | D4                         |
| `get_lr`                                                    | D5                         |
| `load_weights`, `save_checkpoint`                         | D6                         |
| `Throughput`, `format_log`                                | D7                         |
| `build_loader`, `endless`, `evaluate`, `TORCH_DTYPES` | D9 (`train_pretrain.py`) |
| `next_token_loss` (đã có `IGNORE_INDEX`)               | D3                         |

**Chỉ khác 3 chỗ:**

1. **Nạp trọng số pretrain bằng `load_weights`** — không phải resume. Optimizer mới, learning rate schedule mới, step bắt đầu từ 0. Đây chính là lý do D6 tách `load_weights` khỏi `load_checkpoint`.
2. **Kiến trúc lấy từ checkpoint**, và kiểm tra `vocab_size` khớp tokenizer trước khi train.
3. **Đếm tokens/s theo token được học** (`y != -100`), vì phần lớn token trong batch là prompt và pad.

**Trùng lặp với `train_pretrain.py`:** vòng lặp khoảng 30 dòng giống nhau. Đây là lần thứ hai lặp lại — đến chặng DPO (lần thứ ba) nên tách phần chung ra `trainer/loop.py`.

## 3. Vào → Ra

```powershell
python -m trainer.train_sft --config configs/sft_tiny.json
```

Log thật (CPU):

```
step    250/2000 | loss  1.053 | lr 9.86e-04 | gnorm  3.414 |       617 tok/s | vram      0 MB
step    500 | val loss  1.002
step   1000/2000 | loss  0.520 | lr 5.88e-04 | gnorm  3.519 |       614 tok/s | vram      0 MB
step   1000 | val loss  0.819
step   1500 | val loss  0.566
step   2000/2000 | loss  0.208 | lr 1.00e-04 | gnorm  1.881 |       571 tok/s | vram      0 MB
step   2000 | val loss  0.493
```

`tok/s` thấp vì chỉ đếm 2 token được học mỗi mẫu; thực tế model xử lý khoảng 10 lần số đó.

## 4. Khung code

```python
CHECKPOINT_NAME = "sft.pt"


def load_pretrained(config: SFTConfig, tokenizer, device) -> tuple[CausalLM, ModelConfig]:
    # TODO 1: torch.load(init_checkpoint, weights_only=False) → ModelConfig.from_dict(payload["model_config"])
    # TODO 2: vocab_size khác tokenizer.get_vocab_size() → ValueError có chữ "vocab_size"
    # TODO 3: model = CausalLM(...); load_weights(init_checkpoint, model); trả (model.to(device), model_config)


def train_sft(config: SFTConfig) -> dict:
    # TODO 4: seed, device, dtype, autocast, scaler — chép từ train_pretrain
    # TODO 5: tokenizer; model, model_config = load_pretrained(...); optimizer
    # TODO 6: train_loader / eval_loader = build_loader(SFTDataset(...), ...)
    # TODO 7: vòng lặp step giống train_pretrain (không có resume)
    #         throughput.update(int((y != -100).sum()))
    # TODO 8: save_checkpoint(checkpoints/sft.pt, ...); trả dict thống kê


def main() -> None:
    # TODO 9: argparse --config → SFTConfig.from_json → train_sft
```

<details>
<summary><b>Lời giải</b></summary>

```python
"""SFT: nạp trọng số pretrain, dạy model trả lời theo định dạng chat."""

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from core.config import ModelConfig
from core.model import CausalLM
from trainer.checkpoint import load_weights, save_checkpoint
from trainer.losses import next_token_loss
from trainer.lr_scheduler import get_lr
from trainer.metrics import Throughput, format_log, max_vram_mb
from trainer.optim import build_optimizer
from trainer.sft_config import SFTConfig
from trainer.sft_dataset import SFTDataset
from trainer.train_pretrain import TORCH_DTYPES, build_loader, endless, evaluate

CHECKPOINT_NAME = "sft.pt"


def load_pretrained(config: SFTConfig, tokenizer: Tokenizer, device: torch.device) -> tuple[CausalLM, ModelConfig]:
    """Dựng model theo kiến trúc lưu trong checkpoint pretrain rồi nạp trọng số (không nạp optimizer)."""
    payload = torch.load(config.init_checkpoint, map_location="cpu", weights_only=False)
    model_config = ModelConfig.from_dict(payload["model_config"])
    if model_config.vocab_size != tokenizer.get_vocab_size():
        raise ValueError(
            f"vocab_size của checkpoint ({model_config.vocab_size}) khác tokenizer ({tokenizer.get_vocab_size()})"
        )
    model = CausalLM(model_config)
    load_weights(config.init_checkpoint, model)
    return model.to(device), model_config


def train_sft(config: SFTConfig) -> dict[str, Any]:
    """Chạy SFT. Trả về thống kê cuối cùng."""
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    dtype = TORCH_DTYPES[config.dtype]
    use_amp = dtype is not torch.float32 and device.type == "cuda"

    def autocast():
        return torch.autocast(device_type=device.type, dtype=dtype) if use_amp else nullcontext()

    scaler = torch.amp.GradScaler(device.type, enabled=(use_amp and dtype is torch.float16))

    tokenizer = Tokenizer.from_file(config.tokenizer)
    model, model_config = load_pretrained(config, tokenizer, device)
    optimizer = build_optimizer(model, lr=config.lr, weight_decay=config.weight_decay)

    train_loader = build_loader(
        SFTDataset(config.train_file, tokenizer, config.seq_len), config.batch_size, config.seed, shuffle=True
    )
    eval_loader = build_loader(
        SFTDataset(config.eval_file, tokenizer, config.seq_len), config.batch_size, config.seed, shuffle=False
    )

    checkpoint_path = Path(config.out_dir) / CHECKPOINT_NAME
    batches = endless(train_loader)
    throughput = Throughput()
    model.train()

    train_loss, eval_loss, step = float("nan"), float("nan"), 0
    for step in range(config.max_steps):
        lr = get_lr(step, config.lr, config.min_lr, config.warmup_steps, config.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(config.accum_steps):
            x, y = next(batches)
            x, y = x.to(device), y.to(device)
            with autocast():
                loss = next_token_loss(model(x), y) / config.accum_steps
            scaler.scale(loss).backward()
            train_loss += loss.item()
            throughput.update(int((y != -100).sum()))  # chỉ đếm token thật sự được học

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
        scaler.step(optimizer)
        scaler.update()

        if (step + 1) % config.log_interval == 0:
            print(format_log(step + 1, config.max_steps, train_loss, lr, grad_norm,
                             throughput.tokens_per_second(), max_vram_mb()))
            throughput.reset()

        if (step + 1) % config.eval_interval == 0:
            eval_loss = evaluate(model, eval_loader, config.eval_batches, device, autocast)
            print(f"step {step + 1:>6} | val loss {eval_loss:6.3f}")

        if (step + 1) % config.save_interval == 0:
            save_checkpoint(checkpoint_path, model, optimizer, step + 1, model_config, config, scaler)

    save_checkpoint(checkpoint_path, model, optimizer, step + 1, model_config, config, scaler)
    return {"step": step + 1, "train_loss": train_loss, "eval_loss": eval_loss}


def main() -> None:
    """python -m trainer.train_sft --config configs/sft_tiny.json"""
    parser = argparse.ArgumentParser(description="SFT trên thế giới đồ chơi")
    parser.add_argument("--config", default="configs/sft_tiny.json")
    args = parser.parse_args()
    print(f"Xong: {train_sft(SFTConfig.from_json(args.config))}")


if __name__ == "__main__":
    main()
```

</details>

## 5. Test — `tests/trainer/test_train_sft.py`

| Test                                    | Chứng minh                                                                             |
| --------------------------------------- | --------------------------------------------------------------------------------------- |
| `test_sft_runs_and_saves_checkpoint`  | Chạy 3 bước, có checkpoint, loss hữu hạn                                          |
| `test_sft_memorizes_toy_set`          | **Tiêu chí xong chặng 4:** 150 bước → trả lời đúng 100% tập đồ chơi |
| `test_starts_from_pretrained_weights` | Với`lr ≈ 0`, trọng số sau SFT = trọng số pretrain                               |
| `test_vocab_mismatch_raises`          | Checkpoint và tokenizer lệch vocab bị từ chối                                      |
| `test_reply_stops_at_im_end`          | Câu trả lời dừng đúng ở`<\|im_end\|>`                                            |
| `test_sft_config_rejects_unknown_key` | Khóa lạ trong config bị từ chối                                                    |
| `test_sft_config_reads_json_with_bom` | Đọc được JSON có BOM                                                              |

Test `test_starts_from_pretrained_weights` quan trọng hơn vẻ ngoài: model tí hon trong test **tự học thuộc được từ số ngẫu nhiên**, nên nếu quên `load_weights` thì `test_sft_memorizes_toy_set` vẫn xanh. Claude đã thử bỏ dòng `load_weights` — chỉ test này bắt được.

<details>
<summary><b>Code test</b></summary>

```python
import json
import math

import pytest
import torch

from core.config import ModelConfig
from core.model import CausalLM
from inference.chat import reply, toy_accuracy
from trainer.checkpoint import save_checkpoint
from trainer.optim import build_optimizer
from trainer.sft_config import SFTConfig
from trainer.train_sft import CHECKPOINT_NAME, train_sft
from toyworld.arithmetic import Problem, to_messages

TOY_PROBLEMS = [Problem(1, 2, "+"), Problem(3, 1, "-"), Problem(2, 2, "+"), Problem(4, 3, "-")]


@pytest.fixture
def sft_setup(tmp_path, chat_tokenizer, tokenizer_file, make_train_config):
    """Checkpoint "pretrain" tí hon + file jsonl đồ chơi + hàm tạo SFTConfig."""
    model_config = ModelConfig(
        vocab_size=chat_tokenizer.get_vocab_size(), d_model=32, n_layers=2, n_heads=2,
        ffn_hidden=64, max_seq_len=48, rope_theta=10000.0, norm_eps=1e-6, tie_embedding=True,
    )
    model = CausalLM(model_config)
    pretrain_path = tmp_path / "pretrain.pt"
    save_checkpoint(pretrain_path, model, build_optimizer(model, 1e-3, 0.1), 0, model_config, make_train_config())

    data_path = tmp_path / "toy.jsonl"
    with open(data_path, "w", encoding="utf-8") as data_file:
        for problem in TOY_PROBLEMS:
            sample = {"messages": to_messages(problem, 0), "problem": problem.__dict__}
            data_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def make_config(**overrides) -> SFTConfig:
        values = dict(
            train_file=str(data_path), eval_file=str(data_path), init_checkpoint=str(pretrain_path),
            tokenizer=str(tokenizer_file), out_dir=str(tmp_path / "out"), seq_len=40, batch_size=4,
            accum_steps=1, max_steps=3, warmup_steps=1, lr=3e-3, min_lr=1e-4, weight_decay=0.0,
            grad_clip=1.0, log_interval=1000, eval_interval=1000, eval_batches=1, save_interval=1000,
            dtype="fp32", device="cpu", seed=0,
        )
        values.update(overrides)
        return SFTConfig(**values)

    return make_config, data_path


def test_sft_runs_and_saves_checkpoint(sft_setup, tmp_path):
    make_config, _ = sft_setup

    stats = train_sft(make_config())

    assert stats["step"] == 3
    assert math.isfinite(stats["train_loss"])
    assert (tmp_path / "out" / CHECKPOINT_NAME).exists()


def test_sft_memorizes_toy_set(sft_setup, chat_tokenizer, tmp_path):
    """Tiêu chí xong chặng 4: học thuộc tập đồ chơi."""
    make_config, data_path = sft_setup

    train_sft(make_config(max_steps=150, warmup_steps=10))

    payload = torch.load(tmp_path / "out" / CHECKPOINT_NAME, map_location="cpu", weights_only=False)
    model = CausalLM(ModelConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model"])

    assert toy_accuracy(model, chat_tokenizer, data_path) == 1.0


def test_starts_from_pretrained_weights(sft_setup, tmp_path):
    """SFT phải bắt đầu từ trọng số pretrain, không phải từ số ngẫu nhiên."""
    make_config, _ = sft_setup
    config = make_config(max_steps=1, warmup_steps=1, lr=1e-9, min_lr=0.0)

    train_sft(config)

    pretrained = torch.load(config.init_checkpoint, map_location="cpu", weights_only=False)["model"]
    after_sft = torch.load(tmp_path / "out" / CHECKPOINT_NAME, map_location="cpu", weights_only=False)["model"]
    for key, value in pretrained.items():
        torch.testing.assert_close(after_sft[key], value, msg=f"khác nhau ở {key}")


def test_vocab_mismatch_raises(sft_setup, tmp_path, make_train_config):
    make_config, _ = sft_setup
    wrong_config = ModelConfig(
        vocab_size=50, d_model=32, n_layers=1, n_heads=2, ffn_hidden=64,
        max_seq_len=48, rope_theta=10000.0, norm_eps=1e-6, tie_embedding=True,
    )
    wrong_model = CausalLM(wrong_config)
    wrong_path = tmp_path / "wrong.pt"
    save_checkpoint(wrong_path, wrong_model, build_optimizer(wrong_model, 1e-3, 0.1), 0,
                    wrong_config, make_train_config())

    with pytest.raises(ValueError, match="vocab_size"):
        train_sft(make_config(init_checkpoint=str(wrong_path)))


def test_reply_stops_at_im_end(sft_setup, chat_tokenizer, tmp_path):
    """Câu trả lời không chứa <|im_end|> và không kéo dài quá max_new_tokens."""
    make_config, _ = sft_setup
    train_sft(make_config(max_steps=150, warmup_steps=10))
    payload = torch.load(tmp_path / "out" / CHECKPOINT_NAME, map_location="cpu", weights_only=False)
    model = CausalLM(ModelConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model"])

    answer = reply(model, chat_tokenizer, to_messages(TOY_PROBLEMS[0], 0)[:1])

    assert answer == "3"


def test_sft_config_rejects_unknown_key(sft_setup):
    make_config, _ = sft_setup

    with pytest.raises(ValueError, match="foo"):
        SFTConfig.from_dict({**make_config().to_dict(), "foo": 1})


def test_sft_config_reads_json_with_bom(sft_setup, tmp_path):
    """File JSON lưu bằng Notepad/PowerShell 5.1 hay có BOM ở đầu."""
    make_config, _ = sft_setup
    config = make_config()
    path = tmp_path / "config_bom.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8-sig")

    assert SFTConfig.from_json(path) == config
```

</details>

## 6. Xong khi

- [ ] `python -m pytest tests/trainer/test_train_sft.py -v` → **7 passed**
- [ ] Chạy thật: `python -m trainer.train_sft` (khoảng 3–4 phút CPU)
- [ ] Val loss giảm dần

---

---

# F7 — `inference/chat.py`

## 1. Vị trí

```
câu hỏi → build_prompt → generate (dừng ở <|im_end|>) → câu trả lời
```

## 2. Làm gì

| Hàm                                           | Việc                                                                               |
| ---------------------------------------------- | ----------------------------------------------------------------------------------- |
| `reply(model, tokenizer, messages)`          | Sinh câu trả lời, cắt ở`<\|im_end\|>`, decode                                  |
| `toy_accuracy(model, tokenizer, jsonl_path)` | Hỏi từng câu trong file (greedy), chấm bằng`is_correct`, trả tỉ lệ đúng |
| `main()`                                     | `--eval`: in độ chính xác train/eval; không có cờ: vòng lặp chat         |

Dùng lại `generate` của E2 với `eos_id = <|im_end|>` — đây là chỗ KV cache được dùng với **cả prompt nhiều token** (prefill), lý do phải sửa lỗi mask ở E1 trước khi sang chặng này.

Chấm điểm dùng **greedy** (`temperature=0`) để kết quả ổn định giữa các lần chạy.

## 3. Vào → Ra

```powershell
python -m inference.chat --eval
```

```
train: 97.8% trả lời đúng
eval : 80.1% trả lời đúng
```

```powershell
python -m inference.chat
```

```
Gõ câu hỏi (Enter trống để thoát). Ví dụ: Bảy cộng năm bằng bao nhiêu?
Bạn: Bảy cộng năm bằng bao nhiêu?
Model: 12
```

## 4. Lời giải

<details>
<summary><b>Lời giải</b></summary>

```python
"""Hỏi–đáp với model đã SFT, và đo độ chính xác trên thế giới đồ chơi."""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from inference.generate import generate, load_model
from toyworld.arithmetic import Problem, is_correct
from trainer.chat_template import IM_END_TOKEN, build_prompt

DEFAULT_MAX_NEW_TOKENS = 16


@torch.no_grad()
def reply(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_k: int | None = None,
) -> str:
    """Sinh câu trả lời của assistant cho hội thoại `messages`; dừng ở <|im_end|>."""
    prompt_ids = build_prompt(tokenizer, messages)
    im_end_id = tokenizer.token_to_id(IM_END_TOKEN)

    output = generate(
        model,
        torch.tensor([prompt_ids], dtype=torch.long),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=im_end_id,
    )
    new_ids = output[0, len(prompt_ids):].tolist()
    if im_end_id in new_ids:
        new_ids = new_ids[: new_ids.index(im_end_id)]
    return tokenizer.decode(new_ids).strip()


def toy_accuracy(model: torch.nn.Module, tokenizer: Tokenizer, jsonl_path: str | Path) -> float:
    """Tỉ lệ trả lời đúng (greedy) trên một file jsonl của thế giới đồ chơi."""
    correct = total = 0
    with open(jsonl_path, encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if not line.strip():
                continue
            sample = json.loads(line)
            question = [message for message in sample["messages"] if message["role"] != "assistant"]
            answer_text = reply(model, tokenizer, question)
            correct += is_correct(answer_text, Problem(**sample["problem"]))
            total += 1
    return correct / max(total, 1)


def main() -> None:
    """python -m inference.chat            → hỏi–đáp
    python -m inference.chat --eval     → đo độ chính xác trên train/eval"""
    parser = argparse.ArgumentParser(description="Chat với model đã SFT")
    parser.add_argument("--checkpoint", default="checkpoints/sft.pt")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--eval", action="store_true", help="đo độ chính xác thay vì chat")
    parser.add_argument("--data-dir", default="dataset/sft")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = load_model(args.checkpoint)

    if args.eval:
        for split in ("train", "eval"):
            accuracy = toy_accuracy(model, tokenizer, Path(args.data_dir) / f"{split}.jsonl")
            print(f"{split:5}: {accuracy:.1%} trả lời đúng")
        return

    print("Gõ câu hỏi (Enter trống để thoát). Ví dụ: Bảy cộng năm bằng bao nhiêu?")
    while True:
        question = input("Bạn: ").strip()
        if not question:
            break
        print("Model:", reply(model, tokenizer, [{"role": "user", "content": question}]))


if __name__ == "__main__":
    main()
```

</details>

## 5. Test

`reply` và `toy_accuracy` được test trong `tests/trainer/test_train_sft.py` (F6): `test_sft_memorizes_toy_set` và `test_reply_stops_at_im_end`.

## 6. Xong khi

- [ ] `python -m inference.chat --eval` → train trên 95%, eval trên 70%
- [ ] Chat thử vài câu, kể cả câu ngoài phép tính, để thấy giới hạn

---

---

# F8 — `scripts/windows/train_sft.bat` và `chat.bat`

## `train_sft.bat`

Sinh dữ liệu nếu chưa có → SFT → đo độ chính xác. Đổi config bằng biến môi trường `SFT_CONFIG`.

```bat
@echo off
rem ================================================================
rem  SFT: sinh du lieu the gioi do choi (neu chua co) roi train.
rem
rem  Cach dung - chay tu bat ky thu muc nao:
rem    scripts\windows\train_sft.bat
rem
rem  Doi config: sua dong SET ben duoi, hoac dat bien moi truong SFT_CONFIG.
rem  Ma thoat: 0 = xong, 1 = thieu venv, 3 = thieu config,
rem            6 = chua co checkpoint pretrain
rem ================================================================
setlocal

chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"

pushd "%~dp0..\.."

set "PYTHON=.venv\Scripts\python.exe"
set "PRETRAIN=checkpoints\pretrain.pt"
if not defined SFT_CONFIG set "SFT_CONFIG=configs\sft_tiny.json"
set "EXIT_CODE=0"

if not exist "%PYTHON%" (
    echo [LOI] Khong tim thay %PYTHON%
    set "EXIT_CODE=1"
    goto :finish
)

if not exist "%SFT_CONFIG%" (
    echo [LOI] Khong tim thay %SFT_CONFIG%
    set "EXIT_CODE=3"
    goto :finish
)

if not exist "%PRETRAIN%" (
    echo [LOI] Chua co %PRETRAIN%
    echo       Pretrain truoc: scripts\windows\train_pretrain.bat
    set "EXIT_CODE=6"
    goto :finish
)

if not exist "dataset\sft\train.jsonl" (
    echo === Sinh du lieu the gioi do choi ===
    "%PYTHON%" -m scripts.prepare_sft
    if errorlevel 1 (
        set "EXIT_CODE=%ERRORLEVEL%"
        goto :finish
    )
)

echo === SFT voi %SFT_CONFIG% ===
"%PYTHON%" -m trainer.train_sft --config "%SFT_CONFIG%"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto :finish

echo.
echo === Do do chinh xac ===
"%PYTHON%" -m inference.chat --eval
set "EXIT_CODE=%ERRORLEVEL%"

:finish
popd
endlocal & exit /b %EXIT_CODE%
```

## `chat.bat`

```bat
@echo off
rem ================================================================
rem  Hoi - dap voi model da SFT.
rem
rem  Cach dung:
rem    scripts\windows\chat.bat          chat (Enter trong de thoat)
rem    scripts\windows\chat.bat eval     do do chinh xac tren train/eval
rem
rem  Ma thoat: 0 = xong, 1 = thieu venv, 6 = chua co checkpoint SFT
rem ================================================================
setlocal

chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"

pushd "%~dp0..\.."

set "PYTHON=.venv\Scripts\python.exe"
set "CHECKPOINT=checkpoints\sft.pt"
set "EXIT_CODE=0"

if not exist "%PYTHON%" (
    echo [LOI] Khong tim thay %PYTHON%
    set "EXIT_CODE=1"
    goto :finish
)

if not exist "%CHECKPOINT%" (
    echo [LOI] Chua co %CHECKPOINT%
    echo       Train truoc: scripts\windows\train_sft.bat
    set "EXIT_CODE=6"
    goto :finish
)

if /i "%~1"=="eval" (
    "%PYTHON%" -m inference.chat --checkpoint "%CHECKPOINT%" --eval
) else (
    "%PYTHON%" -m inference.chat --checkpoint "%CHECKPOINT%"
)
set "EXIT_CODE=%ERRORLEVEL%"

:finish
popd
endlocal & exit /b %EXIT_CODE%
```

**Claude đã chạy thử cả hai:** `train_sft.bat` (sinh dữ liệu, train, đo độ chính xác, mã thoát 0), `chat.bat eval`, và `chat.bat` với câu hỏi đưa qua stdin → `Model: 11` sau 30 bước train thử.

Nhớ lưu 2 file với xuống dòng kiểu Windows (CRLF).

---

---

## Sau chặng 4

Model giờ **trả lời được câu hỏi** trong thế giới đồ chơi. Chặng tiếp theo trong lộ trình:

- **Chặng 5 — LoRA:** thay vì train cả 2,4 triệu tham số, chỉ train vài nghìn tham số gắn thêm. Tiêu chí xong: đầu ra ban đầu và sau khi gộp không đổi.
- **Chặng 6 — DPO:** dạy model thích câu trả lời đúng hơn câu sai. Thế giới đồ chơi sinh cặp *(đáp án đúng, đáp án sai)* tự động — đây là lúc nó bắt đầu được dùng lại.

Trước khi sang: ghi kết quả SFT (97,8% / 80,1%) vào CLAUDE.md, và cân nhắc tách `trainer/loop.py` như ghi chú ở F6.

---

---

## Lỗi thường gặp

| Thông báo lỗi                                                      | Ở đâu   | Nguyên nhân → Sửa                                                                                      |
| --------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'tokenizers'`                 | A2, C1     | Chưa`pip install tokenizers`                                                                            |
| `ModuleNotFoundError: No module named 'tokenizer'`                  | A1 test    | Thiếu`tokenizer/__init__.py`, hoặc không chạy từ thư mục gốc                                     |
| `ModuleNotFoundError: No module named 'core'`                       | test, C1   | Chạy`pytest` / `python scripts/...` thay vì `python -m ...`; hoặc không đứng ở thư mục gốc |
| `Exception: No such file or directory` khi `Tokenizer.from_file`  | C1         | Chưa train tokenizer (A2 bước cuối)                                                                    |
| Test`repeated_character_becomes_one_token` fail                     | A1         | `merge` gộp chồng lấn (quên `i += 2`)                                                              |
| decode ra chữ khác câu gốc                                        | A1         | `encode` không gộp theo thứ tự học (dùng `max` thay `min` theo id)                             |
| `ValueError: input_ids phải nằm trong [0, ...]`                   | B1, C1     | `vocab_size` trong `ModelConfig` khác `tokenizer.get_vocab_size()`                                  |
| `ValueError: input_ids phải là torch.long`                        | B1, C1     | Tạo tensor không ghi`dtype=torch.long`                                                                 |
| `cannot assign module before Module.__init__() call`                | mọi class | Quên`super().__init__()` ở dòng đầu `__init__`                                                    |
| `The size of tensor a (16) must match the size of tensor b (5)`     | B2         | Quên`keepdim=True`                                                                                      |
| Test`each_token_is_normalized_independently` fail                   | B2         | Quên`dim=-1`                                                                                            |
| Test`large_fp16_input_does_not_overflow` fail                       | B2         | Quên`x.float()`                                                                                         |
| `shape '[2, 5, 4, 16]' is invalid for input of size ...`            | B4         | `view` sai thứ tự: phải `(batch, seq_len, n_heads, head_dim)`                                       |
| `view size is not compatible with input tensor's size and stride`   | B4         | `view` sau `transpose` → đổi thành `reshape`                                                     |
| `The size of tensor a (32) must match ... (5)` trong `apply_rope` | B4, B7     | Truyền cả bảng cos/sin thay vì cắt`[:seq_len]`                                                      |
| Test`future_token_does_not_change_past_outputs` fail                | B4         | Mask sai: phải`torch.triu(..., diagonal=1)`                                                             |
| Kết quả có`nan`                                                  | B4         | `diagonal=0` → hàng 0 bị mask toàn bộ                                                               |
| `loss = nan`                                                        | C1         | Câu chỉ có 1 token → không có cặp để tính loss                                                   |
| Test`initial_loss_is_close_to_log_vocab_size` fail (loss lớn)      | B7         | Quên`self.apply(_init_weights)`                                                                         |
| Test`embedding_and_lm_head_share_weights` fail                      | B7         | Quên dòng tie, hoặc tạo bản sao`nn.Parameter(embed.weight.clone())` thay vì dùng chung            |
| `fixture 'tiny_config' not found`                                   | test       | Chưa tạo`tests/conftest.py`                                                                            |
| `fixture 'make_train_config' not found`                             | test D     | Chưa tạo`tests/trainer/conftest.py`                                                                    |
| `ValueError: file chỉ có N token`                                 | D2         | Chạy D1 trước, hoặc`seq_len` lớn hơn cả file                                                      |
| `WeightsUnpickler error` khi `torch.load`                         | D6         | Thiếu`weights_only=False` (payload có trạng thái RNG)                                                |
| Resume xong trọng số khác lần chạy liền mạch                   | D9         | Quên bỏ qua`start_step * accum_steps` batch                                                            |
| `loss` không giảm sau vài trăm step                             | D9         | LR quá nhỏ hoặc quá lớn, hoặc`.bin` sai tokenizer — đối chiếu `meta.json`                    |
| `CUDA out of memory`                                                | D9         | Giảm`batch_size` hoặc `seq_len`, tăng `accum_steps` để giữ batch hiệu dụng                   |
| Sinh chữ có cache khác không cache                                | E1         | Quên lệch RoPE (`rope_cos[past_len:...]`) hoặc lệch mask (`diagonal=1 + past_len`)                 |
| `TypeError: cannot unpack non-sequence Tensor`                      | E1         | Gọi model có`use_cache=True` nhưng forward chưa trả tuple                                           |
| Chữ sinh ra lặp đi lặp lại một cụm                             | E2         | `temperature=0` hay lặp; dùng `--temperature 0.8 --top-k 40`                                         |
| Chữ sinh ra loạn, toàn ký tự lạ                                 | E2         | `temperature` quá cao, hoặc nạp nhầm tokenizer khác lúc train                                      |
| `JSONDecodeError: Unexpected UTF-8 BOM`                             | F5, D8     | File JSON lưu kèm BOM — đọc bằng`encoding="utf-8-sig"`                                             |
| `ValueError: vocab_size của checkpoint ... khác tokenizer`        | F6         | `init_checkpoint` train bằng tokenizer khác                                                            |
| Loss SFT không giảm, độ chính xác 0%                            | F2, F4     | Nhãn toàn -100 (quên học phần assistant) hoặc quên dịch nhãn                                      |
| Model trả lời xong viết tiếp mãi                                 | F2         | Quên đưa`<\|im_end\|>` vào phần có nhãn                                                             |
| Train 100% nhưng eval rất thấp                                     | F1         | Học vẹt — cần thêm dữ liệu hoặc giảm số bước                                                   |
