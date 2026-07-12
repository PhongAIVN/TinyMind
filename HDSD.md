# Hướng Dẫn Sử Dụng — Train Small Language Model (SLM)

Tài liệu này hướng dẫn từng bước để train một mô hình ngôn ngữ nhỏ (~57M tham số) từ đầu bằng project này: chuẩn bị dữ liệu → train tokenizer → đóng gói dữ liệu → train model → theo dõi → sinh văn bản.

---

## 1. Yêu cầu môi trường

- **Python** 3.10 trở lên
- **GPU NVIDIA** có CUDA (khuyến nghị ≥ 8GB VRAM). Không có GPU vẫn train được trên CPU nhưng rất chậm.
- Cài thư viện:

```bash
pip install -r requirements.txt
```

Kiểm tra GPU đã nhận:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Kiểm tra code hoạt động đúng (nên chạy trước khi train):

```bash
python -m pytest slm/tests -q
```

Tất cả test phải pass (68 tests).

---

## 2. Chuẩn bị dữ liệu văn bản

### 2.1. Định dạng dữ liệu

- File văn bản thuần `.txt`, encoding **UTF-8**.
- Mỗi file được coi là **một tài liệu** (bài báo, chương sách, trang web...).
- Nội dung tiếng Việt và tiếng Anh trộn lẫn đều được.

Đặt dữ liệu vào thư mục `data/raw/`, ví dụ:

```
data/raw/
    train_wiki_vi.txt
    train_news_vi.txt
    train_books_en.txt
    val_wiki_vi.txt
```

### 2.2. Cần bao nhiêu dữ liệu?

| Mục đích | Số token khuyến nghị | Tương đương văn bản |
|---|---|---|
| Chạy thử pipeline | ~1 triệu | vài chục MB |
| Model bắt đầu ra câu có nghĩa | ~100 triệu | ~500 MB |
| Chất lượng tốt cho 57M params | 1–2 tỷ | vài GB |

Quy tắc kinh nghiệm (Chinchilla): số token ≈ 20 × số tham số → với model 57M thì ~1.1 tỷ token là điểm tối ưu.

**Nguồn dữ liệu gợi ý:** Wikipedia tiếng Việt/Anh (dump), OSCAR, CC100-vi, sách/truyện public domain, tin tức đã crawl hợp pháp.

### 2.3. Tách train / validation

Để riêng khoảng **1–5%** dữ liệu làm validation (file khác nhau, KHÔNG trùng nội dung với train), ví dụ đặt tên `val_*.txt`.

---

## 3. Train tokenizer (SentencePiece)

Tokenizer học cách cắt văn bản thành các mảnh nhỏ (subword). Chỉ cần train **một lần** trên dữ liệu đại diện:

```bash
python -m slm.tokenizer.train_tokenizer ^
    --input data/raw/train_wiki_vi.txt data/raw/train_books_en.txt ^
    --output-prefix tokenizer/slm ^
    --vocab-size 32000 ^
    --model-type unigram
```

> Lưu ý: trên PowerShell dùng dấu **`` ` ``** hoặc viết cả lệnh trên 1 dòng; ký tự `^` là cho CMD. Trên Linux/macOS dùng `\`.

Kết quả: `tokenizer/slm.model` và `tokenizer/slm.vocab`.

**Các lựa chọn:**
- `--vocab-size 32000` — phải khớp với `vocab_size` trong config model.
- `--model-type unigram` (mặc định, khuyến nghị) hoặc `bpe`.
- Dữ liệu train tokenizer không cần toàn bộ corpus — vài trăm MB đại diện là đủ.

Kiểm tra nhanh tokenizer:

```bash
python -c "from slm.tokenizer.tokenizer import Tokenizer; t = Tokenizer('tokenizer/slm.model'); ids = t.encode('Xin chào Việt Nam'); print(ids); print(t.decode(ids))"
```

---

## 4. Đóng gói dữ liệu thành file nhị phân

Bước này tokenize toàn bộ văn bản và ghi thành file `.bin` (uint16, đọc bằng memmap nên không tốn RAM khi train):

```bash
# Dữ liệu train
python -m slm.dataset.preprocess ^
    --input data/raw/train_wiki_vi.txt data/raw/train_news_vi.txt data/raw/train_books_en.txt ^
    --tokenizer tokenizer/slm.model ^
    --output data/train.bin

# Dữ liệu validation
python -m slm.dataset.preprocess ^
    --input data/raw/val_wiki_vi.txt ^
    --tokenizer tokenizer/slm.model ^
    --output data/val.bin
```

Lệnh sẽ in ra tổng số token đã ghi — **ghi nhớ con số này** để tính `max_steps` ở bước sau.

Quá trình xử lý tự động: chuẩn hoá Unicode NFC (quan trọng với dấu tiếng Việt), chuẩn hoá xuống dòng, thêm `<eos>` cuối mỗi tài liệu, nối tất cả thành một dòng token liên tục.

---

## 5. Cấu hình training

Mở file `slm/configs/slm_base.json`. Các tham số quan trọng nhất:

### 5.1. Phần `model`

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `vocab_size` | 32000 | Phải khớp với tokenizer |
| `num_layers` | 12 | Số tầng Transformer |
| `hidden_size` | 512 | Kích thước ẩn (= num_heads × head_dim) |
| `context_length` | 2048 | Độ dài ngữ cảnh tối đa |

Thường **không cần sửa** phần này, trừ khi muốn model to/nhỏ hơn.

### 5.2. Phần `train`

| Tham số | Mặc định | Ý nghĩa & cách chỉnh |
|---|---|---|
| `train_data_path` | `data/train.bin` | Đường dẫn file token train |
| `val_data_path` | `data/val.bin` | Đường dẫn file token validation |
| `sequence_length` | 2048 | Độ dài mỗi mẫu. Giảm xuống 1024/512 nếu thiếu VRAM |
| `batch_size` | 8 | Số mẫu mỗi lần forward. **Giảm nếu bị out-of-memory** |
| `gradient_accumulation_steps` | 4 | Gộp gradient nhiều batch nhỏ. Batch hiệu dụng = batch_size × số này |
| `max_steps` | 10000 | Tổng số bước optimizer |
| `learning_rate` | 3e-4 | Tốc độ học đỉnh |
| `warmup_steps` | 500 | Số bước khởi động LR (~5% max_steps) |
| `precision` | `bf16` | `bf16` (GPU Ampere trở lên), `fp16` (GPU cũ), `fp32` (CPU/an toàn nhất) |
| `eval_interval` | 500 | Bao nhiêu bước thì chạy validation một lần |
| `checkpoint_interval` | 1000 | Bao nhiêu bước thì lưu checkpoint |
| `checkpoint_dir` | `checkpoints` | Thư mục lưu checkpoint |

### 5.3. Tính `max_steps` phù hợp

```
token mỗi bước = batch_size × gradient_accumulation_steps × sequence_length
               = 8 × 4 × 2048 = 65,536 token/bước

max_steps ≈ (tổng token × số epoch mong muốn) / token mỗi bước
```

Ví dụ: corpus 200 triệu token, muốn học ~3 epoch → `max_steps ≈ 200e6 × 3 / 65536 ≈ 9,200`.

### 5.4. Gợi ý cấu hình theo VRAM

| VRAM | batch_size | sequence_length | grad_accum | precision |
|---|---|---|---|---|
| 6 GB | 2 | 1024 | 16 | bf16/fp16 |
| 8 GB | 4 | 2048 | 8 | bf16/fp16 |
| 12 GB | 8 | 2048 | 4 | bf16 |
| 24 GB | 16 | 2048 | 2 | bf16 |

> Nguyên tắc: khi giảm `batch_size` thì tăng `gradient_accumulation_steps` tương ứng để giữ nguyên batch hiệu dụng (~32 mẫu).

Có thể tạo file config riêng (ví dụ `slm/configs/my_run.json`) thay vì sửa file gốc.

---

## 6. Chạy training

```bash
python -m slm.train.trainer --config slm/configs/slm_base.json
```

Màn hình sẽ hiện:

```
Model parameters: 57.3M | device: cuda | precision: bf16
step     10 | loss 9.8123 | lr 6.00e-06 | grad_norm 2.31 | 45,000 tok/s | mem 5.42GiB
step     20 | loss 9.5214 | lr 1.20e-05 | grad_norm 1.87 | 46,200 tok/s | mem 5.42GiB
...
step    500 | val_loss 6.8412
Checkpoint saved: checkpoints/step_1000.pt
```

### 6.1. Đọc hiểu log

- **loss** — mục tiêu chính, phải **giảm dần**. Bắt đầu ≈ `ln(32000) ≈ 10.4` (đoán ngẫu nhiên). Xuống ~4–5 là model bắt đầu ra câu tạm được; ~3–3.5 là khá tốt với model cỡ này.
- **val_loss** — loss trên dữ liệu chưa từng thấy. Nếu train loss giảm mà val_loss tăng → **overfitting** (dữ liệu quá ít, dừng sớm hoặc thêm dữ liệu).
- **lr** — tự tăng trong warmup rồi giảm dần theo cosine. Không cần can thiệp.
- **grad_norm** — bình thường dao động 0.1–2.0. Nếu liên tục hàng chục/trăm hoặc thành `nan` → giảm `learning_rate`.
- **tok/s** — tốc độ. Dùng để ước lượng thời gian: `thời_gian = max_steps × token_mỗi_bước / tok_s`.
- **mem** — VRAM đỉnh. Nếu sát giới hạn GPU, giảm `batch_size`.

### 6.2. Train tiếp từ checkpoint (resume)

Bị ngắt điện, tắt máy, hoặc muốn train thêm — chạy lại với `--resume`:

```bash
python -m slm.train.trainer --config slm/configs/slm_base.json --resume checkpoints/step_5000.pt
```

Checkpoint lưu đầy đủ model + optimizer + scheduler + trạng thái random, nên train tiếp **giống hệt** như chưa từng dừng.

> Muốn train thêm sau khi đã đạt `max_steps`: tăng `max_steps` trong config rồi resume từ `final.pt`.

---

## 7. Sinh văn bản với model đã train

```bash
python -m slm.inference.generate ^
    --checkpoint checkpoints/final.pt ^
    --tokenizer tokenizer/slm.model ^
    --prompt "Việt Nam là" ^
    --max-new-tokens 200 ^
    --temperature 0.8 --top-k 40 --top-p 0.9 --repetition-penalty 1.1
```

**Chỉnh tham số sinh:**

| Tham số | Tác dụng |
|---|---|
| `--temperature 0` | Greedy — luôn chọn token tốt nhất, kết quả cố định |
| `--temperature 0.7–0.9` | Cân bằng giữa ổn định và đa dạng (khuyến nghị) |
| `--temperature > 1` | Sáng tạo hơn nhưng dễ lan man |
| `--top-k 40` | Chỉ chọn trong 40 token tốt nhất |
| `--top-p 0.9` | Chỉ chọn trong nhóm token chiếm 90% xác suất |
| `--repetition-penalty 1.1` | Giảm lặp từ (1.0 = tắt; đừng quá 1.3) |

---

## 8. Xử lý sự cố thường gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `CUDA out of memory` | Batch quá lớn | Giảm `batch_size` (và tăng `gradient_accumulation_steps`), hoặc giảm `sequence_length` |
| Loss thành `nan`/`inf` | LR quá cao, hoặc fp16 tràn số | Giảm `learning_rate` (thử 1e-4); nếu dùng `fp16` hãy đổi sang `bf16` hoặc `fp32` |
| Loss đứng yên ~10.4 | Model không học được | Kiểm tra dữ liệu `.bin` có đúng không (bước 4 in ra số token > 0), LR không quá nhỏ |
| Loss giảm rồi tăng vọt | LR cao ở giai đoạn giữa | Giảm `learning_rate`, tăng `warmup_steps` |
| val_loss tăng, train loss giảm | Overfitting | Thêm dữ liệu, hoặc dừng ở checkpoint có val_loss thấp nhất |
| `Vocabulary size too high` khi train tokenizer | Corpus quá nhỏ so với vocab | Thêm dữ liệu hoặc giảm `--vocab-size` |
| Train quá chậm trên CPU | Không có GPU | Giảm kích thước model để thử nghiệm (num_layers=4, hidden_size=256...), dùng `precision: fp32` |
| Text sinh ra vô nghĩa | Train chưa đủ | Bình thường nếu loss còn > 6; cần thêm bước train / thêm dữ liệu |
| Text sinh lặp vòng vô hạn | Thiếu penalty | Tăng `--repetition-penalty` lên 1.1–1.2, dùng `--top-p 0.9` |

---

## 9. Quy trình tóm tắt (checklist)

```
[ ] pip install -r requirements.txt
[ ] python -m pytest slm/tests -q          → 68 passed
[ ] Đặt file .txt (UTF-8) vào data/raw/, tách train/val
[ ] Train tokenizer  → tokenizer/slm.model
[ ] Preprocess train → data/train.bin (ghi lại số token)
[ ] Preprocess val   → data/val.bin
[ ] Chỉnh slm/configs/slm_base.json (batch_size theo VRAM, max_steps theo số token)
[ ] python -m slm.train.trainer --config slm/configs/slm_base.json
[ ] Theo dõi loss giảm, val_loss không tăng
[ ] Sinh thử văn bản với checkpoints/final.pt
```

Chúc train thành công! 🚀
