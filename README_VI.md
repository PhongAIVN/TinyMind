# TinyMind

[English](README.md) | **Tiếng Việt**

Mã nguồn trên GitHub: [github.com/PhongAIVN/TinyMind](https://github.com/PhongAIVN/TinyMind)

Trọng số trên Hugging Face: [huggingface.co/PhongAIVN/TinyMind-57M-vi](https://huggingface.co/PhongAIVN/TinyMind-57M-vi)

Mô hình ngôn ngữ nhỏ (SLM) bằng PyTorch: từ tokenizer, kiến trúc Transformer, đến vòng lặp pretrain. Mục tiêu là đi qua toàn bộ chuỗi huấn luyện của một LLM hiện đại:

**Pretrain → SFT → LoRA → DPO → Tool Use → Tư duy thích ứng → Chưng cất → GRPO/CISPO → PPO → Agentic RL**

## Quá trình

| # | Chặng              | Trạng thái                                                                                          |
| - | ------------------- | ----------------------------------------------------------------------------------------------------- |
| 0 | Pretrain            | Xong: model 57M, 1,15 tỷ token, val loss **2,437**                                             |
| 1 | SFT                 | Xong: 155K hội thoại, val loss **1,555**                                                      |
| 2 | LoRA                | Xong: phép tính đồ chơi đúng từ **6,3%** lên **23,7%**, chỉ học 0,34% tham số |
| 3 | DPO                 | Xong: chosen thắng 72%, độ chính xác giữ nguyên 22% nhờ loss SFT trên chosen (kiểu RPO)     |
| 4 | Tool Use            | Xong: gọi máy tính đúng **100%**                                                           |
| 5 | Tư duy thích ứng | Xong: cùng model, tắt tư duy **38,3%**, bật tư duy **67,3%**                         |
| 6 | Chưng cất         | Xong: student 13,3M đạt **58,0%** (bằng 86% teacher 57M)                                     |
| 7 | GRPO / CISPO        | Xong: 67,7% → **70,0%** / **70,7%**                                                      |
| 8 | PPO                 | Xong: 67,7% → **68,7%**                                                                        |
| 9 | Agentic RL          | Chạy xong nhưng không học được gì: model đã đúng 100% từ đầu, cần bài khó hơn      |

## Model 57M

`configs/model_50m.json`

| Tham số                 | Giá trị            |
| ------------------------ | -------------------- |
| `vocab_size`           | 32.000               |
| `d_model`              | 512                  |
| `n_layers`             | 12                   |
| `n_heads`              | 8 (head_dim 64)      |
| `ffn_hidden`           | 1.536                |
| `max_seq_len`          | 512                  |
| **Tổng tham số** | **57.291.264** |

Số tham số khớp công thức, và có test kiểm tra:

```
P = V·d + L·(4d² + 3·d·f + 2d) + d
  = 32000·512 + 12·(4·512² + 3·512·1536 + 2·512) + 512
```

Trong đó: embedding `V·d` (tied nên không tính `lm_head` thêm lần nữa), attention `4d²` (Q, K, V, O), SwiGLU `3·d·f` (gate, up, down), mỗi lớp có 2 RMSNorm (`2d`), và một RMSNorm cuối (`d`). RoPE không có tham số.

---

## Dữ liệu pretrain

Khoảng **1,15 tỷ token train** + 5,8 triệu token eval, tức khoảng 20 token cho mỗi tham số (mức tối ưu theo Chinchilla). Mỗi nguồn để riêng 1/200 số văn bản làm tập eval.

| Nguồn                                                                                                                                              | Nội dung                                    |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| [epfml/FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ) (`vie_Latn`)                                                                | Văn bản web tiếng Việt chất lượng cao |
| [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (`20231101.vi`)                                                         | Kiến thức nền                             |
| [vilm/OpenOrca-Viet](https://huggingface.co/datasets/vilm/OpenOrca-Viet)                                                                             | Hỏi đáp, giải thích                     |
| [5CD-AI/Vietnamese-alpaca-gpt4-gg-translated](https://huggingface.co/datasets/5CD-AI/Vietnamese-alpaca-gpt4-gg-translated)                           | Hỏi đáp cơ bản                          |
| [bkai-foundation-models/vi-alpaca](https://huggingface.co/datasets/bkai-foundation-models/vi-alpaca)                                                 | Hỏi đáp, chỉ dẫn                        |
| [5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated](https://huggingface.co/datasets/5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated) | Toán có lời giải                         |
| [taidng/UIT-ViQuAD2.0](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0)                                                                         | Đọc hiểu                                  |
| [PeterPaker123/vietnamese-code-dataset](https://huggingface.co/datasets/PeterPaker123/vietnamese-code-dataset)                                       | Đề bài tiếng Việt, lời giải code      |
| [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)                         | Chỉ dẫn + code Python                      |
| [OpenCoder-LLM/opc-annealing-corpus](https://huggingface.co/datasets/OpenCoder-LLM/opc-annealing-corpus) (`synthetic_qa`, chỉ lấy Python)        | Code chất lượng cao, tối đa 600MB       |

Dữ liệu hỏi đáp được ghép thành văn bản thường theo khuôn `Câu hỏi: … / Trả lời: …`.

## Dữ liệu SFT

Mỗi nguồn lấy tối đa 20.000 hội thoại, tổng **155.056 hội thoại train** + 778 eval (`scripts/build_sft_data.py`). Khuôn hội thoại:

```
<|bos|><|im_start|>user
{câu hỏi}<|im_end|>
<|im_start|>assistant
{trả lời}<|im_end|>
```

- **Hỏi đáp một lượt**: 7 nguồn hỏi đáp ở trên (OpenOrca, 2 bộ alpaca, MetaMath, ViQuAD, 2 bộ code), đổi sang dạng hội thoại.
- **Hội thoại nhiều lượt**, chỉ dùng cho SFT:

| Nguồn                                                                                                                    | Nội dung                    |
| ------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| [5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca](https://huggingface.co/datasets/5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca)       | Hội thoại nhiều lượt    |
| [5CD-AI/Vietnamese-Ecommerce-Multi-turn-Chat](https://huggingface.co/datasets/5CD-AI/Vietnamese-Ecommerce-Multi-turn-Chat) | Tư vấn, bán hàng         |
| [nguyenthanhdo/ultrachat-75k-vi](https://huggingface.co/datasets/nguyenthanhdo/ultrachat-75k-vi)                           | UltraChat dịch tiếng Việt |

Hội thoại dài hơn 512 token bị cắt đuôi; hội thoại mà phần trả lời bị cắt mất hết thì bỏ (5.104 cuộc, khoảng 3%).

---

## Kết quả

**Model tí hon** (2,4M tham số, tokenizer cũ 16.000 token, 8,2M token, CPU):

|                  | Val loss        | Perplexity    |
| ---------------- | --------------- | ------------- |
| Chưa train      | 9,705           | 16.408        |
| Sau 2.000 bước | **5,030** | **153** |

**Model 57M, pretrain** (RTX 5070 12GB, bf16, SDPA, 65.536 token/bước, 17.584 bước):

- Tốc độ khoảng **73.000 token/giây**, dùng 7,2GB VRAM, khoảng 4,5 giờ, chi phí thuê khoảng 50.000đ.
- Val loss: 10,37 lúc khởi tạo (= ln 32000) → 3,86 (bước 750) → 2,67 (6.750) → 2,49 (13.250) → **2,437** (cuối, perplexity khoảng 11,4). Log đầy đủ: `logs/pretrain_log.txt`.
- Model viết tiếng Việt đúng ngữ pháp và viết được code Python đúng cú pháp, nhưng kiến thức còn mỏng, hay lặp lại và không biết khi nào dừng.

**Model 57M, SFT** (khởi đầu từ bản pretrain, 2.000 bước × 128 hội thoại, khoảng 1,65 epoch, khoảng 30 phút):

- Val loss **1,555** (chỉ tính trên phần trả lời của assistant). Trên 4 hội thoại mẫu, loss phần trả lời giảm từ 3,44 (bản pretrain) xuống 1,24.
- Đã biết trả lời theo khuôn và **tự dừng** bằng `<|im_end|>`.

| Câu hỏi                                       | Trả lời (chọn token xác suất cao nhất)                                     |
| ----------------------------------------------- | -------------------------------------------------------------------------------- |
| Thủ đô của Việt Nam là gì?               | Thủ đô của Việt Nam là Hà Nội.                                           |
| Kể tên 3 món ăn nổi tiếng của Việt Nam. | 1. Phở … 2. Bánh mì … 3. Bánh xèo … (đúng tên món, mô tả còn sai) |
| Viết hàm Python tính giai thừa.             | Hàm đệ quy đúng, nhưng lặp lại nhiều lần, không dừng                 |
| Thủ đô của Trung Quốc là gì?             | "Thành Đô" (sai, nhưng nghe có lý)                                         |

Giới hạn: 57M tham số nhớ được ít kiến thức nên hay bịa; không hiểu tiếng Việt không dấu và tiếng Anh (gần như không có trong dữ liệu).

**Model 57M, LoRA** (từ bản SFT, r = 8, alpha = 16, trên `q_proj` + `v_proj` của 12 lớp, 600 bước × 32 bài, CPU khoảng 20 phút):

- Học 196.608 tham số (0,34%); val loss 1,22 → **0,516**. Log: `logs/lora_log.txt`.
- Chấm 300 bài eval (không trùng bài train), sinh tham lam:

| Phép tính                             | SFT           | SFT + LoRA               |
| --------------------------------------- | ------------- | ------------------------ |
| Tất cả                                | 19/300 = 6,3% | **71/300 = 23,7%** |
| Cộng (0–99)                           |               | 6/112 = 5%               |
| Trừ (0–99)                            |               | 7/114 = 6%               |
| Nhân (0–12)                           |               | **58/74 = 78%**    |
| Trả lời đúng khuôn `a op b = c.` |               | **300/300**        |

- LoRA dạy được **khuôn trả lời** gần như tuyệt đối, và phép nhân trong bảng cửu chương (ít cặp số, có sẵn trong dữ liệu pretrain).
- Cộng/trừ hai chữ số gần như không học được: tokenizer gộp cả số `29`, `30` thành **một token**, nên model phải học thuộc khoảng 10.000 cặp số như tra bảng chứ không tính từng chữ số. Đây là lý do cho chặng Tool Use (gọi máy tính) và RL sau này.
- Có dấu hiệu quên nhẹ: hỏi "Kể tên 3 món ăn…" thì chỉ trả lời một câu cụt, vì LoRA chỉ thấy câu trả lời rất ngắn. Hỏi thủ đô vẫn đúng.

**Model 57M, DPO** (từ bản LoRA đã gộp, β = 0,1, lr 1e-5, 400 bước × 16 cặp, train cả model, CPU khoảng 25 phút):

- Dữ liệu: 8.000 cặp phép tính; chosen = đáp số đúng, rejected = đáp số sai kiểu model hay mắc (chép lại một vế, lệch vài đơn vị). Eval là đúng 300 bài của bộ đồ chơi.
- Loss bắt đầu ở **0,677 ≈ ln 2** đúng như lý thuyết, val loss 0,693 → **0,554**. Log: `logs/dpo_log.txt`.
- Trên 300 cặp eval, **chosen thắng 72%**, reward chênh trung bình 0,93.
- Nhưng **độ chính xác khi tự sinh giảm 23,7% → 11,0%** (nhân 78% → 30%). Nguyên nhân: DPO kéo **cả hai** câu xuống, chỉ là rejected xuống nhanh hơn:

|                 | Trước (ref) | Sau DPO |
| --------------- | ------------- | ------- |
| log p(chosen)   | -3,79         | -6,42   |
| log p(rejected) | -4,67         | -12,35  |

  Xác suất bị lấy khỏi cả hai câu dồn sang token khác: model trả lời "1 + 29 = 29 **tuổi**". Đây là lỗi đã biết của DPO khi chosen và rejected gần giống nhau (ở đây chỉ khác đúng một token). Hướng sửa: cộng thêm loss SFT trên câu chosen để giữ xác suất của nó (kiểu RPO), tăng β, hoặc lấy rejected từ chính câu trả lời của model.

## Cấu trúc thư mục

```
core/        Model: config, embedding, rmsnorm, rope, attention, feedforward, block, model, lora, value_model (critic)
tokenizer/   BPE: bpe_toy.py (thuật toán thuần), train_tokenizer.py, tokenizer.json
trainer/     Vòng lặp train dùng chung (nhận hàm loss tùy chọn), dataset, loss, optimizer, checkpoint, RL (rl.py),
             train_pretrain / sft / lora / dpo / distill / grpo / ppo.py
inference/   Sinh chữ (temperature, top-k, KV cache, --chat), tool + vòng lặp agent, chấm thế giới đồ chơi
scripts/     Chuẩn bị dữ liệu: build_pretrain_corpus, prepare_pretrain, check_pretrain_data, build_sft_data, build_toy_*; train_all.py, eval_all.py
scripts/windows/  Script .bat / .ps1 cho Windows và máy GPU thuê
configs/     Config model (kiến trúc) và config train (từng chặng), tách riêng
tests/       Test cho từng thành phần
docs/        Sơ đồ kiến trúc (draw.io)
logs/        Log train
```

## Chạy thử

Cần Python 3.10+.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
```

Bản cu128 cần cho GPU RTX 50xx, và vẫn chạy được trên 30xx/40xx. Máy chỉ có CPU thì bỏ `--index-url ...`.

**Chạy test**

```bash
.venv/Scripts/python -m pytest tests -q
```

**Train trọn chuỗi bằng một lệnh** (máy GPU Windows, từ thư mục gốc project)

```bat
scripts\windows\train_all.bat
```

Lệnh này cài môi trường và tải dữ liệu thô (nếu chưa có), rồi chạy lần lượt: dữ liệu pretrain + tokenizer → pretrain → dữ liệu SFT → SFT → dữ liệu đồ chơi → LoRA → DPO → Tool Use → Thinking → Chưng cất → GRPO → CISPO → PPO → Agentic RL → bảng kết quả `logs/eval_results.md`.

- Mỗi bước ghi log riêng vào `logs/<bước>.log` và đánh dấu xong ở `checkpoints/50m/done/`. Bị ngắt thì chạy lại đúng lệnh cũ: bước đã xong được bỏ qua, bước train đang dở tự `--resume`.
- `train_all.bat --list` xem trạng thái; `--from dpo` chạy từ một bước; `--only sft` chạy riêng; `--dry-run` chỉ in lệnh.
- Đã có `pretrain.pt` thì chép vào `checkpoints\50m\` rồi chạy `scripts\windows\train_from_pretrain.bat`: chỉ tải dữ liệu SFT (khoảng 1GB), kiểm tra checkpoint khớp tokenizer, rồi chạy từ SFT tới eval.

**Chạy từng bước bằng tay** (PowerShell)

```powershell
# venv + torch CUDA + tải khoảng 14,6GB dữ liệu thô vào dataset\raw\
powershell -ExecutionPolicy Bypass -File scripts\windows\setup_gpu_vm.ps1

# dữ liệu thô -> jsonl -> tokenizer -> train.bin / eval.bin -> kiểm tra
powershell -ExecutionPolicy Bypass -File scripts\windows\build_pretrain_data.ps1

# pretrain (thêm --resume để chạy tiếp từ checkpoint)
.venv\Scripts\python.exe -u -m trainer.train_pretrain --model-config configs/model_50m.json --train-config configs/pretrain_50m.json
```

Với GPU 12GB, giữ `batch_size 16` × `accum_steps 8`. GPU từ 16GB trở lên có thể thử `32 × 4`; số token mỗi bước không đổi nên không phải sửa `max_steps`. Trên Windows, khi hết VRAM, driver không báo lỗi mà lấy RAM hệ thống bù vào, làm train chậm đi khoảng 3 lần. Vì vậy hãy để ý số VRAM trong log.

**SFT** (cần `checkpoints/50m/pretrain.pt` và dữ liệu thô ở `dataset\raw\`)

```powershell
# hỏi đáp + hội thoại -> dataset\sft\train.jsonl, eval.jsonl
.venv\Scripts\python.exe -m scripts.build_sft_data --skip-missing

# SFT, nạp trọng số từ pretrain.pt (init_from trong configs/sft_50m.json)
.venv\Scripts\python.exe -u -m trainer.train_sft --model-config configs/model_50m.json --train-config configs/sft_50m.json
```

**LoRA trên thế giới đồ chơi** (cần `checkpoints/50m/sft.pt`; chạy được trên CPU, khoảng 1.000 token/giây)

```powershell
# 20.000 bài train + 300 bài eval, không trùng nhau
.venv\Scripts\python.exe -m scripts.build_toy_math

# độ chính xác trước khi train
.venv\Scripts\python.exe -m inference.eval_toy_math --checkpoint checkpoints/50m/sft.pt

# train LoRA (trên CPU: sửa "device": "cpu", "dtype": "fp32" trong configs/lora_50m.json), tự gộp ra lora_merged.pt
.venv\Scripts\python.exe -u -m trainer.train_lora --rank 8 --alpha 16

# độ chính xác sau khi train
.venv\Scripts\python.exe -m inference.eval_toy_math --checkpoint checkpoints/50m/lora_merged.pt
```

**DPO trên thế giới đồ chơi** (cần `checkpoints/50m/lora_merged.pt`)

```powershell
.venv\Scripts\python.exe -m scripts.build_toy_dpo
.venv\Scripts\python.exe -u -m trainer.train_dpo --beta 0.1
.venv\Scripts\python.exe -m inference.eval_toy_math --checkpoint checkpoints/50m/dpo.pt
```

**Dùng trọng số có sẵn, không cần train**

[PhongAIVN/TinyMind-57M-vi](https://huggingface.co/PhongAIVN/TinyMind-57M-vi) có 7 checkpoint dạng safetensors: `pretrain`, `sft`, `lora_merged`, `tool`, `think`, `cispo`, `student`. Hỏi đáp chung dùng `sft`; toán đồ chơi dùng `cispo` (bật tư duy) hoặc `tool`. Cách nạp xem model card.

```bash
.venv/Scripts/hf download PhongAIVN/TinyMind-57M-vi --include "sft/*" --include "tokenizer.json" --local-dir hf
```

**Sinh chữ**

```bash
# model pretrain: viết tiếp văn bản
.venv/Scripts/python -m inference.generate --checkpoint checkpoints/50m/pretrain.pt --prompt "Hà Nội là"

# model SFT: hỏi đáp theo khuôn hội thoại
.venv/Scripts/python -m inference.generate --checkpoint checkpoints/50m/sft.pt --chat --prompt "Thủ đô của Việt Nam là gì?"
```

`tokenizer/tokenizer.json` phải đúng là file đã dùng để tokenize dữ liệu pretrain. Dùng nhầm tokenizer khác (dù cùng số vocab) thì model ra chữ vô nghĩa.
