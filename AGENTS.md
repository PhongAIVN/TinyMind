# AGENTS.md

## Dự án là gì

Tự xây một **mô hình ngôn ngữ nhỏ (SLM) tiếng Việt từ con số 0**, đi qua **toàn bộ chuỗi huấn luyện của một LLM hiện đại**:

**Pretrain → SFT → LoRA → DPO → Tool Use → Tư duy thích ứng → Chưng cất → GRPO/CISPO → PPO → Agentic RL**

Mục tiêu số 1 là **hiểu**: người dùng tự viết từng dòng code. Mục tiêu số 2 là một project gọn gàng, có test, đủ chuyên nghiệp để đưa vào portfolio.

`minimind-master/` là **tài liệu tham khảo chỉ để đọc**: học ý tưởng, đối chiếu cách làm, **không** sao chép bố cục hay code.

---

## Vai trò

- **Người dùng** tự tổ chức thư mục, tự thiết kế API, **tự viết code và test**.
- **Codex** là người hướng dẫn và review:
  - Giải thích lý thuyết + phần code minimind tương ứng của từng chặng.
  - Review code người dùng viết: chỉ ra lỗi, giải thích vì sao, gợi ý hướng sửa.
  - **Chỉ viết code khi người dùng yêu cầu rõ ràng.**
  - **Không tự tạo thư mục, file hay cấu trúc project** khi chưa được yêu cầu.

---

## Trạng thái hiện tại (2026-09-16)

**Đã xong: chặng 0, 1, 2.** Toàn bộ test: 207 passed.

| Đường dẫn | Nội dung |
|---|---|
| `dataset/pretrain/train/` | `part_00.txt` … `part_08.txt` (~174MB) |
| `dataset/pretrain/eval/` | `part_09.txt` (~19MB) — tập validation |
| `dataset/pretrain/tokens/` | `train.bin` 35.264.352 token (70.5MB), `eval.bin` 3.969.973 token, kèm `*_meta.json` |
| `tokenizer/` | `bpe_toy.py` (BPE tự viết), `train_tokenizer.py`, `tokenizer.json` — **vocab 16000**, byte-level BPE, 19 token đặc biệt (id 0–18) |
| `core/` | config, embedding, rmsnorm, rope, attention, feedforward, block, model (`Model` = thân, `CausalLM` = thân + đầu) — **2.425.472 tham số** với config tí hon |
| `trainer/` | pretrain_dataset, losses, optim, lr_scheduler, checkpoint, metrics, train_config, train_pretrain |
| `scripts/` | `prepare_pretrain.py`, `text_to_logits.py`; `windows/*.bat`: test_full, prepare_pretrain, train_pretrain, continue-train_pretrain |
| `configs/` | `model_tiny.json`, `pretrain_tiny.json` |
| `checkpoints/` | `pretrain.pt` — 2000 bước, fp32 |
| `CODE_STEP.md` | Hướng dẫn code từng file (PHẦN A tokenizer → B model → C ghép → D pretrain) |
| `minimind-master/` | Repo tham khảo |

**Kết quả pretrain** (model 2,4M tham số, 8,19 triệu token, CPU, ~10.000 token/s):

| | Val loss | Perplexity |
|---|---|---|
| Chưa train | 9.705 | 16.408 |
| Sau 2000 bước | **5.030** | **153** |

- Tốc độ tokenize: 174MB → 35M token trong khoảng 15 giây.
- Còn nợ: `tests/trainer/test_train_config.py` chưa tạo lại (thiếu 13 test, `tests/trainer` đang 60 test).
- GitHub `origin/main` (commit `c4b5317`) vẫn là bản slm cũ, chưa push code mới.

---

## Môi trường

- Windows 11, PowerShell + Git Bash
- GPU: **NVIDIA RTX 3050 Laptop 4GB**, driver hỗ trợ CUDA 12.7
- Virtualenv `.venv/` (Python 3.11.9), đang có **`torch 2.14.0+cpu`** → phải cài bản CUDA trước khi train
- Hệ quả của 4GB VRAM:
  - Train trên máy bằng **model và dữ liệu tí hon** để kiểm tra code đúng; thuê GPU cho lần train thật.
  - RL trên máy dùng **reward theo quy tắc**, không dùng reward model lớn.
  - Mọi thứ phải chạy được trên **Windows**: không dùng `signal.SIGALRM`, chú ý đường dẫn có dấu cách.

---

## Lộ trình

Làm **tuần tự**, xong chặng này mới sang chặng sau. Mỗi chặng phải có **tiêu chí "xong" đo được**.

| # | Chặng | Học từ minimind | Xong khi |
|---|---|---|---|
| 0 | Chuẩn bị: torch CUDA, tokenizer, dữ liệu tí hon, thế giới đồ chơi | `trainer/train_tokenizer.py`, `model/tokenizer_config.json` | encode→decode tiếng Việt đúng; token đặc biệt không bị tách |
| 1 | Kiến trúc Transformer | `model/model_minimind.py` | test từng thành phần; đếm tham số khớp công thức |
| 2 | Pretrain | `trainer/train_pretrain.py`, `trainer/trainer_utils.py`, `PretrainDataset` | loss giảm; val loss giảm; resume khớp chạy liền mạch |
| 3 | Sinh chữ + KV cache | `generate` (`model_minimind.py:257–287`), `eval_llm.py` | sinh từng bước có cache = forward cả chuỗi |
| 4 | SFT | `SFTDataset` (`lm_dataset.py:58`), `train_full_sft.py` | nhãn chỉ phủ token assistant; học thuộc tập đồ chơi |
| 5 | LoRA | `model/model_lora.py`, `train_lora.py` | đầu ra ban đầu và sau khi gộp không đổi |
| 6 | DPO | `train_dpo.py`, `DPODataset` | loss = ln 2 khi policy = ref; tỉ lệ chosen thắng tăng |
| 7 | Tool Use | chat_template (`tools`, `tool_calls`, role `tool`), `scripts/eval_toolcall.py` | kết quả tool không được học; model gọi tool đúng trên tập đồ chơi |
| 8 | Tư duy thích ứng | `<think>` + `open_thinking` trong template, `post_processing_chat` | cùng model bật/tắt được phần tư duy |
| 9 | Chưng cất | `train_distillation.py` | KL = 0 khi student = teacher |
| 10 | Nền RL: rollout, log-prob, reward, KL | `trainer/rollout_engine.py`, `calculate_rewards` | log-prob lúc sinh = log-prob tính lại |
| 11 | GRPO → CISPO | `train_grpo.py` | lợi thế trung bình 0 trong nhóm; reward tăng |
| 12 | PPO | `train_ppo.py` | GAE với λ = 0 bằng TD một bước; reward tăng |
| 13 | Agentic RL | `train_agent.py` | token kết quả tool có mask 0; tỉ lệ đúng khi dùng tool tăng |

**Nhánh phụ** (sau chặng 2, khi nào cũng được): GQA + QK-Norm, công tắc flash attention, MoE, YaRN.
**Khi thuê GPU:** DDP, preprocess dạng stream cho kho dữ liệu lớn (phải tải lại), dữ liệu thật, reward model thật.

### Những quyết định phải chốt sớm

- **Tokenizer ở chặng 0 phải chừa sẵn token** cho chat, `<think>`/`</think>`, `<tool_call>`/`</tool_call>`, `<tool_response>`/`</tool_response>` và vài token buffer dự phòng. Thêm token sau này = train lại tokenizer = mọi model đã train thành vô dụng.
- **Thế giới đồ chơi dùng xuyên suốt**: một bộ bài toán kiểm tra được tự động (ví dụ phép tính hỏi bằng tiếng Việt) dùng chung cho SFT, DPO, tool use, thinking, RL, chưng cất — để chặng nào cũng đo được kết quả trên máy yếu.
- **Teacher của chưng cất phải dùng cùng tokenizer** → không dùng được trọng số minimind.
- **Tách thân / đầu** cho model (kiểu `MiniMindModel` / `MiniMindForCausalLM`) trước chặng PPO, vì Critic cần vector ẩn.

---

## Quy ước thư mục

- `core/` = **model**. `core` không import gì từ `trainer/`; `trainer/` import `core`.
- `trainer/` = file chạy `train_<chặng>.py` + module dùng chung (optimizer, LR schedule, checkpoint, metrics).
- `dataset/` **chỉ chứa dữ liệu** (bị gitignore), không đặt code ở đây. Chia theo chặng: `dataset/pretrain/`, sau này `dataset/sft/`, ...
- Config: dataclass, tách `ModelConfig` (kiến trúc, đi cùng trọng số) và config train (riêng từng chặng).

---

## Quy trình mỗi chặng

1. **Codex giải thích**: lý thuyết, công thức, shape, và đoạn code minimind tương ứng kèm số dòng.
2. **Người dùng thiết kế**: vị trí file, API, config.
3. **Người dùng viết code + test.**
4. **Codex review.**
5. **Chạy tí hon** trên máy, đối chiếu với tiêu chí "xong".
6. **Ghi lại** kết quả và điều học được, rồi mới sang chặng sau.

---

## Chuẩn code

### Nguyên tắc

- Đơn giản > khéo léo. Dễ đọc > ngắn. Tường minh > ngầm định.
- Mỗi file làm một việc. Hàm nhỏ, không lồng sâu.
- **Tự cài đặt thủ công** các phần lõi: attention (không dùng `torch.nn.MultiheadAttention`), RMSNorm, RoPE, LoRA, loss DPO/GRPO/PPO.
- Không thêm framework nặng khi chưa cần. `transformers` chỉ dùng để so sánh hoặc chuyển đổi trọng số.
- Không chép code minimind hay HuggingFace từng dòng.

### Mỗi thành phần phải có

- Type hint và docstring.
- Với thành phần toán học, docstring nêu: **Mục đích, Shape vào/ra, Công thức, Độ phức tạp, Lưu ý ổn định số học**.
- Hằng số có tên, không số "ma thuật".
- **Test chứng minh tính chất toán học**, không chỉ test shape. Ví dụ: RoPE chỉ phụ thuộc vị trí tương đối; causal mask chặn tương lai; DPO loss = ln 2 khi policy = ref.

### Những lỗi của minimind cần tránh

| minimind | Làm thay |
|---|---|
| Checkpoint resume lưu trọng số fp16 → resume không khớp | Lưu fp32 + optimizer + scheduler + trạng thái random |
| LR không warmup | Warmup rồi cosine |
| AdamW không weight decay | Weight decay cho ma trận, không cho norm/bias |
| Pretrain pad từng mẫu | Nối gói (packing) |
| Config dùng `kwargs.get` → tên sai bị bỏ qua im lặng | Dataclass, báo lỗi khi gặp khóa lạ |
| LoRA không có hệ số `alpha/r`, chỉ bắt Linear vuông, monkey-patch `forward` | Class bọc module, có `alpha/r`, chọn lớp theo tên |
| Tạo nhãn SFT bằng cách dò chuỗi token | Dựng nhãn theo từng tin nhắn |
| `SIGALRM` chỉ chạy trên Linux | Giới hạn thời gian chạy được trên Windows |
| Không có test | Test cho từng thành phần |

### Train và log

- Hỗ trợ fp32 / fp16 (có GradScaler) / bf16.
- Log: loss, learning rate, grad norm, tokens/s, VRAM, val loss.
- Checkpoint phân biệt rõ **resume** (tiếp tục đúng run đó) và **khởi tạo từ trọng số** (bắt đầu giai đoạn mới).

---

## Codex cần tuân thủ

### Khi giải thích

- Trả lời bằng **tiếng Việt**, thẳng thắn, dựa trên code thật (có đường dẫn và số dòng).
- Yêu cầu "giải thích" = **đọc rồi giải thích**; không cài thư viện, không viết script demo, không chạy thử, trừ khi được yêu cầu.
- Giải thích theo thứ tự: ý tưởng → công thức → shape → code → độ phức tạp → lưu ý ổn định số học.
- Có nhiều cách làm thì so sánh ngắn gọn và nêu lý do chọn.

### Khi review

- Chỉ ra vị trí `file:dòng`, mức độ nghiêm trọng, vì sao sai, hướng sửa.
- **Không tự sửa code của người dùng** trừ khi được yêu cầu.
- Kiểm tra: đúng toán học, đúng shape, có test chứng minh tính chất, chạy được trên Windows.

### Không được làm khi chưa hỏi

- Sửa bất cứ thứ gì trong `minimind-master/`.
- Xóa hay ghi đè `dataset/`.
- Cài thư viện, chạy train dài.
- Tạo thư mục/file hoặc áp đặt cấu trúc project.
- Commit hoặc push.

### Git

- Commit theo Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
- Chỉ commit/push khi người dùng yêu cầu.
