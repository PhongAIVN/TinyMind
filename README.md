# TinyMind

**English** | [Tiếng Việt](README_VI.md)

Source code on GitHub: [github.com/PhongAIVN/TinyMind](https://github.com/PhongAIVN/TinyMind)

Weights on Hugging Face: [huggingface.co/PhongAIVN/TinyMind-57M-vi](https://huggingface.co/PhongAIVN/TinyMind-57M-vi)

A small Vietnamese language model (SLM) in PyTorch, built from the tokenizer and Transformer architecture up to the pretraining loop. The goal is to go through the full training pipeline of a modern LLM:

**Pretrain → SFT → LoRA → DPO → Tool Use → Adaptive Thinking → Distillation → GRPO/CISPO → PPO → Agentic RL**

## Progress

| # | Stage             | Status                                                                                       |
| - | ----------------- | -------------------------------------------------------------------------------------------- |
| 0 | Pretrain          | Done: 57M model, 1.15B tokens, val loss **2.437**                                             |
| 1 | SFT               | Done: 155K conversations, val loss **1.555**                                                  |
| 2 | LoRA              | Done: toy arithmetic accuracy **6.3%** → **23.7%**, training only 0.34% of parameters        |
| 3 | DPO               | Done: chosen wins 72%, accuracy holds at 22% thanks to an SFT loss on chosen (RPO-style)      |
| 4 | Tool Use          | Done: calculator calls **100%** correct                                                       |
| 5 | Adaptive Thinking | Done: same model, thinking off **38.3%**, thinking on **67.3%**                               |
| 6 | Distillation      | Done: 13.3M student reaches **58.0%** (86% of the 57M teacher)                                |
| 7 | GRPO / CISPO      | Done: 67.7% → **70.0%** / **70.7%**                                                           |
| 8 | PPO               | Done: 67.7% → **68.7%**                                                                       |
| 9 | Agentic RL        | Ran, but learned nothing: the model was already 100% correct at the start; needs harder tasks |

## 57M model

`configs/model_50m.json`

| Parameter              | Value              |
| ---------------------- | ------------------ |
| `vocab_size`           | 32,000             |
| `d_model`              | 512                |
| `n_layers`             | 12                 |
| `n_heads`              | 8 (head_dim 64)    |
| `ffn_hidden`           | 1,536              |
| `max_seq_len`          | 512                |
| **Total parameters**   | **57,291,264**     |

The parameter count matches the formula, and a test checks it:

```
P = V·d + L·(4d² + 3·d·f + 2d) + d
  = 32000·512 + 12·(4·512² + 3·512·1536 + 2·512) + 512
```

Where: embedding `V·d` (tied, so `lm_head` is not counted again), attention `4d²` (Q, K, V, O), SwiGLU `3·d·f` (gate, up, down), 2 RMSNorms per layer (`2d`), and one final RMSNorm (`d`). RoPE has no parameters.

---

## Pretraining data

About **1.15B training tokens** + 5.8M eval tokens, roughly 20 tokens per parameter (Chinchilla-optimal). Each source holds out 1/200 of its documents for eval.

| Source                                                                                                                                              | Content                               |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| [epfml/FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ) (`vie_Latn`)                                                                | High-quality Vietnamese web text      |
| [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (`20231101.vi`)                                                         | General knowledge                     |
| [vilm/OpenOrca-Viet](https://huggingface.co/datasets/vilm/OpenOrca-Viet)                                                                             | Q&A, explanations                     |
| [5CD-AI/Vietnamese-alpaca-gpt4-gg-translated](https://huggingface.co/datasets/5CD-AI/Vietnamese-alpaca-gpt4-gg-translated)                           | Basic Q&A                             |
| [bkai-foundation-models/vi-alpaca](https://huggingface.co/datasets/bkai-foundation-models/vi-alpaca)                                                 | Q&A, instructions                     |
| [5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated](https://huggingface.co/datasets/5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated) | Math with worked solutions            |
| [taidng/UIT-ViQuAD2.0](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0)                                                                         | Reading comprehension                 |
| [PeterPaker123/vietnamese-code-dataset](https://huggingface.co/datasets/PeterPaker123/vietnamese-code-dataset)                                       | Vietnamese problems, code solutions   |
| [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)                         | Instructions + Python code            |
| [OpenCoder-LLM/opc-annealing-corpus](https://huggingface.co/datasets/OpenCoder-LLM/opc-annealing-corpus) (`synthetic_qa`, Python only)             | High-quality code, up to 600MB        |

Q&A data is flattened into plain text using the template `Câu hỏi: … / Trả lời: …` ("Question: … / Answer: …").

## SFT data

Up to 20,000 conversations per source, **155,056 training conversations** + 778 eval in total (`scripts/build_sft_data.py`). Chat template:

```
<|bos|><|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
```

- **Single-turn Q&A**: the 7 Q&A sources above (OpenOrca, 2 alpaca sets, MetaMath, ViQuAD, 2 code sets), converted to chat format.
- **Multi-turn conversations**, used for SFT only:

| Source                                                                                                                    | Content                          |
| ------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| [5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca](https://huggingface.co/datasets/5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca)       | Multi-turn chat                  |
| [5CD-AI/Vietnamese-Ecommerce-Multi-turn-Chat](https://huggingface.co/datasets/5CD-AI/Vietnamese-Ecommerce-Multi-turn-Chat) | Customer support, sales          |
| [nguyenthanhdo/ultrachat-75k-vi](https://huggingface.co/datasets/nguyenthanhdo/ultrachat-75k-vi)                           | UltraChat translated to Vietnamese |

Conversations longer than 512 tokens are truncated at the end; conversations whose answers are cut off entirely are dropped (5,104 of them, about 3%).

---

## Results

**Tiny model** (2.4M parameters, old 16,000-token tokenizer, 8.2M tokens, CPU):

|                  | Val loss  | Perplexity |
| ---------------- | --------- | ---------- |
| Untrained        | 9.705     | 16,408     |
| After 2,000 steps | **5.030** | **153**    |

**57M model, pretrain** (RTX 5070 12GB, bf16, SDPA, 65,536 tokens/step, 17,584 steps):

- About **73,000 tokens/s**, 7.2GB VRAM, about 4.5 hours, rental cost about 50,000 VND (~$2).
- Val loss: 10.37 at init (= ln 32000) → 3.86 (step 750) → 2.67 (6,750) → 2.49 (13,250) → **2.437** (final, perplexity about 11.4). Full log: `logs/pretrain_log.txt`.
- The model writes grammatical Vietnamese and syntactically valid Python, but its knowledge is thin, it repeats itself and does not know when to stop.

**57M model, SFT** (initialized from pretrain, 2,000 steps × 128 conversations, about 1.65 epochs, about 30 minutes):

- Val loss **1.555** (computed only on assistant answers). On 4 sample conversations, answer loss drops from 3.44 (pretrain model) to 1.24.
- Learned to answer in the template and to **stop on its own** with `<|im_end|>`.

| Question                                                         | Answer (greedy decoding)                                                         |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Thủ đô của Việt Nam là gì? (What is the capital of Vietnam?)     | Thủ đô của Việt Nam là Hà Nội. (The capital of Vietnam is Hanoi.)               |
| Kể tên 3 món ăn nổi tiếng của Việt Nam. (Name 3 famous Vietnamese dishes.) | 1. Phở … 2. Bánh mì … 3. Bánh xèo … (dish names right, descriptions wrong) |
| Viết hàm Python tính giai thừa. (Write a Python factorial function.) | Correct recursive function, but repeated many times without stopping         |
| Thủ đô của Trung Quốc là gì? (What is the capital of China?)     | "Thành Đô" / Chengdu (wrong, but plausible)                                      |

Limitations: 57M parameters hold little knowledge, so it often makes things up; it does not understand Vietnamese without diacritics or English (almost absent from the data).

**57M model, LoRA** (from SFT, r = 8, alpha = 16, on `q_proj` + `v_proj` of all 12 layers, 600 steps × 32 problems, about 20 minutes on CPU):

- Trains 196,608 parameters (0.34%); val loss 1.22 → **0.516**. Log: `logs/lora_log.txt`.
- Graded on 300 eval problems (disjoint from training), greedy decoding:

| Operation                          | SFT           | SFT + LoRA          |
| ---------------------------------- | ------------- | ------------------- |
| All                                | 19/300 = 6.3% | **71/300 = 23.7%**  |
| Addition (0–99)                    |               | 6/112 = 5%          |
| Subtraction (0–99)                 |               | 7/114 = 6%          |
| Multiplication (0–12)              |               | **58/74 = 78%**     |
| Answer follows `a op b = c.` format |              | **300/300**         |

- LoRA teaches the **answer format** almost perfectly, plus multiplication-table products (few pairs, already present in pretraining data).
- Two-digit addition/subtraction is barely learned: the tokenizer merges whole numbers like `29`, `30` into **a single token**, so the model must memorize about 10,000 number pairs like a lookup table instead of computing digit by digit. This motivates the Tool Use (calculator) and RL stages.
- Slight forgetting: asked "Name 3 dishes…", it now gives one short sentence, because LoRA only saw very short answers. The capital question is still right.

**57M model, DPO** (from merged LoRA, β = 0.1, lr 1e-5, 400 steps × 16 pairs, full-model training, about 25 minutes on CPU):

- Data: 8,000 arithmetic pairs; chosen = correct answer, rejected = a wrong answer of the kind the model tends to make (copying an operand, off by a few). Eval uses the same 300 toy problems.
- Loss starts at **0.677 ≈ ln 2**, as theory predicts; val loss 0.693 → **0.554**. Log: `logs/dpo_log.txt`.
- On 300 eval pairs, **chosen wins 72%**, mean reward margin 0.93.
- But **generation accuracy drops 23.7% → 11.0%** (multiplication 78% → 30%). Cause: DPO pushes **both** answers down, rejected just faster:

|                 | Before (ref) | After DPO |
| --------------- | ------------ | --------- |
| log p(chosen)   | -3.79        | -6.42     |
| log p(rejected) | -4.67        | -12.35    |

  Probability taken from both answers flows to other tokens: the model answers "1 + 29 = 29 **tuổi** (years old)". This is a known DPO failure when chosen and rejected are nearly identical (here they differ by exactly one token). Fixes: add an SFT loss on chosen to keep its probability (RPO-style), raise β, or sample rejected answers from the model itself.

## Project structure

```
core/        Model: config, embedding, rmsnorm, rope, attention, feedforward, block, model, lora, value_model (critic)
tokenizer/   BPE: bpe_toy.py (pure algorithm), train_tokenizer.py, tokenizer.json
trainer/     Shared training loop (pluggable loss), datasets, losses, optimizer, checkpoint, RL (rl.py),
             train_pretrain / sft / lora / dpo / distill / grpo / ppo.py
inference/   Generation (temperature, top-k, KV cache, --chat), tools + agent loop, toy-world grading
scripts/     Data prep: build_pretrain_corpus, prepare_pretrain, check_pretrain_data, build_sft_data, build_toy_*; train_all.py, eval_all.py
scripts/windows/  .bat / .ps1 scripts for Windows and rented GPU machines
configs/     Model configs (architecture) and training configs (per stage), kept separate
tests/       Tests for every component
docs/        Architecture diagrams (draw.io)
logs/        Training logs
```

## Getting started

Requires Python 3.10+.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
```

The cu128 build is needed for RTX 50xx GPUs and still works on 30xx/40xx. On CPU-only machines, drop `--index-url ...`.

**Run tests**

```bash
.venv/Scripts/python -m pytest tests -q
```

**Train the whole pipeline with one command** (Windows GPU machine, from the project root)

```bat
scripts\windows\train_all.bat
```

This sets up the environment and downloads raw data (if missing), then runs in order: pretraining data + tokenizer → pretrain → SFT data → SFT → toy data → LoRA → DPO → Tool Use → Thinking → Distillation → GRPO → CISPO → PPO → Agentic RL → results table `logs/eval_results.md`.

- Each step logs to `logs/<step>.log` and is marked done in `checkpoints/50m/done/`. If interrupted, rerun the same command: finished steps are skipped and an unfinished training step resumes with `--resume`.
- `train_all.bat --list` shows status; `--from dpo` starts from a step; `--only sft` runs one step; `--dry-run` only prints commands.
- If you already have `pretrain.pt`, copy it into `checkpoints\50m\` and run `scripts\windows\train_from_pretrain.bat`: it downloads only the SFT data (about 1GB), checks that the checkpoint matches the tokenizer, then runs from SFT to eval.

**Run each step by hand** (PowerShell)

```powershell
# venv + CUDA torch + download about 14.6GB of raw data into dataset\raw\
powershell -ExecutionPolicy Bypass -File scripts\windows\setup_gpu_vm.ps1

# raw data -> jsonl -> tokenizer -> train.bin / eval.bin -> checks
powershell -ExecutionPolicy Bypass -File scripts\windows\build_pretrain_data.ps1

# pretrain (add --resume to continue from a checkpoint)
.venv\Scripts\python.exe -u -m trainer.train_pretrain --model-config configs/model_50m.json --train-config configs/pretrain_50m.json
```

On a 12GB GPU, keep `batch_size 16` × `accum_steps 8`. With 16GB or more, try `32 × 4`; tokens per step stay the same, so `max_steps` does not change. On Windows, running out of VRAM does not raise an error: the driver spills into system RAM and training gets about 3× slower. Watch the VRAM number in the log.

**SFT** (needs `checkpoints/50m/pretrain.pt` and raw data in `dataset\raw\`)

```powershell
# Q&A + conversations -> dataset\sft\train.jsonl, eval.jsonl
.venv\Scripts\python.exe -m scripts.build_sft_data --skip-missing

# SFT, loading weights from pretrain.pt (init_from in configs/sft_50m.json)
.venv\Scripts\python.exe -u -m trainer.train_sft --model-config configs/model_50m.json --train-config configs/sft_50m.json
```

**LoRA on the toy world** (needs `checkpoints/50m/sft.pt`; runs on CPU at about 1,000 tokens/s)

```powershell
# 20,000 train + 300 eval problems, no overlap
.venv\Scripts\python.exe -m scripts.build_toy_math

# accuracy before training
.venv\Scripts\python.exe -m inference.eval_toy_math --checkpoint checkpoints/50m/sft.pt

# train LoRA (on CPU: set "device": "cpu", "dtype": "fp32" in configs/lora_50m.json); writes merged lora_merged.pt
.venv\Scripts\python.exe -u -m trainer.train_lora --rank 8 --alpha 16

# accuracy after training
.venv\Scripts\python.exe -m inference.eval_toy_math --checkpoint checkpoints/50m/lora_merged.pt
```

**DPO on the toy world** (needs `checkpoints/50m/lora_merged.pt`)

```powershell
.venv\Scripts\python.exe -m scripts.build_toy_dpo
.venv\Scripts\python.exe -u -m trainer.train_dpo --beta 0.1
.venv\Scripts\python.exe -m inference.eval_toy_math --checkpoint checkpoints/50m/dpo.pt
```

**Use the released weights, no training needed**

[PhongAIVN/TinyMind-57M-vi](https://huggingface.co/PhongAIVN/TinyMind-57M-vi) has 7 checkpoints in safetensors format: `pretrain`, `sft`, `lora_merged`, `tool`, `think`, `cispo`, `student`. For general Q&A use `sft`; for toy math use `cispo` (thinking on) or `tool`. See the model card for how to load them.

```bash
.venv/Scripts/hf download PhongAIVN/TinyMind-57M-vi --include "sft/*" --include "tokenizer.json" --local-dir hf
```

**Generate text**

```bash
# pretrained model: continue text
.venv/Scripts/python -m inference.generate --checkpoint checkpoints/50m/pretrain.pt --prompt "Hà Nội là"

# SFT model: Q&A with the chat template
.venv/Scripts/python -m inference.generate --checkpoint checkpoints/50m/sft.pt --chat --prompt "Thủ đô của Việt Nam là gì?"
```

`tokenizer/tokenizer.json` must be the exact file used to tokenize the pretraining data. Using a different tokenizer (even with the same vocab size) makes the model output gibberish.
