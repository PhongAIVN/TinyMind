@echo off
REM Train tiep tu pretrain.pt co san: SFT -> LoRA -> DPO -> Tool -> Think -> Chung cat -> GRPO -> CISPO -> PPO -> Agentic RL -> eval.
REM
REM Truoc khi chay: chep pretrain.pt vao checkpoints\50m\pretrain.pt
REM   scripts\windows\train_from_pretrain.bat
REM Bi ngat giua chung: chay lai dung lenh nay (buoc da xong bo qua, buoc dang do tu --resume).
REM
REM KHONG train lai tokenizer: tokenizer\tokenizer.json trong repo la dung file da tokenize du lieu cua pretrain.pt.
REM Chi tai du lieu cho SFT (khoang 1GB), khong tai du lieu pretrain (khoang 14GB).
setlocal
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0..\.."

set "PRETRAIN=checkpoints\50m\pretrain.pt"
set "DONE_DIR=checkpoints\50m\done"
set "SFT_SOURCES=openorca_vi,alpaca_gpt4_vi,vi_alpaca,metamath_vi,viquad,code_vi,python_instruct,chat_alpaca_vi,ecommerce_chat_vi,ultrachat_vi"

if not exist "%PRETRAIN%" (
    echo [LOI] Chua co %PRETRAIN%. Chep file pretrain.pt vao do roi chay lai.
    exit /b 1
)
if not exist "%DONE_DIR%" mkdir "%DONE_DIR%"

if exist "%DONE_DIR%\setup_sft.done" goto check
echo === 1. Moi truong + du lieu cho SFT ===
powershell -ExecutionPolicy Bypass -Command "& '.\scripts\windows\setup_gpu_vm.ps1' -Only '%SFT_SOURCES%'"
if errorlevel 1 (
    echo [LOI] setup_gpu_vm.ps1 that bai.
    exit /b 1
)
echo ok> "%DONE_DIR%\setup_sft.done"

:check
echo === 2. Kiem tra pretrain.pt mo duoc va khop tokenizer ===
".venv\Scripts\python.exe" -c "import sys, torch; from tokenizers import Tokenizer; p = torch.load('checkpoints/50m/pretrain.pt', map_location='cpu', weights_only=False); m = p['model_config']['vocab_size']; t = Tokenizer.from_file('tokenizer/tokenizer.json').get_vocab_size(); print('pretrain.pt: buoc', p['step'], '- vocab model', m, '- vocab tokenizer', t); sys.exit(0 if m == t else 1)"
if errorlevel 1 (
    echo [LOI] pretrain.pt hong ^(copy chua het?^) hoac khong khop tokenizer\tokenizer.json.
    exit /b 1
)

REM Danh dau 2 buoc pretrain da xong de train_all khong chay lai (va khong train lai tokenizer).
echo from-pretrain> "%DONE_DIR%\pretrain_data.done"
echo from-pretrain> "%DONE_DIR%\pretrain.done"

echo === 3. Train tu SFT den het ===
".venv\Scripts\python.exe" -u -m scripts.train_all --from sft_data %*
exit /b %errorlevel%
