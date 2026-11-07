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
if "%PROMPT%"=="" set "PROMPT=Dân tộc Việt Nam"

"%PYTHON%" -m inference.generate --checkpoint "%CHECKPOINT%" --tokenizer "%TOKENIZER%" --prompt "%PROMPT%" --max-new-tokens %MAX_NEW_TOKENS% --temperature %TEMPERATURE% --top-k %TOP_K%
set "EXIT_CODE=%ERRORLEVEL%"

:finish

pause

popd
endlocal & exit /b %EXIT_CODE%