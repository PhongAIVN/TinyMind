@echo off
rem ================================================================
rem  Pretrain TU DAU (bat dau tu trong so ngau nhien).
rem
rem  Cach dung - chay tu bat ky thu muc nao:
rem    scripts\windows\train_pretrain.bat          train moi
rem    scripts\windows\train_pretrain.bat force    ghi de checkpoint cu
rem
rem  Chay tiep mot run dang do: dung continue-train_pretrain.bat
rem
rem  Doi config: sua 2 dong SET ben duoi, hoac dat bien moi truong
rem  MODEL_CONFIG / TRAIN_CONFIG truoc khi goi.
rem
rem  Ma thoat: 0 = xong, 1 = thieu venv, 3 = thieu config,
rem            4 = thieu du lieu .bin, 5 = da co checkpoint
rem ================================================================
setlocal

chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"

rem Thu muc goc project = 2 cap tren thu muc chua file .bat nay
pushd "%~dp0..\.."

set "PYTHON=.venv\Scripts\python.exe"
set "CHECKPOINT=checkpoints\pretrain.pt"
if not defined MODEL_CONFIG set "MODEL_CONFIG=configs\model_tiny.json"
if not defined TRAIN_CONFIG set "TRAIN_CONFIG=configs\pretrain_tiny.json"
set "EXIT_CODE=0"

if not exist "%PYTHON%" (
    echo [LOI] Khong tim thay %PYTHON%
    set "EXIT_CODE=1"
    goto :finish
)

if not exist "%MODEL_CONFIG%" (
    echo [LOI] Khong tim thay %MODEL_CONFIG%
    set "EXIT_CODE=3"
    goto :finish
)

if not exist "%TRAIN_CONFIG%" (
    echo [LOI] Khong tim thay %TRAIN_CONFIG%
    set "EXIT_CODE=3"
    goto :finish
)

if not exist "dataset\pretrain\tokens\train.bin" (
    echo [LOI] Chua co dataset\pretrain\tokens\train.bin
    echo       Chay truoc: scripts\windows\prepare_pretrain.bat
    set "EXIT_CODE=4"
    goto :finish
)

if exist "%CHECKPOINT%" if /i not "%~1"=="force" (
    echo [DUNG] Da co %CHECKPOINT%
    echo        - Chay tiep run do:      scripts\windows\continue-train_pretrain.bat
    echo        - Train lai tu dau:      scripts\windows\train_pretrain.bat force
    set "EXIT_CODE=5"
    goto :finish
)

echo Model config: %MODEL_CONFIG%
echo Train config: %TRAIN_CONFIG%
echo.

"%PYTHON%" -m trainer.train_pretrain --model-config "%MODEL_CONFIG%" --train-config "%TRAIN_CONFIG%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
    echo [OK] Xong. Checkpoint: %CHECKPOINT%
) else (
    echo [FAIL] Train dung lai - ma thoat %EXIT_CODE%.
)

:finish
popd
echo.
pause
endlocal & exit /b %EXIT_CODE%
