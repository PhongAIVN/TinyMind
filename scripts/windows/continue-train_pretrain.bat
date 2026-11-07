@echo off
rem ================================================================
rem  Chay TIEP mot run pretrain dang do (doc checkpoints\pretrain.pt).
rem
rem  Cach dung - chay tu bat ky thu muc nao:
rem    scripts\windows\continue-train_pretrain.bat
rem
rem  Train moi tu dau: dung train_pretrain.bat
rem
rem  LUU Y: phai dung DUNG config cua lan chay truoc. Doi max_steps
rem  se lam lich learning rate nhay bac, ket qua khong con lien mach.
rem
rem  Ma thoat: 0 = xong, 1 = thieu venv, 3 = thieu config,
rem            4 = thieu du lieu .bin, 6 = chua co checkpoint
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

if not exist "%CHECKPOINT%" (
    echo [LOI] Chua co %CHECKPOINT% de chay tiep
    echo       Train moi: scripts\windows\train_pretrain.bat
    set "EXIT_CODE=6"
    goto :finish
)

echo Chay tiep tu: %CHECKPOINT%
echo Model config: %MODEL_CONFIG%
echo Train config: %TRAIN_CONFIG%
echo.

"%PYTHON%" -m trainer.train_pretrain --model-config "%MODEL_CONFIG%" --train-config "%TRAIN_CONFIG%" --resume
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
