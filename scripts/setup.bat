@echo off
rem Cai moi truong lan dau: .venv + thu vien backend + node_modules cua client.
rem GPU NVIDIA: chay "scripts\setup.bat cuda" de cai torch ban CUDA.
setlocal
cd /d "%~dp0.."

if not exist .venv\Scripts\python.exe (
    echo [setup] Tao .venv ...
    py -3.11 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 ( echo [setup] Khong tao duoc .venv. Can Python 3.10+. & exit /b 1 )
)
set PY=.venv\Scripts\python.exe

%PY% -m pip install -q --upgrade pip
if /i "%~1"=="cuda" (
    echo [setup] Cai torch CUDA ...
    %PY% -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128 || exit /b 1
)
echo [setup] Cai thu vien backend ...
%PY% -m pip install -q -r backend\requirements.txt || exit /b 1

echo [setup] Cai thu vien client ...
pushd client
call npm install || ( popd & exit /b 1 )
popd

if not exist backend\models\sft\model.safetensors (
    echo [setup] Chua co trong so. Chay scripts\download_models.bat hoac scripts\export_models.bat
)
echo [setup] Xong. Chay scripts\start_sys.bat de mo ung dung.
