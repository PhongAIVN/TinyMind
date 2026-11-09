@echo off
REM Train TinyMind tu dau den cuoi tren may GPU Windows.
REM   scripts\windows\train_all.bat                 chay moi buoc chua xong
REM   scripts\windows\train_all.bat --list          xem trang thai cac buoc
REM   scripts\windows\train_all.bat --from sft      chay tu buoc sft tro di
REM Bi ngat giua chung: chay lai dung lenh nay, buoc da xong duoc bo qua, buoc dang do tu --resume.
setlocal
cd /d "%~dp0..\.."

if not exist ".venv\Scripts\python.exe" goto setup
if not exist "dataset\raw\fineweb_vi" goto setup
goto train

:setup
echo === Cai moi truong + tai du lieu tho (khoang 15GB) ===
powershell -ExecutionPolicy Bypass -File scripts\windows\setup_gpu_vm.ps1
if errorlevel 1 (
    echo [LOI] setup_gpu_vm.ps1 that bai.
    exit /b 1
)

:train
".venv\Scripts\python.exe" -u -m scripts.train_all %*
exit /b %errorlevel%
