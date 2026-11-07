@echo off
rem ================================================================
rem  Chay toan bo test cua project (thu muc tests\).
rem
rem  Cach dung - chay tu bat ky thu muc nao:
rem    scripts\windows\test_full.bat              tat ca test
rem    scripts\windows\test_full.bat -x           dung o test fail dau tien
rem    scripts\windows\test_full.bat -k rope      chi chay test co chu "rope"
rem    scripts\windows\test_full.bat -q           in gon, khong liet ke tung test
rem
rem  Moi tham so them vao deu duoc chuyen thang cho pytest.
rem  Ma thoat: 0 = tat ca pass, khac 0 = co test fail hoac loi.
rem ================================================================
setlocal

rem In duoc tieng Viet (ten test, thong bao loi) ra console
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"

rem Thu muc goc project = 2 cap tren thu muc chua file .bat nay
pushd "%~dp0..\.."

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo [LOI] Khong tim thay %PYTHON%
    echo       Tao virtualenv truoc: python -m venv .venv
    popd
    endlocal
    exit /b 1
)

echo Thu muc project: %CD%
echo Python:          %PYTHON%
echo.

rem "python -m pytest" (khong phai "pytest") de import duoc core\ va tokenizer\
"%PYTHON%" -m pytest tests -v %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
    echo [OK] Tat ca test deu pass.
) else (
    echo [FAIL] Co test that bai hoac loi - ma thoat %EXIT_CODE%.
)

pause

popd
endlocal & exit /b %EXIT_CODE%
