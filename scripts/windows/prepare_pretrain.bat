@echo off
rem ================================================================
rem  Tokenize du lieu pretrain: .txt  ->  .bin + _meta.json
rem
rem  Cach dung - chay tu bat ky thu muc nao:
rem    scripts\windows\prepare_pretrain.bat          ca train va eval
rem    scripts\windows\prepare_pretrain.bat train    chi tap train
rem    scripts\windows\prepare_pretrain.bat eval     chi tap eval
rem
rem  Doc:  dataset\pretrain\<split>\*.txt
rem  Ghi:  dataset\pretrain\tokens\<split>.bin va <split>_meta.json
rem
rem  Ma thoat: 0 = xong, 1 = thieu venv, 2 = tham so sai,
rem            3 = chua co tokenizer, 4 = thieu thu muc du lieu
rem ================================================================
setlocal

chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"

rem Thu muc goc project = 2 cap tren thu muc chua file .bat nay
pushd "%~dp0..\.."

set "PYTHON=.venv\Scripts\python.exe"
set "TOKENIZER=tokenizer\tokenizer.json"
set "EXIT_CODE=0"

if not exist "%PYTHON%" (
    echo [LOI] Khong tim thay %PYTHON%
    echo       Tao virtualenv truoc: python -m venv .venv
    set "EXIT_CODE=1"
    goto :finish
)

if not exist "%TOKENIZER%" (
    echo [LOI] Khong tim thay %TOKENIZER%
    echo       Train tokenizer truoc: python -m tokenizer.train_tokenizer
    set "EXIT_CODE=3"
    goto :finish
)

set "MODE=%~1"
if "%MODE%"=="" set "MODE=all"

if /i "%MODE%"=="all"   goto :run_all
if /i "%MODE%"=="train" goto :run_train
if /i "%MODE%"=="eval"  goto :run_eval

echo [LOI] Tham so khong hop le: %MODE%
echo       Chi nhan: train, eval, hoac de trong (ca hai).
set "EXIT_CODE=2"
goto :finish

:run_all
call :prepare train
if errorlevel 1 goto :failed
call :prepare eval
if errorlevel 1 goto :failed
goto :done

:run_train
call :prepare train
if errorlevel 1 goto :failed
goto :done

:run_eval
call :prepare eval
if errorlevel 1 goto :failed
goto :done

:done
echo.
echo [OK] Xong. File nam trong dataset\pretrain\tokens\
goto :finish

:failed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo [FAIL] Dung lai - ma thoat %EXIT_CODE%.
goto :finish

rem ---------------------------------------------------------------
rem  Ham con: tokenize mot split (train hoac eval)
rem ---------------------------------------------------------------
:prepare
set "SPLIT=%~1"
if not exist "dataset\pretrain\jsonl\%SPLIT%" (
    echo [LOI] Khong co thu muc dataset\pretrain\jsonl\%SPLIT%
    echo       Chay truoc: python -m scripts.build_pretrain_corpus
    exit /b 4
)
echo.
echo === %SPLIT% ===
"%PYTHON%" -m scripts.prepare_pretrain --tokenizer "%TOKENIZER%" --in-dir "dataset/pretrain/jsonl/%SPLIT%" --out-prefix "dataset/pretrain/tokens/%SPLIT%"
exit /b

:finish
popd
echo.
pause
endlocal & exit /b %EXIT_CODE%
