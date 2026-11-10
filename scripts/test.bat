@echo off
rem Test backend (pytest) + lint va build client (tsc kiem tra kieu).
setlocal
cd /d "%~dp0.."
set FAILED=0

echo === backend: pytest ===
pushd backend
"..\.venv\Scripts\python.exe" -m pytest tests -q || set FAILED=1
popd

echo === client: lint ===
pushd client
call npm run lint || set FAILED=1
echo === client: build ===
call npm run build || set FAILED=1
popd

if %FAILED%==1 ( echo [test] CO LOI & exit /b 1 )
echo [test] Tat ca deu qua
