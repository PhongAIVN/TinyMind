@echo off
rem Tai 7 checkpoint safetensors + tokenizer tu Hugging Face vao backend\models.
setlocal
cd /d "%~dp0.."
set REPO=PhongAIVN/TinyMind-57M-vi
".venv\Scripts\hf.exe" download %REPO% --local-dir backend\models --exclude "README.md" --exclude ".gitattributes" --exclude ".gitignore"
if errorlevel 1 ( echo [download] Loi. Da chay scripts\setup.bat chua? & exit /b 1 )
echo [download] Xong: backend\models
