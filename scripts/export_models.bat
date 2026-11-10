@echo off
rem Chuyen checkpoint .pt (checkpoints\50m) sang backend\models\<ten>\model.safetensors.
rem Moi file duoc nap lai va so logits voi ban goc truoc khi bao "ok".
setlocal
cd /d "%~dp0..\backend"
set SOURCE=%~1
if "%SOURCE%"=="" set SOURCE=..\checkpoints\50m
"..\.venv\Scripts\python.exe" -m scripts.export_models --source "%SOURCE%"
