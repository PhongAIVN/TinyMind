@echo off
rem Backend FastAPI tai http://127.0.0.1:8000
cd /d "%~dp0..\backend"
"..\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
