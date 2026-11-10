@echo off
rem Client Vite tai http://localhost:5173 (goi /api qua proxy sang backend)
cd /d "%~dp0..\client"
if not exist node_modules call npm install
npm run dev
