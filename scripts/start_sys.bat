@echo off
rem Mo backend va client trong hai cua so rieng
start "TinyMind backend" cmd /k call "%~dp0start_be.bat"
start "TinyMind client" cmd /k call "%~dp0start_fe.bat"
