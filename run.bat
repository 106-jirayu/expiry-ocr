@echo off
setlocal
cd /d "%~dp0"
title Expiry Date AI OCR

rem Double-click to install (first time) and start the system.

if exist ".venv\Scripts\python.exe" goto run

echo [1/2] Creating virtual environment ...
py -3 -m venv .venv >nul 2>&1
if exist ".venv\Scripts\python.exe" goto install
python -m venv .venv
if exist ".venv\Scripts\python.exe" goto install
echo.
echo ERROR: Python not found.
echo Install Python 3.10+ from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during setup, then run this file again.
pause
exit /b 1

:install
echo [2/2] Installing packages - first time only, takes a few minutes ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto pipfail

:run
if not exist ".env" copy ".env.example" ".env" >nul
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" run.py %*
pause
exit /b 0

:pipfail
echo.
echo ERROR: package installation failed. See the messages above.
rmdir /s /q .venv
pause
exit /b 1
