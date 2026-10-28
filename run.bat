@echo off
chcp 65001 >nul
cd /d "%~dp0"
REM ดับเบิลคลิกไฟล์นี้เพื่อติดตั้ง (ครั้งแรก) และเปิดระบบ

if not exist ".venv\Scripts\python.exe" (
    echo [1/2] Creating virtual environment...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo ไม่พบ Python - ติดตั้งจาก https://www.python.org/downloads/ แล้วติ๊ก "Add python.exe to PATH"
        pause
        exit /b 1
    )
    echo [2/2] Installing packages (first time only, a few minutes)...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ติดตั้งแพ็กเกจไม่สำเร็จ
        pause
        exit /b 1
    )
)
if not exist ".env" copy ".env.example" ".env" >nul
".venv\Scripts\python.exe" run.py %*
pause
