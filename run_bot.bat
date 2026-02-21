@echo off
REM Ghost-Shell Bot Launcher
REM This batch file runs the Ghost-Shell chess bot

cd /d "%~dp0"

REM Use full path to Python 3.12
set PYTHON=C:\Python312\python.exe

REM Check if Python exists
if not exist "%PYTHON%" (
    echo Python 3.12 not found at C:\Python312\python.exe
    echo Please install Python or update the PYTHON path in this script
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found
    echo Please run: "%PYTHON%" -m venv venv
    pause
    exit /b 1
)

REM Activate virtual environment and run the bot
call venv\Scripts\activate.bat
"%PYTHON%" main.py
pause
