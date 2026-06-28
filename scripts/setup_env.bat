@echo off
rem Setup python virtual environment and verify dependencies for Upscalerr
echo ===================================================
echo   Upscalerr - Setup Environment & Verify Pipeline
echo ===================================================

set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%\..

rem 1. Check Python version
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

rem 2. Create virtual environment
if not exist venv (
    echo Creating python virtual environment in venv/...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

rem 3. Activate venv and install dependencies
call venv\Scripts\activate.bat
echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch, Torchvision and ONNX pipeline dependencies...
pip install -r training/requirements.txt

echo Installing PySide6 & Windows API wrappers...
pip install PySide6 pywin32

rem 4. Verify CUDA runtime availability in Python
python -c "import torch; print('CUDA available in PyTorch:', torch.cuda.is_available())"

echo.
echo ===================================================
echo [SUCCESS] Virtual environment ready.
echo ===================================================
pause
