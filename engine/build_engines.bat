@echo off
rem Batch script to run the TensorRT engine compiler
echo ===================================================
echo   Upscalerr - TensorRT Engine Compilation Batch
echo ===================================================

set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

rem Check if python is available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH.
    pause
    exit /b 1
)

rem Find trtexec
where trtexec >nul 2>nul
if %errorlevel% neq 0 (
    if not defined TRTEXEC_PATH (
        echo [WARNING] trtexec not found in PATH. Checking common NVIDIA paths...
        if exist "C:\Program Files\NVIDIA\TensorRT\bin\trtexec.exe" (
            set "TRTEXEC_PATH=C:\Program Files\NVIDIA\TensorRT\bin\trtexec.exe"
            echo Found trtexec at: C:\Program Files\NVIDIA\TensorRT\bin\trtexec.exe
        ) else if exist "C:\TensorRT\bin\trtexec.exe" (
            set "TRTEXEC_PATH=C:\TensorRT\bin\trtexec.exe"
            echo Found trtexec at: C:\TensorRT\bin\trtexec.exe
        ) else (
            echo [ERROR] Could not find trtexec. Please install TensorRT or set TRTEXEC_PATH.
            pause
            exit /b 1
        )
    )
)

echo Building engines for Scale 2x...
python build_engines.py --scale 2 --opt-level 5

if %errorlevel% neq 0 (
    echo [ERROR] Engine compilation failed.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo [SUCCESS] Engine compilation completed successfully!
echo ===================================================
pause
