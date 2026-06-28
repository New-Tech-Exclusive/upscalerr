@echo off
rem Launch batch script starting the C++ engines and UI dashboard
echo ===================================================
echo   Upscalerr - Hardware Accelerator Launcher
echo ===================================================

set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%\..

rem 1. Check if backend executable is built
set BACKEND_EXE=backend\build\bin\Release\upscalerr_backend.exe
if not exist "%BACKEND_EXE%" (
    set BACKEND_EXE=backend\build\bin\upscalerr_backend.exe
    if not exist "%BACKEND_EXE%" (
        echo [WARNING] C++ backend upscalerr_backend.exe not found.
        echo Please compile the backend using CMake first:
        echo   cd backend
        echo   mkdir build
        echo   cd build
        echo   cmake ..
        echo   cmake --build . --config Release
        echo.
        echo Attempting to start the UI anyway...
    )
)

rem 2. Copy engine profiles next to the backend executable so paths resolve correctly
if exist "%BACKEND_EXE%" (
    for %%f in (profiles\*.engine) do (
        copy /Y "%%f" "backend\build\bin\Release\" >nul 2>&1
        copy /Y "%%f" "backend\build\bin\" >nul 2>&1
    )
    echo [INFO] Engine profiles copied to backend executable directory.
)

rem 3. Activate virtual environment
if exist venv (
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] venv not found. Ensure python dependencies are installed.
)

rem 4. Start C++ backend in a separate terminal window if executable exists
if exist "%BACKEND_EXE%" (
    echo Launching backend: %BACKEND_EXE%
    start "Upscalerr C++ Engines" "%BACKEND_EXE%"
    rem Wait briefly to let server socket map
    timeout /t 3 >nul
) else (
    echo [WARNING] Backend executable not found, starting UI only.
)

rem 5. Start Python Dashboard
echo Launching PySide6 Dashboard...
python ui/main.py

echo Launcher exiting...