@echo off
setlocal

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

if errorlevel 1 (
    echo Failed to install dependencies.
    exit /b 1
)

python -m PyInstaller --onefile --name DesktopAgentServer --collect-submodules uvicorn main.py

if errorlevel 1 (
    echo Failed to build executable.
    exit /b 1
)

echo.
echo Build successful: dist\DesktopAgentServer.exe
pause