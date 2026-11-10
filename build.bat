@echo off
pip install -r requirements-dev.txt
pyinstaller --onefile --name DesktopAgentServer --collect-submodules uvicorn main.py
echo Done: dist\DesktopAgentServer.exe