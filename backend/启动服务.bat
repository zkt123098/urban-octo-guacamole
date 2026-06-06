@echo off
cd /d "D:\A\backend"
start "" "D:\A\frontend\index.html"
echo Starting Yunshu backend...
start /B "YunshuBackend" uvicorn main:app --host 127.0.0.1 --port 8000
echo Starting FuXi inference service...
start /B "FuXi" python fuxi_server_win.py
echo.
echo All services are running. Do not close this window.
echo Press any key to stop...
pause