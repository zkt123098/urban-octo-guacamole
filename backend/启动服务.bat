@echo off
cd /d "D:\A\backend"
start "" "D:\A\frontend\index.html"
uvicorn main:app --host 127.0.0.1 --port 8000
pause