@echo off
cd /d "%~dp0"
if not exist .env copy .env.example .env
python -m uvicorn main:app --host 127.0.0.1 --port 8010 --reload
pause
