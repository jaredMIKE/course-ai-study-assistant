@echo off
chcp 65001 >nul
cd /d "%~dp0backend"
if not exist .venv\Scripts\python.exe (
  echo [1/4] Creating Python virtual environment...
  py -3 -m venv .venv
)
call .venv\Scriptsctivate.bat
echo [2/4] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if not exist .env copy .env.example .env
echo [3/4] Backend is starting on port 8010. Browser will open automatically.
start "" /min cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:8010"
echo [4/4] Server address: http://127.0.0.1:8010
echo Press Ctrl+C to stop the server.
python -m uvicorn main:app --host 127.0.0.1 --port 8010 --reload
pause
