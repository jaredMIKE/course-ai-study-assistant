$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
Set-Location $PSScriptRootackend
if (!(Test-Path ".venv\Scripts\python.exe")) {
  Write-Host "[1/4] Creating Python virtual environment..."
  py -3 -m venv .venv
}
. .\.venv\Scripts\Activate.ps1
Write-Host "[2/4] Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if (!(Test-Path ".env")) { Copy-Item ".env.example" ".env" }
Write-Host "[3/4] Backend is starting on port 8010. Browser will open automatically."
Start-Process "http://127.0.0.1:8010"
Write-Host "[4/4] Server address: http://127.0.0.1:8010"
python -m uvicorn main:app --host 127.0.0.1 --port 8010 --reload
