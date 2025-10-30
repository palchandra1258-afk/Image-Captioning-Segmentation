# Quick Start Script for PowerShell
# Image Captioning & Segmentation Application

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Image Captioning & Segmentation App" -ForegroundColor Cyan
Write-Host "Quick Start Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Activate virtual environment
Write-Host "[1/4] Activating virtual environment..." -ForegroundColor Yellow
$venvPath = ".\venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    & $venvPath
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv venv" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Step 2: Clear Python cache
Write-Host "[2/4] Clearing Python cache..." -ForegroundColor Yellow
Get-ChildItem -Path "." -Filter "__pycache__" -Recurse -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path "." -Filter "*.pyc" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Write-Host "✓ Cache cleared" -ForegroundColor Green

Write-Host ""

# Step 3: Verify imports
Write-Host "[3/4] Verifying imports..." -ForegroundColor Yellow
python verify_imports.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Import verification failed!" -ForegroundColor Red
    Write-Host "Please install requirements: pip install -r requirements.txt" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Step 4: Start Streamlit
Write-Host "[4/4] Starting Streamlit application..." -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Application will open in your browser" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

streamlit run app.py
