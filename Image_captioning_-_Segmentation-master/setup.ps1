# Quick Start Script for Image Captioning & Segmentation App
# PowerShell script for Windows

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Image Captioning & Segmentation Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "Python 3\.(1[0-9]|[2-9][0-9])") {
    Write-Host "✓ $pythonVersion found" -ForegroundColor Green
} else {
    Write-Host "✗ Python 3.10+ required. Please install from https://www.python.org/" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists. Skipping..." -ForegroundColor Yellow
} else {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "✓ pip upgraded" -ForegroundColor Green

# Install requirements
Write-Host ""
Write-Host "Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "✗ Error installing dependencies" -ForegroundColor Red
    exit 1
}

# Download NLTK data
Write-Host ""
Write-Host "Downloading NLTK data..." -ForegroundColor Yellow
python -c "import nltk; nltk.download('punkt', quiet=True)"
Write-Host "✓ NLTK data downloaded" -ForegroundColor Green

# Create necessary directories
Write-Host ""
Write-Host "Creating directories..." -ForegroundColor Yellow
$directories = @(
    "models\checkpoints",
    "static\samples",
    "outputs"
)
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "✓ Directories created" -ForegroundColor Green

# Check for GPU support
Write-Host ""
Write-Host "Checking GPU support..." -ForegroundColor Yellow
$gpuCheck = python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>&1
if ($gpuCheck -eq "CUDA") {
    $gpuName = python -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
    Write-Host "✓ GPU available: $gpuName" -ForegroundColor Green
} else {
    Write-Host "ℹ GPU not available. Using CPU mode." -ForegroundColor Cyan
}

# Final summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Add sample images to static/samples/ (optional)" -ForegroundColor White
Write-Host "2. Download model checkpoints to models/checkpoints/ (see README)" -ForegroundColor White
Write-Host "3. Run the app with: streamlit run app.py" -ForegroundColor White
Write-Host ""
Write-Host "Or start now with:" -ForegroundColor Yellow
Write-Host "  streamlit run app.py" -ForegroundColor Cyan
Write-Host ""

# Ask if user wants to run the app
$response = Read-Host "Do you want to start the app now? (y/n)"
if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host ""
    Write-Host "Starting Streamlit app..." -ForegroundColor Green
    Write-Host "The app will open in your browser at http://localhost:8501" -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop the app" -ForegroundColor Yellow
    Write-Host ""
    streamlit run app.py
} else {
    Write-Host ""
    Write-Host "You can start the app later with: streamlit run app.py" -ForegroundColor Cyan
}
