@echo off
echo ========================================
echo Image Captioning ^& Segmentation App
echo Quick Start Script
echo ========================================
echo.

echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Could not activate virtual environment!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)
echo ✓ Virtual environment activated

echo.
echo [2/3] Verifying imports...
python verify_imports.py
if %errorlevel% neq 0 (
    echo ERROR: Import verification failed!
    echo Please install requirements: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Streamlit application...
echo.
echo ========================================
echo Application will open in your browser
echo Press Ctrl+C to stop the server
echo ========================================
echo.

streamlit run app.py

pause
