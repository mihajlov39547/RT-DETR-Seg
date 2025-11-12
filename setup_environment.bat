@echo off
REM Setup Virtual Environment for RF-DETR-Seg
REM Simple batch script alternative to PowerShell

echo ================================================
echo RF-DETR-Seg Environment Setup
echo ================================================
echo.

REM Check Python installation
echo Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found! Please install Python 3.8 or later.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version
echo.

REM Create virtual environment
if exist venv (
    echo Virtual environment already exists.
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "%RECREATE%"=="y" (
        echo Removing existing virtual environment...
        rmdir /s /q venv
    ) else (
        echo Using existing virtual environment.
        echo.
        echo To activate: venv\Scripts\activate
        pause
        exit /b 0
    )
)

echo Creating virtual environment...
python -m venv venv

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create virtual environment!
    pause
    exit /b 1
)

echo [SUCCESS] Virtual environment created!
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo [SUCCESS] Virtual environment activated!
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel --quiet
echo [SUCCESS] pip upgraded!
echo.

REM Install PyTorch
echo Installing PyTorch with CUDA support...
echo (This may take a few minutes)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] CUDA installation failed, trying CPU version...
    python -m pip install torch torchvision torchaudio
)

echo.

REM Install other requirements
echo Installing other dependencies...
echo (This may take a few minutes)
python -m pip install -r requirements.txt

echo.

REM Verify installation
echo Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

echo.
echo ================================================
echo Setup Complete!
echo ================================================
echo.
echo Next steps:
echo   1. Activate environment: venv\Scripts\activate
echo   2. Validate setup: python validate_setup.py
echo   3. Test training: python train_detector_stage1.py --size nano --epochs 1 --batch 1 --accum 4 --workers 0 --multi_scale 0
echo.
pause
