# Setup Virtual Environment for RF-DETR-Seg
# PowerShell script to create and activate a Python virtual environment

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "RF-DETR-Seg Environment Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
$pythonCmd = $null

# Try different Python commands
$pythonCommands = @("python", "python3", "py")
foreach ($cmd in $pythonCommands) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = $cmd
            Write-Host "✓ Found: $version using '$cmd'" -ForegroundColor Green
            break
        }
    } catch {
        continue
    }
}

if ($null -eq $pythonCmd) {
    Write-Host "✗ Python not found! Please install Python 3.8 or later." -ForegroundColor Red
    Write-Host "  Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Check Python version
$versionOutput = & $pythonCmd --version 2>&1
if ($versionOutput -match "Python (\d+)\.(\d+)") {
    $major = [int]$matches[1]
    $minor = [int]$matches[2]
    
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
        Write-Host "✗ Python 3.8+ required, found $major.$minor" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Create virtual environment
$venvPath = "venv"
if (Test-Path $venvPath) {
    Write-Host "Virtual environment already exists at: $venvPath" -ForegroundColor Yellow
    $response = Read-Host "Do you want to recreate it? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvPath
    } else {
        Write-Host "Using existing virtual environment." -ForegroundColor Green
        Write-Host ""
        Write-Host "To activate the environment, run:" -ForegroundColor Cyan
        Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
        exit 0
    }
}

Write-Host "Creating virtual environment..." -ForegroundColor Yellow
& $pythonCmd -m venv $venvPath

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to create virtual environment!" -ForegroundColor Red
    Write-Host "  Try: $pythonCmd -m pip install --upgrade pip" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Virtual environment created successfully!" -ForegroundColor Green
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
$activateScript = ".\venv\Scripts\Activate.ps1"

# Check if activation is allowed
$executionPolicy = Get-ExecutionPolicy
if ($executionPolicy -eq "Restricted") {
    Write-Host "⚠ PowerShell execution policy is Restricted" -ForegroundColor Yellow
    Write-Host "  Run this command as Administrator:" -ForegroundColor Yellow
    Write-Host "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor White
    Write-Host ""
    Write-Host "  Or activate manually with:" -ForegroundColor Yellow
    Write-Host "  $activateScript" -ForegroundColor White
    exit 0
}

try {
    & $activateScript
    Write-Host "✓ Virtual environment activated!" -ForegroundColor Green
} catch {
    Write-Host "⚠ Could not auto-activate. Run manually:" -ForegroundColor Yellow
    Write-Host "  $activateScript" -ForegroundColor White
}

Write-Host ""

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
& python -m pip install --upgrade pip setuptools wheel --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ pip upgraded successfully!" -ForegroundColor Green
} else {
    Write-Host "⚠ Failed to upgrade pip (continuing anyway)" -ForegroundColor Yellow
}

Write-Host ""

# Install PyTorch (CUDA 11.8 for most GPUs)
Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Yellow
Write-Host "  (This may take a few minutes)" -ForegroundColor Gray

$torchInstalled = $false

# Try CUDA 11.8 first (most compatible)
Write-Host "  Attempting CUDA 11.8 installation..." -ForegroundColor Gray
& python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet

if ($LASTEXITCODE -eq 0) {
    $torchInstalled = $true
    Write-Host "✓ PyTorch with CUDA 11.8 installed!" -ForegroundColor Green
} else {
    Write-Host "⚠ CUDA 11.8 installation failed, trying CPU version..." -ForegroundColor Yellow
    & python -m pip install torch torchvision torchaudio --quiet
    
    if ($LASTEXITCODE -eq 0) {
        $torchInstalled = $true
        Write-Host "✓ PyTorch CPU version installed (GPU support not available)" -ForegroundColor Yellow
    }
}

if (-not $torchInstalled) {
    Write-Host "✗ Failed to install PyTorch!" -ForegroundColor Red
    Write-Host "  Install manually from: https://pytorch.org/get-started/locally/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Install other requirements
Write-Host "Installing other dependencies..." -ForegroundColor Yellow
Write-Host "  (This may take a few minutes)" -ForegroundColor Gray

& python -m pip install -r requirements.txt --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ All dependencies installed successfully!" -ForegroundColor Green
} else {
    Write-Host "⚠ Some dependencies may have failed to install" -ForegroundColor Yellow
    Write-Host "  Check errors above and install manually if needed" -ForegroundColor Yellow
}

Write-Host ""

# Verify installation
Write-Host "Verifying installation..." -ForegroundColor Yellow

$verification = @"
import sys
import torch
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"@

& python -c $verification

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Activate environment (if not already):" -ForegroundColor White
Write-Host "     .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Validate setup:" -ForegroundColor White
Write-Host "     python validate_setup.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Test training (1 epoch):" -ForegroundColor White
Write-Host "     python train_detector_stage1.py --size nano --epochs 1 --batch 1 --accum 4 --workers 0 --multi_scale 0" -ForegroundColor Gray
Write-Host ""
