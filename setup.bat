@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo  SraVaani Flow - setup
echo ============================================================
echo.

set "PY="
for %%V in (3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1
        if !errorlevel! equ 0 (
            set "PY=py -%%V"
            echo [ok] Found Python %%V
        )
    )
)

if not defined PY (
    python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info<(3,13) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY=python"
        echo [ok] Using python from PATH
    )
)

if not defined PY (
    echo [!!] Python 3.10, 3.11 or 3.12 is required.
    echo      PyTorch does not publish wheels for 3.13+ yet.
    echo      Install from https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [..] Creating virtual environment
    %PY% -m venv .venv
    if errorlevel 1 goto fail
)
set "VPY=.venv\Scripts\python.exe"

echo [..] Upgrading pip
"%VPY%" -m pip install --upgrade pip --quiet

nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [ok] NVIDIA GPU detected - installing CUDA build of PyTorch
    "%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu124
) else (
    echo [!!] No NVIDIA GPU detected - installing CPU build ^(slower^)
    "%VPY%" -m pip install torch
)
if errorlevel 1 goto fail

echo [..] Installing remaining dependencies
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo.
"%VPY%" -c "import torch;print('[ok] torch',torch.__version__,'CUDA' if torch.cuda.is_available() else 'CPU',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

if not exist ".env" (
    echo.
    echo [!!] No .env file found.
    echo      Create one containing your Hugging Face token:
    echo          HF_PAT = "hf_xxxxxxxxxxxxxxxxx"
    echo      and accept the model terms at
    echo          https://huggingface.co/ARTPARK-IISc/SraVaani-1.0
    echo.
)

echo [..] Downloading the SraVaani-1.0 model ^(about 900 MB, one time^)
"%VPY%" -m sravaani_flow.fetch
if errorlevel 1 goto fail

echo.
echo [..] Running self-test
"%VPY%" selftest.py
if errorlevel 1 (
    echo.
    echo [!!] Self-test reported failures - see above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete. Launch with run.bat
echo ============================================================
pause
exit /b 0

:fail
echo.
echo [!!] Setup failed. See the messages above.
pause
exit /b 1
