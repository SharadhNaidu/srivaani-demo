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
    echo      During install, tick "Add python.exe to PATH".
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

echo.
echo [..] Checking for an NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%G in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do (
        echo [ok] Found: %%G
    )
    echo [..] Installing PyTorch 2.6.0 with CUDA 12.4
    "%VPY%" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
) else (
    echo [!!] No NVIDIA GPU detected - installing the CPU build.
    echo      The app will still work, but transcription will be slower.
    "%VPY%" -m pip install torch==2.6.0
)
if errorlevel 1 goto fail

echo.
echo [..] Installing the remaining pinned dependencies
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo.
echo [..] Verifying the install
"%VPY%" -c "import torch, transformers, sentencepiece, sounddevice, soundfile, scipy, numpy, noisereduce, webrtcvad, pynput, pyperclip, win32api; print('[ok] all imports fine')"
if errorlevel 1 (
    echo [!!] An import failed. If it was a pywin32 error, run:
    echo        .venv\Scripts\python.exe .venv\Scripts\pywin32_postinstall.py -install
    goto fail
)

"%VPY%" -c "import torch; ok=torch.cuda.is_available(); print('[ok] torch',torch.__version__,('CUDA '+torch.cuda.get_device_name(0)) if ok else 'CPU only')"

if not exist ".env" (
    echo.
    echo ============================================================
    echo  [!!] No .env file found - the model download needs one.
    echo.
    echo   1. Open https://huggingface.co/ARTPARK-IISc/SraVaani-1.0
    echo      sign in, and accept the model terms.
    echo   2. Create a READ token at
    echo      https://huggingface.co/settings/tokens
    echo   3. Create a file called .env next to this script with:
    echo          HF_PAT = "hf_your_token_here"
    echo   4. Run setup.bat again.
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo [..] Downloading the SraVaani-1.0 model ^(about 900 MB, one time^)
"%VPY%" -m sravaani_flow.fetch
if errorlevel 1 goto fail

echo.
echo [..] Running the self-test
"%VPY%" selftest.py
if errorlevel 1 (
    echo.
    echo [!!] Self-test reported failures - see the named check above.
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
