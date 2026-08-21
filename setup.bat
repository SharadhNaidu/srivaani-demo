@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ================================================================
echo   SraVaani Flow - setup
echo ================================================================
echo.
echo   this installs everything and downloads the speech model.
echo   no account, no login, no token needed.
echo   it takes 5-15 minutes the first time. safe to re-run.
echo.

rem ---------------------------------------------------------------- python
set "PY="
for %%V in (3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1
        if !errorlevel! equ 0 set "PY=py -%%V"
    )
)
if not defined PY (
    for %%C in (python python3) do (
        if not defined PY (
            %%C -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<(3,13) else 1)" >nul 2>&1
            if !errorlevel! equ 0 set "PY=%%C"
        )
    )
)

if not defined PY (
    echo   [FAIL] no suitable python found.
    echo.
    python --version 2>nul
    if !errorlevel! equ 0 (
        echo   you have python installed, but this project needs 3.10, 3.11 or 3.12.
        echo   pytorch does not publish wheels for 3.13 or newer yet.
    )
    echo.
    echo   install python 3.12 from:
    echo       https://www.python.org/downloads/release/python-3128/
    echo   during install, tick "Add python.exe to PATH", then run setup.bat again.
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%G in ('%PY% -c "import sys;print(sys.version.split()[0])"') do set "PYVER=%%G"
echo   [ok]   python !PYVER!

rem ---------------------------------------------------------------- venv
if not exist ".venv\Scripts\python.exe" (
    echo   [..]   creating virtual environment
    %PY% -m venv .venv
    if errorlevel 1 (
        echo   [FAIL] could not create the virtual environment.
        pause
        exit /b 1
    )
)
set "VPY=.venv\Scripts\python.exe"
echo   [ok]   virtual environment ready

rem ---------------------------------------------------------------- network
"%VPY%" -c "import urllib.request;urllib.request.urlopen('https://huggingface.co',timeout=15)" >nul 2>&1
if errorlevel 1 (
    echo   [FAIL] cannot reach huggingface.co
    echo          check your internet connection and run setup.bat again.
    pause
    exit /b 1
)
echo   [ok]   internet reachable

echo   [..]   upgrading pip
"%VPY%" -m pip install --upgrade pip --quiet --disable-pip-version-check

rem ---------------------------------------------------------------- torch
echo.
echo   looking for an nvidia gpu...
set "HASGPU="
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 set "HASGPU=1"

if defined HASGPU (
    for /f "tokens=*" %%G in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do echo   [ok]   found %%G
    echo   [..]   installing pytorch 2.6.0 with cuda 12.4  ^(about 2.5 GB^)
    "%VPY%" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --disable-pip-version-check
    if errorlevel 1 (
        echo   [!!]   cuda install failed - falling back to the cpu build
        "%VPY%" -m pip install torch==2.6.0 --disable-pip-version-check
    )
) else (
    echo   [ok]   no nvidia gpu - installing the cpu build ^(about 250 MB^)
    echo          this is fine: on cpu it still runs ~10x faster than real time.
    "%VPY%" -m pip install torch==2.6.0 --disable-pip-version-check
)
if errorlevel 1 (
    echo   [FAIL] could not install pytorch.
    pause
    exit /b 1
)

rem ---------------------------------------------------------------- deps
echo.
echo   [..]   installing the remaining dependencies
"%VPY%" -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 (
    echo   [FAIL] could not install the dependencies.
    pause
    exit /b 1
)

rem ---------------------------------------------------------------- model
echo.
echo   [..]   downloading the speech model ^(about 870 MB, one time^)
"%VPY%" -m sravaani_flow.fetch
if errorlevel 1 (
    echo   [FAIL] the model download did not finish.
    echo          re-run setup.bat - it resumes where it left off.
    pause
    exit /b 1
)

rem ---------------------------------------------------------------- verify
"%VPY%" verify.py
if errorlevel 1 (
    echo.
    echo   setup did not complete. see the [FAIL] lines above.
    pause
    exit /b 1
)

echo.
echo   start the app with:  run.bat
echo.
pause
exit /b 0
