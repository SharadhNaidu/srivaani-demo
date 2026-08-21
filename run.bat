@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo.
    echo   setup has not been run yet.
    echo   run setup.bat first, then try again.
    echo.
    pause
    exit /b 1
)

start "" ".venv\Scripts\pythonw.exe" main.py

rem give the app a moment; if it died immediately, show why
timeout /t 3 /nobreak >nul
tasklist /fi "imagename eq pythonw.exe" 2>nul | find /i "pythonw.exe" >nul
if errorlevel 1 (
    echo.
    echo   the app did not start. running it again to show the error:
    echo.
    ".venv\Scripts\python.exe" main.py
    echo.
    pause
)
exit /b 0
