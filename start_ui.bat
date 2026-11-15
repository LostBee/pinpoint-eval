@echo off
cd /d "%~dp0"
echo Starting Pinpoint Web UI at http://127.0.0.1:8000 ...
".venv\Scripts\pinpoint.exe" serve
echo.
echo Server exited with code %ERRORLEVEL%.
pause
