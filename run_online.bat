@echo off
REM ============================================================
REM  Cricket Attack - play online with friends via Cloudflare
REM  Double-click this file (or run it in a terminal).
REM  It starts the game server, then opens a public HTTPS tunnel.
REM  Share the https://XXXX.trycloudflare.com URL it prints.
REM ============================================================
cd /d "%~dp0"

echo Starting Cricket Attack server on http://localhost:8000 ...
start "Cricket Attack Server" cmd /k python src\server.py

echo Waiting for the server to boot...
timeout /t 4 >nul

echo.
echo ============================================================
echo  Opening Cloudflare tunnel. Share the trycloudflare.com URL
echo  below with your friends. Keep THIS window open while playing.
echo ============================================================
echo.

"C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8000
