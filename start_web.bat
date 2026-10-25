@echo off
chcp 65001 >nul
cd /d D:\xiangqi-vision-assistant
title 象棋视觉助手

rem 服务已在运行就直接打开页面，避免重复启动导致端口冲突
curl -s -o nul --max-time 2 http://127.0.0.1:8765/api/state
if %errorlevel%==0 (
    echo 服务已在运行，正在打开界面…
    start "" http://127.0.0.1:8765/
    timeout /t 1 >nul
    exit /b
)

echo ============================================
echo   象棋视觉助手  -  正在启动
echo   界面会自动在浏览器打开
echo   关闭本窗口即停止服务
echo ============================================
D:\Python312\python.exe -u webapp\server.py
echo.
echo 服务已停止，按任意键关闭。
pause >nul
