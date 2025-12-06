@echo off
REM ============================================
REM Dream Weaver 一键安装脚本 (Windows)
REM ============================================
REM 使用方法：双击运行或在命令行执行 install_all.bat
REM ============================================

echo =========================================
echo Dream Weaver 项目一键安装
echo =========================================
echo.

REM 检查 Python
echo [1/5] 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)
python --version
echo [OK] Python 已安装

REM 检查 Node.js
echo.
echo [2/5] 检查 Node.js 环境...
node --version >nul 2>&1
if errorlevel 1 (
    echo [警告] 未找到 Node.js，前端将无法构建
    echo 请访问 https://nodejs.org/ 安装 Node.js 18+
) else (
    node --version
    echo [OK] Node.js 已安装
)

REM 安装 Python 依赖
echo.
echo [3/5] 安装 Python 依赖...
if exist requirements.txt (
    echo 正在安装 Python 包（这可能需要几分钟）...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] Python 依赖安装失败
        pause
        exit /b 1
    )
    echo [OK] Python 依赖安装完成
) else (
    echo [错误] 未找到 requirements.txt
    pause
    exit /b 1
)

REM 安装前端依赖
echo.
echo [4/5] 安装前端依赖...
if exist dream_weaver\package.json (
    cd dream_weaver
    if exist node_modules (
        echo 前端依赖已存在，跳过安装
    ) else (
        if exist "C:\Program Files\nodejs\npm.cmd" (
            echo 正在安装 Node.js 包...
            call npm install
            if errorlevel 1 (
                echo [警告] 前端依赖安装失败
            ) else (
                echo [OK] 前端依赖安装完成
            )
        ) else (
            echo [警告] 未找到 npm，跳过前端依赖安装
        )
    )
    cd ..
) else (
    echo [警告] 未找到前端项目目录
)

REM 检查 Redis
echo.
echo [5/5] 检查 Redis...
redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo [警告] Redis 未运行或未安装
    echo 可选：安装 Redis 以启用缓存功能
    echo Windows: 使用 WSL 或 Docker Desktop
    echo Docker: docker run -d -p 6379:6379 redis
) else (
    echo [OK] Redis 服务正在运行
)

echo.
echo =========================================
echo [OK] 安装完成！
echo =========================================
echo.
echo 下一步：
echo 1. 配置 .env 文件（设置 DASHSCOPE_API_KEY 等）
echo 2. 启动后端: python api\main.py
echo 3. 启动前端: cd dream_weaver ^&^& npm run dev
echo.
pause

