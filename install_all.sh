#!/bin/bash
# ============================================
# Dream Weaver 一键安装脚本
# ============================================
# 使用方法：chmod +x install_all.sh && ./install_all.sh
# ============================================

set -e  # 遇到错误立即退出

echo "========================================="
echo "Dream Weaver 项目一键安装"
echo "========================================="
echo ""

# 检查 Python 版本
echo "[1/5] 检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误：未找到 Python3，请先安装 Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python 版本: $(python3 --version)"

# 检查 Node.js 版本
echo ""
echo "[2/5] 检查 Node.js 环境..."
if ! command -v node &> /dev/null; then
    echo "⚠️  警告：未找到 Node.js，前端将无法构建"
    echo "   请访问 https://nodejs.org/ 安装 Node.js 18+"
else
    echo "✅ Node.js 版本: $(node --version)"
fi

# 安装 Python 依赖
echo ""
echo "[3/5] 安装 Python 依赖..."
if [ -f "requirements.txt" ]; then
    echo "正在安装 Python 包（这可能需要几分钟）..."
    pip3 install -r requirements.txt
    echo "✅ Python 依赖安装完成"
else
    echo "❌ 错误：未找到 requirements.txt"
    exit 1
fi

# 安装前端依赖
echo ""
echo "[4/5] 安装前端依赖..."
if [ -d "dream_weaver" ] && [ -f "dream_weaver/package.json" ]; then
    cd dream_weaver
    if command -v npm &> /dev/null; then
        echo "正在安装 Node.js 包..."
        npm install
        echo "✅ 前端依赖安装完成"
    else
        echo "⚠️  警告：未找到 npm，跳过前端依赖安装"
    fi
    cd ..
else
    echo "⚠️  警告：未找到前端项目目录"
fi

# 检查 Redis
echo ""
echo "[5/5] 检查 Redis..."
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "✅ Redis 服务正在运行"
    else
        echo "⚠️  警告：Redis 已安装但未运行"
        echo "   请运行: redis-server 或使用 Docker: docker run -d -p 6379:6379 redis"
    fi
else
    echo "⚠️  警告：未找到 Redis"
    echo "   可选：安装 Redis 以启用缓存功能"
    echo "   Ubuntu/Debian: sudo apt-get install redis-server"
    echo "   Mac: brew install redis"
    echo "   Docker: docker run -d -p 6379:6379 redis"
fi

echo ""
echo "========================================="
echo "✅ 安装完成！"
echo "========================================="
echo ""
echo "下一步："
echo "1. 配置 .env 文件（设置 DASHSCOPE_API_KEY 等）"
echo "2. 启动后端: python api/main.py"
echo "3. 启动前端: cd dream_weaver && npm run dev"
echo ""

