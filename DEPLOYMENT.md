# Dream Weaver 项目部署指南

## 📋 目录
1. [系统要求](#系统要求)
2. [一键安装](#一键安装)
3. [手动安装](#手动安装)
4. [环境配置](#环境配置)
5. [启动服务](#启动服务)
6. [生产环境部署](#生产环境部署)

---

## 系统要求

### 必需
- **Python 3.8+** （推荐 3.10+）
- **Node.js 18+** （用于前端构建）
- **Redis** （可选，用于缓存功能）

### 推荐配置
- **内存**: 4GB+ （如果使用 BLIP 模型，建议 8GB+）
- **存储**: 10GB+ （用于模型文件和依赖）
- **CPU**: 2核+ （如果使用 BLIP 模型，建议 4核+）

---

## 一键安装

### Linux/macOS
```bash
chmod +x install_all.sh
./install_all.sh
```

### Windows
```cmd
install_all.bat
```

---

## 手动安装

### 1. 安装 Python 依赖

#### 完整版本（包含 BLIP 模型）
```bash
pip install -r requirements.txt
```

#### 最小版本（不含 AI 模型，使用演示模式）
```bash
pip install -r requirements_minimal.txt
```

### 2. 安装前端依赖
```bash
cd dream_weaver
npm install
cd ..
```

### 3. 安装 Redis（可选）

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis-server
```

#### macOS
```bash
brew install redis
brew services start redis
```

#### Docker
```bash
docker run -d --name redis-dream -p 6379:6379 redis
```

---

## 环境配置

### 1. 创建 `.env` 文件

在项目根目录创建 `.env` 文件：

```env
# DashScope API Key（必需，用于 LLM 分析）
DASHSCOPE_API_KEY=your_api_key_here

# Redis 配置（可选，如果使用 Redis 缓存）
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# 后端服务配置（可选）
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
```

### 2. 获取 DashScope API Key

1. 访问 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/)
2. 注册/登录账号
3. 创建 API Key
4. 将 API Key 填入 `.env` 文件

---

## 启动服务

### 开发环境

#### 1. 启动后端
```bash
python api/main.py
```
后端将在 `http://localhost:8000` 启动

#### 2. 启动前端（新终端）
```bash
cd dream_weaver
npm run dev
```
前端将在 `http://localhost:5173` 启动

### 生产环境

#### 1. 构建前端
```bash
cd dream_weaver
npm run build
cd ..
```
构建产物在 `dream_weaver/dist/` 目录

#### 2. 使用 Gunicorn 启动后端（推荐）
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 api.main:app
```

#### 3. 使用 systemd 管理服务（Linux）

创建 `/etc/systemd/system/dream-weaver.service`:
```ini
[Unit]
Description=Dream Weaver API
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/your/project
Environment="PATH=/path/to/your/venv/bin"
ExecStart=/path/to/your/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 api.main:app

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl start dream-weaver
sudo systemctl enable dream-weaver
```

---

## 生产环境部署

### 使用 Nginx 反向代理

#### 1. 安装 Nginx
```bash
sudo apt-get install nginx
```

#### 2. 配置 Nginx

编辑 `/etc/nginx/sites-available/dream-weaver`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 前端静态文件
    location / {
        root /path/to/your/project/dream_weaver/dist;
        try_files $uri $uri/ /index.html;
    }

    # 后端 API
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # 文件上传大小限制
    client_max_body_size 10M;
}
```

#### 3. 启用配置
```bash
sudo ln -s /etc/nginx/sites-available/dream-weaver /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 使用 SSL 证书（Let's Encrypt）

```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 使用 Docker Compose（推荐）

创建 `docker-compose.yml`:
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - redis
    volumes:
      - ./data:/app/data

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "80:80"
    depends_on:
      - backend

volumes:
  redis_data:
```

---

## 常见问题

### 1. BLIP 模型加载失败
- 检查网络连接（首次运行需要下载模型）
- 确保有足够的磁盘空间（模型约 1GB）
- 如果不需要 BLIP，使用 `requirements_minimal.txt`

### 2. Redis 连接失败
- 检查 Redis 是否运行：`redis-cli ping`
- 检查防火墙设置
- 如果不需要缓存，可以不安装 Redis

### 3. 前端构建失败
- 确保 Node.js 版本 >= 18
- 删除 `node_modules` 和 `package-lock.json`，重新安装
- 检查网络连接（npm 需要下载包）

### 4. API 调用失败
- 检查 `.env` 文件中的 `DASHSCOPE_API_KEY` 是否正确
- 检查 API Key 是否有效
- 查看后端日志获取详细错误信息

---

## 依赖说明

### Python 依赖
- **完整版**: `requirements.txt` - 包含所有功能
- **最小版**: `requirements_minimal.txt` - 不含 BLIP 模型

### Node.js 依赖
- **前端**: `dream_weaver/package.json`
- **根目录**: `package.json`（用于统一管理）

---

## 更新依赖

### Python
```bash
pip install --upgrade -r requirements.txt
```

### Node.js
```bash
cd dream_weaver
npm update
```

---

## 技术支持

如有问题，请查看：
- [安装指南](docs/install_guide.md)
- [系统架构说明](docs/项目架构说明.md)
- [API 文档](docs/api模块详细设计文档.md)

