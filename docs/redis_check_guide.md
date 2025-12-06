# Redis 运行状态检查指南

## 方法一：通过代码自动检查（推荐）

代码已经内置了自动检查功能。当你启动 API 服务时，会自动尝试连接 Redis：

- **如果连接成功**：会看到日志 `[缓存] Redis 连接成功 (host=localhost, port=6379, db=0)`
- **如果连接失败**：会看到警告 `[警告] Redis 连接失败: ...，缓存功能已禁用`

### 通过 API 检查状态

启动服务后，访问以下端点查看 Redis 状态：

```bash
GET http://localhost:8000/cache/status
```

返回示例：
```json
{
  "success": true,
  "cache": {
    "redis_library_installed": true,
    "cache_enabled": true,
    "redis_available": true,
    "connection_info": {
      "host": "localhost",
      "port": 6379,
      "db": 0
    },
    "redis_info": {
      "version": "7.0.0",
      "used_memory_human": "1.2M",
      "connected_clients": 1
    }
  }
}
```

### 在 Python 代码中检查

```python
from db.redis_cache import get_cache

cache = get_cache()

# 方法1：简单检查
if cache.is_available():
    print("Redis 可用")
else:
    print("Redis 不可用")

# 方法2：获取详细状态
status = cache.get_status()
print(status)
```

## 方法二：命令行检查

### Windows PowerShell

```powershell
# 检查 Redis 进程是否运行
Get-Process redis-server -ErrorAction SilentlyContinue

# 或者检查端口是否被占用
netstat -an | findstr :6379

# 如果使用 Docker
docker ps | findstr redis
```

### Linux/Mac

```bash
# 检查 Redis 进程
ps aux | grep redis

# 检查端口
netstat -an | grep 6379
# 或
lsof -i :6379

# 如果使用 Docker
docker ps | grep redis
```

### 使用 Redis CLI 测试连接

如果已安装 Redis CLI：

```bash
# 连接 Redis（默认 localhost:6379）
redis-cli ping

# 如果返回 PONG，说明 Redis 正在运行
# 如果连接失败，会显示错误信息

# 指定主机和端口
redis-cli -h localhost -p 6379 ping

# 如果有密码
redis-cli -h localhost -p 6379 -a your_password ping
```

## 方法三：使用 Python 脚本快速检查

创建一个简单的检查脚本：

```python
# check_redis.py
import redis
import sys

try:
    r = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=3)
    r.ping()
    print("✅ Redis 正在运行")
    print(f"Redis 版本: {r.info()['redis_version']}")
    sys.exit(0)
except redis.ConnectionError:
    print("❌ Redis 未运行或无法连接")
    sys.exit(1)
except ImportError:
    print("❌ Redis Python 库未安装，请运行: pip install redis")
    sys.exit(1)
except Exception as e:
    print(f"❌ 检查失败: {e}")
    sys.exit(1)
```

运行：
```bash
python check_redis.py
```

## 常见问题

### 1. Redis 未安装

**Windows:**
- 使用 WSL (Windows Subsystem for Linux) 安装 Redis
- 或使用 Docker: `docker run -d -p 6379:6379 redis`

**Linux:**
```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis-server
```

**Mac:**
```bash
brew install redis
brew services start redis
```

### 2. Redis 在运行但连接失败

检查以下几点：
- Redis 是否监听在正确的端口（默认 6379）
- 防火墙是否阻止了连接
- Redis 配置是否允许外部连接（检查 `bind` 配置）
- 是否需要密码（检查 `.env` 文件中的 `REDIS_PASSWORD`）

### 3. 查看 Redis 配置

```bash
# 查看 Redis 配置文件位置
redis-cli CONFIG GET dir

# 查看监听地址
redis-cli CONFIG GET bind

# 查看端口
redis-cli CONFIG GET port
```

## 快速启动 Redis（Docker）

如果不想安装 Redis，可以使用 Docker：

```bash
# 启动 Redis 容器
docker run -d --name redis-dream -p 6379:6379 redis

# 检查容器状态
docker ps

# 停止容器
docker stop redis-dream

# 启动已存在的容器
docker start redis-dream
```

## 总结

最简单的方法：
1. 启动你的 API 服务
2. 查看启动日志，看是否有 `[缓存] Redis 连接成功` 的消息
3. 或访问 `http://localhost:8000/cache/status` 查看详细状态

如果 Redis 未运行，系统会自动禁用缓存功能，但不会影响其他功能的正常使用。


