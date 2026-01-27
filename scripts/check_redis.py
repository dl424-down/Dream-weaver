#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis 运行状态检查脚本
可以直接运行此脚本来检查 Redis 是否可用
"""

import sys
import os

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from db.redis_cache import get_cache

def main():
    """主函数"""
    print("=" * 50)
    print("Redis 运行状态检查")
    print("=" * 50)
    print()
    
    # 获取缓存实例
    cache = get_cache()
    
    # 检查 Redis Python 库
    try:
        import redis
        print("✅ Redis Python 库已安装")
    except ImportError:
        print("❌ Redis Python 库未安装")
        print("   请运行: pip install redis")
        print()
        sys.exit(1)
    
    print()
    
    # 检查 Redis 连接状态
    print("正在检查 Redis 连接...")
    print()
    
    status = cache.get_status()
    
    # 显示结果
    if status["redis_library_installed"]:
        print("✅ Redis Python 库: 已安装")
    else:
        print("❌ Redis Python 库: 未安装")
    
    if status["cache_enabled"]:
        print("✅ 缓存功能: 已启用")
    else:
        print("⚠️  缓存功能: 未启用")
    
    if status["redis_available"]:
        print("✅ Redis 服务: 正在运行")
        
        if status.get("connection_info"):
            conn = status["connection_info"]
            print(f"   连接信息: {conn['host']}:{conn['port']} (DB {conn['db']})")
        
        if status.get("redis_info"):
            info = status["redis_info"]
            print(f"   Redis 版本: {info.get('version', 'unknown')}")
            print(f"   内存使用: {info.get('used_memory_human', 'unknown')}")
            print(f"   连接客户端数: {info.get('connected_clients', 0)}")
        
        # 测试读写
        print()
        print("正在测试缓存读写...")
        try:
            test_key = "test:connection"
            test_value = "test_value"
            cache.redis_client.setex(test_key, 10, test_value)
            result = cache.redis_client.get(test_key)
            if result == test_value:
                print("✅ 缓存读写: 正常")
                cache.redis_client.delete(test_key)
            else:
                print("⚠️  缓存读写: 异常")
        except Exception as e:
            print(f"⚠️  缓存读写测试失败: {e}")
        
        print()
        print("=" * 50)
        print("✅ Redis 运行正常，缓存功能可用")
        print("=" * 50)
        sys.exit(0)
    else:
        print("❌ Redis 服务: 未运行或无法连接")
        
        if status.get("error"):
            print(f"   错误信息: {status['error']}")
        
        print()
        print("=" * 50)
        print("❌ Redis 未运行，缓存功能已禁用")
        print("=" * 50)
        print()
        print("解决方案:")
        print("1. 如果使用 Docker:")
        print("   docker run -d --name redis-dream -p 6379:6379 redis")
        print()
        print("2. 如果使用本地安装:")
        print("   Windows (WSL): sudo service redis-server start")
        print("   Linux: sudo systemctl start redis-server")
        print("   Mac: brew services start redis")
        print()
        print("3. 检查 Redis 是否在运行:")
        print("   Windows: netstat -an | findstr :6379")
        print("   Linux/Mac: redis-cli ping")
        print()
        sys.exit(1)

if __name__ == "__main__":
    main()


