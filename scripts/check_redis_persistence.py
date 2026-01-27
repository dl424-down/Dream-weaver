#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 Redis 持久化配置
"""

import redis

def main():
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        
        print("=" * 60)
        print("Redis 持久化配置检查")
        print("=" * 60)
        print()
        
        # 检查持久化配置
        persistence_info = r.info('persistence')
        
        print("📋 当前持久化状态:")
        print()
        
        # RDB 配置
        rdb_status = persistence_info.get('rdb_last_bgsave_status', 'unknown')
        rdb_enabled = persistence_info.get('rdb_changes_since_last_save', None) is not None
        print(f"RDB 持久化:")
        print(f"  状态: {rdb_status}")
        print(f"  是否启用: {'是' if rdb_enabled else '否'}")
        
        # AOF 配置
        aof_enabled = persistence_info.get('aof_enabled', 0)
        print(f"AOF 持久化:")
        print(f"  是否启用: {'是' if aof_enabled == 1 else '否'}")
        print()
        
        # 获取配置
        try:
            save_config = r.config_get('save')
            print(f"RDB 保存策略: {save_config.get('save', '未配置')}")
        except:
            pass
        
        try:
            data_dir = r.config_get('dir')
            print(f"数据目录: {data_dir.get('dir', '默认')}")
        except:
            pass
        
        print()
        print("=" * 60)
        print("⚠️  重要说明:")
        print("=" * 60)
        
        if rdb_status == 'ok' or aof_enabled == 1:
            print("✅ 已配置持久化，数据会保存到磁盘")
            print("   重启后数据不会丢失")
        else:
            print("❌ 未配置持久化，数据只存在内存中")
            print("   重启后所有缓存数据会丢失")
            print()
            print("💡 解决方案:")
            print("   1. 使用 Docker 启动 Redis 时挂载数据卷")
            print("   2. 配置 Redis 的 RDB 或 AOF 持久化")
            print("   3. 或者接受数据在重启后丢失（缓存本身有7天过期时间）")
        
        print()
        
    except redis.ConnectionError:
        print("❌ 无法连接到 Redis 服务")
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()


