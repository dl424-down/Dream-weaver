#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看 Redis 缓存存储信息
"""

import redis
import json

def main():
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
        
        print("=" * 60)
        print("Redis 缓存存储信息")
        print("=" * 60)
        print()
        
        # 1. 基本信息
        info = r.info()
        print("📊 Redis 基本信息:")
        print(f"   版本: {info.get('redis_version', 'unknown')}")
        print(f"   运行模式: {info.get('redis_mode', 'unknown')}")
        print(f"   运行时间: {info.get('uptime_in_days', 0)} 天")
        print()
        
        # 2. 内存使用
        print("💾 内存使用:")
        print(f"   已使用内存: {info.get('used_memory_human', 'unknown')}")
        print(f"   内存峰值: {info.get('used_memory_peak_human', 'unknown')}")
        print()
        
        # 3. 数据存储位置
        print("📁 数据存储位置:")
        try:
            config = r.config_get('dir')
            data_dir = config.get('dir', '默认目录')
            print(f"   数据目录: {data_dir}")
        except:
            print("   数据目录: (无法获取)")
        
        # 持久化配置
        try:
            persistence_info = r.info('persistence')
            rdb_enabled = persistence_info.get('rdb_last_bgsave_status', 'unknown')
            aof_enabled = persistence_info.get('aof_enabled', 0)
            print(f"   RDB 持久化: {'启用' if rdb_enabled == 'ok' else '未启用'}")
            print(f"   AOF 持久化: {'启用' if aof_enabled == 1 else '未启用'}")
        except:
            print("   持久化配置: (无法获取)")
        print()
        
        # 4. 当前缓存数据
        print("🗂️  当前缓存数据:")
        try:
            all_keys = r.keys('*')
            dream_analysis_keys = r.keys('dream:analysis:*')
            dream_image_keys = r.keys('dream:image_gen:*')
            dream_history_keys = r.keys('dream:history:*')
            dream_detail_keys = r.keys('dream:detail:*')
            dream_comprehensive_keys = r.keys('dream:comprehensive:*')
            
            print(f"   总键数量: {len(all_keys)}")
            print(f"   梦境分析缓存: {len(dream_analysis_keys)} 个")
            print(f"   图像生成缓存: {len(dream_image_keys)} 个")
            print(f"   历史记录缓存: {len(dream_history_keys)} 个")
            print(f"   详情缓存: {len(dream_detail_keys)} 个")
            print(f"   综合分析缓存: {len(dream_comprehensive_keys)} 个")
            print()
            
            dream_keys = dream_analysis_keys
            
            if dream_keys:
                print("   示例缓存键:")
                for key in dream_keys[:3]:
                    try:
                        ttl = r.ttl(key)
                        try:
                            value_preview = r.get(key)
                        except UnicodeDecodeError:
                            # 如果解码失败，使用二进制模式读取
                            r_binary = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
                            value_bytes = r_binary.get(key)
                            if value_bytes:
                                try:
                                    value_preview = value_bytes.decode('utf-8', errors='ignore')
                                except:
                                    value_preview = None
                            else:
                                value_preview = None
                        
                        if value_preview:
                            try:
                                data = json.loads(value_preview)
                                preview = str(data)[:80] + "..." if len(str(data)) > 80 else str(data)
                            except json.JSONDecodeError:
                                # 如果不是 JSON，直接显示字符串（处理编码问题）
                                try:
                                    preview = value_preview[:80] + "..." if len(value_preview) > 80 else value_preview
                                except:
                                    preview = "(无法解码)"
                            except Exception:
                                preview = "(解析失败)"
                        else:
                            preview = "(空)"
                        
                        ttl_str = f"{ttl}秒" if ttl > 0 else "永久" if ttl == -1 else "已过期"
                        print(f"     - {key}")
                        print(f"       过期时间: {ttl_str}")
                        print(f"       内容预览: {preview}")
                        print()
                    except Exception as e:
                        print(f"     - {key} (读取失败: {e})")
                        print()
            else:
                print("   (暂无缓存数据)")
                print()
        except Exception as e:
            print(f"   (无法读取缓存数据: {e})")
            print()
        
        # 5. 存储说明
        print("=" * 60)
        print("💡 存储说明:")
        print("=" * 60)
        print("1. Redis 是内存数据库，数据主要存储在内存中")
        print("2. 缓存键格式: dream:analysis:{text_only|with_image}:{文本MD5哈希}")
        print("3. 缓存内容: JSON 格式的分析结果")
        print("4. 过期时间: 7天 (604800秒)")
        print("5. 数据持久化:")
        print("   - 如果配置了 RDB/AOF，数据会保存到磁盘")
        print("   - 默认情况下，Redis 重启后内存中的数据会丢失")
        print("   - 持久化文件通常保存在 Redis 数据目录中")
        print()
        
    except redis.ConnectionError:
        print("❌ 无法连接到 Redis 服务")
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()

