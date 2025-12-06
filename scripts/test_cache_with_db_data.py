#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用数据库中的梦境文本测试缓存功能
"""

import sys
import os
import time
import requests

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from db.database import get_recent_entries

def test_cache_with_db_data():
    """使用数据库中的梦境文本测试缓存"""
    print("=" * 60)
    print("使用数据库中的梦境文本测试缓存功能")
    print("=" * 60)
    print()
    
    # 1. 从数据库获取最近的梦境记录
    print("📖 从数据库获取梦境记录...")
    entries = get_recent_entries(limit=5)
    
    if not entries:
        print("❌ 数据库中没有梦境记录，无法测试")
        return
    
    print(f"✅ 找到 {len(entries)} 条记录")
    print()
    
    # 2. 选择一条记录进行测试
    test_entry = entries[0]
    dream_text = test_entry.get('dream_text', '')
    entry_id = test_entry.get('id')
    
    if not dream_text:
        print("❌ 选中的记录没有梦境文本")
        return
    
    print(f"📝 测试记录 ID: {entry_id}")
    print(f"📝 梦境文本: {dream_text[:100]}{'...' if len(dream_text) > 100 else ''}")
    print()
    
    # 3. 第一次请求（应该缓存未命中，执行分析）
    print("=" * 60)
    print("🔄 第一次请求（应该缓存未命中）")
    print("=" * 60)
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8000/analyze",
            data={"dream_text": dream_text},
            timeout=60
        )
        first_duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 请求成功")
            print(f"⏱️  耗时: {first_duration:.2f} 秒")
            print(f"📊 返回结果包含: {list(result.keys())}")
            print()
            
            # 检查后端日志（通过观察响应时间判断）
            if first_duration > 5:
                print("💡 提示: 耗时较长，可能执行了 AI 分析（缓存未命中）")
            else:
                print("💡 提示: 耗时较短，可能使用了缓存（缓存命中）")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"   响应: {response.text}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到后端服务器 (http://localhost:8000)")
        print("   请确保后端服务器正在运行")
        return
    except Exception as e:
        print(f"❌ 请求出错: {e}")
        return
    
    print()
    print("⏸️  等待 2 秒后再次请求...")
    time.sleep(2)
    print()
    
    # 4. 第二次请求（应该缓存命中）
    print("=" * 60)
    print("🔄 第二次请求（应该缓存命中）")
    print("=" * 60)
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8000/analyze",
            data={"dream_text": dream_text},
            timeout=60
        )
        second_duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 请求成功")
            print(f"⏱️  耗时: {second_duration:.2f} 秒")
            print(f"📊 返回结果包含: {list(result.keys())}")
            print()
            
            # 比较两次请求的耗时
            if second_duration < first_duration * 0.5:
                print("✅ 缓存测试成功！")
                print(f"   第一次耗时: {first_duration:.2f} 秒")
                print(f"   第二次耗时: {second_duration:.2f} 秒")
                print(f"   性能提升: {(1 - second_duration/first_duration)*100:.1f}%")
                print()
                print("💡 说明: 第二次请求明显更快，说明缓存命中成功")
            else:
                print("⚠️  缓存可能未生效")
                print(f"   第一次耗时: {first_duration:.2f} 秒")
                print(f"   第二次耗时: {second_duration:.2f} 秒")
                print()
                print("💡 提示: 请查看后端服务器日志，确认是否有 '[缓存] 使用缓存的分析结果' 消息")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"   响应: {response.text}")
    except Exception as e:
        print(f"❌ 请求出错: {e}")
    
    print()
    print("=" * 60)
    print("📋 测试完成")
    print("=" * 60)
    print()
    print("💡 提示:")
    print("   1. 查看后端服务器日志，确认缓存命中情况")
    print("   2. 运行 'python scripts/show_redis_storage.py' 查看缓存内容")
    print("   3. 运行 'python scripts/check_redis.py' 检查 Redis 状态")

if __name__ == "__main__":
    test_cache_with_db_data()

