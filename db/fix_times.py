#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复数据库中旧记录的时间
将 UTC 时间转换为中国时区（UTC+8）时间
"""

import os
import sqlite3
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "dreams.db")

def fix_old_times():
    """修复旧记录的时间，将 UTC 时间转换为中国时区时间"""
    china_tz = timezone(timedelta(hours=8))
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT id, created_at FROM dream_entries")
        rows = cursor.fetchall()
        
        updated_count = 0
        for row_id, old_time in rows:
            if not old_time:
                continue
            
            try:
                # 尝试解析时间
                # 如果是 UTC 时间格式（带 T 或 Z），转换为中国时区
                if 'T' in old_time or old_time.endswith('Z'):
                    # ISO 格式，可能是 UTC
                    if old_time.endswith('Z'):
                        dt = datetime.fromisoformat(old_time.replace('Z', '+00:00'))
                    else:
                        # 尝试解析为 UTC
                        dt = datetime.fromisoformat(old_time.replace(' ', 'T'))
                        if dt.tzinfo is None:
                            # 假设是 UTC 时间
                            dt = dt.replace(tzinfo=timezone.utc)
                    
                    # 转换为中国时区
                    china_time = dt.astimezone(china_tz).strftime("%Y-%m-%d %H:%M:%S")
                elif old_time.count(':') == 2 and ' ' in old_time:
                    # 格式 "YYYY-MM-DD HH:MM:SS"，检查是否是 UTC 时间
                    # 如果时间看起来是 UTC（比如凌晨时间），可能需要加 8 小时
                    dt = datetime.strptime(old_time, "%Y-%m-%d %H:%M:%S")
                    # 假设旧记录是 UTC 时间，转换为中国时区
                    dt_utc = dt.replace(tzinfo=timezone.utc)
                    china_time = dt_utc.astimezone(china_tz).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    # 无法解析，跳过
                    print(f"[跳过] ID {row_id}: 无法解析时间格式 '{old_time}'")
                    continue
                
                # 更新数据库
                conn.execute(
                    "UPDATE dream_entries SET created_at = ? WHERE id = ?",
                    (china_time, row_id)
                )
                updated_count += 1
                print(f"[修复] ID {row_id}: {old_time} -> {china_time}")
                
            except Exception as e:
                print(f"[错误] ID {row_id}: 处理时间 '{old_time}' 时出错: {e}")
        
        conn.commit()
        print(f"\n[完成] 共修复 {updated_count} 条记录的时间")

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"[错误] 数据库文件不存在: {DB_PATH}")
    else:
        print(f"[开始] 修复数据库中的时间记录...")
        print(f"[数据库] {DB_PATH}")
        fix_old_times()

