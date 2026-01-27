#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLite 数据库工具
负责初始化数据库并提供写入梦境记录的便捷函数
"""

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "dreams.db")

_lock = threading.Lock()


def init_db() -> None:
    """初始化数据库和表结构"""
    os.makedirs(DATA_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dream_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dream_text TEXT NOT NULL,
                text_analysis_json TEXT,
                combined_analysis TEXT,
                visualization_prompt TEXT,
                image_caption TEXT,
                image_path TEXT,
                suggestions TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()
    print(f"[数据库] 数据库初始化完成: {DB_PATH}")


def save_dream_entry(
    dream_text: str,
    text_analysis: Optional[Dict] = None,
    combined_analysis: Optional[str] = None,
    visualization_prompt: Optional[str] = None,
    image_caption: Optional[str] = None,
    image_path: Optional[str] = None,
    suggestions: Optional[str] = None,
) -> int:
    """将梦境分析记录写入数据库，返回插入的记录ID"""
    payload = json.dumps(text_analysis or {}, ensure_ascii=False)
    suggestion_text = suggestions
    if not suggestion_text and text_analysis:
        suggestion_text = text_analysis.get("analysis")

    # 使用中国时区（UTC+8）的本地时间
    from datetime import timezone, timedelta
    # 明确使用 UTC+8 时区
    china_tz = timezone(timedelta(hours=8))
    # 获取 UTC 时间并转换为中国时区
    utc_now = datetime.now(timezone.utc)
    china_now = utc_now.astimezone(china_tz)
    current_time = china_now.strftime("%Y-%m-%d %H:%M:%S")

    with _lock:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                """
                INSERT INTO dream_entries (
                    dream_text,
                    text_analysis_json,
                    combined_analysis,
                    visualization_prompt,
                    image_caption,
                    image_path,
                    suggestions,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dream_text,
                    payload,
                    combined_analysis,
                    visualization_prompt,
                    image_caption,
                    image_path,
                    suggestion_text,
                    current_time,
                ),
            )
            conn.commit()
            entry_id = cursor.lastrowid
            print(f"[数据库] 梦境记录已保存 (ID: {entry_id}, 文本长度: {len(dream_text)}字符, 时间: {current_time})")
            return entry_id


def update_image_path(entry_id: int, image_path: str) -> None:
    """更新指定梦境记录的图片路径"""
    with _lock:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                UPDATE dream_entries
                SET image_path = ?
                WHERE id = ?
                """,
                (image_path, entry_id),
            )
            conn.commit()
            print(f"[数据库] 已更新梦境记录 {entry_id} 的图片路径为: {image_path}")

def update_dream_entry(
    entry_id: int,
    dream_text: str,
    text_analysis: Optional[Dict] = None,
    combined_analysis: Optional[str] = None,
    visualization_prompt: Optional[str] = None,
    image_caption: Optional[str] = None,
) -> bool:
    """更新指定梦境记录的内容和分析结果"""
    with _lock:
        with sqlite3.connect(DB_PATH) as conn:
            # 准备更新数据
            updates = []
            params = []
            
            if dream_text:
                updates.append("dream_text = ?")
                params.append(dream_text)
            
            if text_analysis is not None:
                updates.append("text_analysis_json = ?")
                params.append(json.dumps(text_analysis, ensure_ascii=False))
            
            if combined_analysis is not None:
                updates.append("combined_analysis = ?")
                params.append(combined_analysis)
            
            if visualization_prompt is not None:
                updates.append("visualization_prompt = ?")
                params.append(visualization_prompt)
            
            if image_caption is not None:
                updates.append("image_caption = ?")
                params.append(image_caption)
            
            if not updates:
                print(f"[数据库] 警告：没有要更新的字段，entry_id={entry_id}")
                return False
            
            # 执行更新
            params.append(entry_id)
            sql = f"""
                UPDATE dream_entries
                SET {', '.join(updates)}
                WHERE id = ?
            """
            cursor = conn.execute(sql, params)
            conn.commit()
            
            if cursor.rowcount > 0:
                print(f"[数据库] 已更新梦境记录 {entry_id}")
                return True
            else:
                print(f"[数据库] 警告：未找到要更新的记录，entry_id={entry_id}")
                return False

def get_recent_entries(limit: int = 20):
    """（可选）获取最近的梦境记录，便于调试"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT id, dream_text, combined_analysis, created_at, image_path, text_analysis_json, visualization_prompt
            FROM dream_entries
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        result = []
        for row in rows:
            entry = dict(row)
            # 解析JSON字段
            if entry.get('text_analysis_json'):
                try:
                    entry['text_analysis'] = json.loads(entry['text_analysis_json'])
                except:
                    entry['text_analysis'] = {}
            result.append(entry)
        return result

def get_dream_entry_by_id(entry_id: int):
    """根据ID获取单条梦境记录的完整信息"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT * FROM dream_entries WHERE id = ?
            """,
            (entry_id,),
        )
        row = cursor.fetchone()
        if row:
            entry = dict(row)
            # 解析JSON字段
            if entry.get('text_analysis_json'):
                try:
                    entry['text_analysis'] = json.loads(entry['text_analysis_json'])
                except:
                    entry['text_analysis'] = {}
            return entry
        return None

