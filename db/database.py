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
                    suggestions
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dream_text,
                    payload,
                    combined_analysis,
                    visualization_prompt,
                    image_caption,
                    image_path,
                    suggestion_text,
                ),
            )
            conn.commit()
            entry_id = cursor.lastrowid
            print(f"[数据库] 梦境记录已保存 (ID: {entry_id}, 文本长度: {len(dream_text)}字符)")
            return entry_id


def get_recent_entries(limit: int = 20):
    """（可选）获取最近的梦境记录，便于调试"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT id, dream_text, combined_analysis, created_at
            FROM dream_entries
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

