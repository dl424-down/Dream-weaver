#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
from typing import Optional

# 确保项目根目录在导入路径中
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 复用现有的 DreamAnalyzer（位于 analyse_script 包内）
from analyse_script.dream_analyzer import DreamAnalyzer

app = FastAPI(
    title="Dream Weaver API", 
    version="1.0.0",
    description="梦境分析系统API，提供梦境文本和图像的分析功能"
)

# 允许前端本地开发访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = DreamAnalyzer()

# 简单的内存存储（生产环境请使用数据库）
analysis_history = []

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy", 
        "service": "Dream Weaver API",
        "version": "1.0.0"
    }

@app.post("/analyze/text")
async def analyze_text_only(
    dream_text: str = Form(..., description="梦境文本描述")
):
    """仅分析梦境文本，不处理图片"""
    try:
        result = analyzer.analyze_dream(dream_text, image_path=None)
        
        # 记录到历史
        analysis_history.append({
            "dream_text": dream_text,
            "analysis": result,
            "timestamp": "刚刚"
        })
        
        return {
            "success": True,
            "data": {
                "emotions": result["core_elements"]["emotions"],
                "themes": result["core_elements"]["themes"], 
                "keywords": result["core_elements"]["keywords"],
                "detailed_analysis": result["detailed_analysis"],
                "visualization_prompt": result["visualization_prompt"]
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/analyze", 
          summary="分析梦境",
          description="根据文本描述和可选图片分析梦境内容，返回情绪、主题、关键词等分析结果")
async def analyze(
    dream_text: str = Form(..., description="梦境文本描述"),
    image: Optional[UploadFile] = File(None, description="可选梦境相关图片")
):
    image_path = None
    tmp_file = None
    try:
        if image is not None:
            suffix = os.path.splitext(image.filename or "")[1] or ".jpg"
            fd, tmp_file = tempfile.mkstemp(prefix="dream_img_", suffix=suffix)
            with os.fdopen(fd, "wb") as f:
                f.write(await image.read())
            image_path = tmp_file

        result = analyzer.analyze_dream(dream_text, image_path=image_path)
        
        # 记录到历史
        analysis_history.append({
            "dream_text": dream_text,
            "analysis": result,
            "timestamp": "刚刚",
            "has_image": image is not None
        })
        
        # 返回结构化的响应
        return {
            "success": True,
            "data": {
                "emotions": result["core_elements"]["emotions"],
                "themes": result["core_elements"]["themes"],
                "keywords": result["core_elements"]["keywords"],
                "detailed_analysis": result["detailed_analysis"],
                "visualization_prompt": result["visualization_prompt"],
                "image_caption": result.get("image_caption")
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
    finally:
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass

@app.get("/analysis/history")
async def get_analysis_history(limit: int = 10):
    """获取分析历史"""
    return {
        "success": True,
        "history": analysis_history[-limit:]
    }

@app.delete("/analysis/history")
async def clear_analysis_history():
    """清空分析历史"""
    analysis_history.clear()
    return {"success": True, "message": "历史记录已清空"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


