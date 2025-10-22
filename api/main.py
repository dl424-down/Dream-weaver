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

app = FastAPI(title="Dream Weaver API", version="1.0.0")

# 允许前端本地开发访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = DreamAnalyzer()


@app.post("/analyze")
async def analyze(
    dream_text: str = Form(...),
    image: Optional[UploadFile] = File(None),
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
        return JSONResponse(result)
    finally:
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


