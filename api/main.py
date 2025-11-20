#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# 加载 .env 文件
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

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



# 新增 /generate-image 路由，返回演示图片
from fastapi import Form
from io import BytesIO
import base64
from PIL import Image, ImageDraw

@app.post("/generate-image")
async def generate_image(dream_text: str = Form(...)):
    # 1. 优化英文 prompt
    prompt = f"Translate this dream description into a detailed English image generation prompt. Return ONLY the English prompt. Dream: {dream_text} Requirements: Cinematic, vivid, atmospheric, surreal, detailed visual descriptors."
    try:
        import dashscope
        from dashscope import Generation, ImageSynthesis
        import requests
        import os
        # 确保 API KEY 被设置
        api_key = os.environ.get("DASHSCOPE_API_KEY") or "sk-1bb88c7976254a628f3fa470a25b83c0"
        dashscope.api_key = api_key
        print(f"[DEBUG] Set dashscope.api_key to: {api_key[:20]}...")
        # 1. 用文本模型优化英文 prompt
        opt_response = Generation.call(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
            timeout=30,
        )
        optimized_prompt = None
        if opt_response and getattr(opt_response, 'status_code', None) == 200:
            output = getattr(opt_response, 'output', None)
            if output:
                choices = getattr(output, 'choices', None)
                if choices and len(choices) > 0:
                    msg_content = getattr(choices[0], 'message', None)
                    if msg_content:
                        content = getattr(msg_content, 'content', None)
                        if content:
                            optimized_prompt = content.strip()
        if not optimized_prompt:
            optimized_prompt = f"Cinematic dream scene: {dream_text}. Style: ethereal, mysterious, surreal, atmospheric. 8K quality."
        # 2. 调用 qwen-image-plus 生成图片
        img_result = ImageSynthesis.call(
            model="qwen-image-plus",
            prompt=optimized_prompt,
            size="1328*1328",
            n=1
        )
        image_url = None
        if img_result.output:
            task_status = img_result.output.get('task_status', 'UNKNOWN')
            if task_status == "SUCCEEDED":
                results = img_result.output.get("results", [])
                if results and len(results) > 0:
                    image_url = results[0].get("url")
        # 3. 下载图片并转为 data URI
        if image_url:
            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()
            img_base64 = base64.b64encode(img_response.content).decode('utf-8')
            content_type = img_response.headers.get('Content-Type', 'image/png')
            data_uri = f'data:{content_type};base64,{img_base64}'
            return JSONResponse({
                "success": True,
                "image": data_uri,
                "type": "datauri_real",
                "optimized_prompt": optimized_prompt[:200],
                "message": "图像生成成功（qwen-image-plus）"
            })
    except Exception as e:
        print(f"[ERROR] 图像生成失败: {e}")
    # 4. Fallback：生成演示 PNG
    img = Image.new('RGB', (800, 450), color=(11, 18, 32))
    draw = ImageDraw.Draw(img)
    for i in range(450):
        r = int(11 + (140 - 11) * (i / 450))
        g = int(18 + (120 - 18) * (i / 450))
        b = int(32 + (255 - 32) * (i / 450))
        draw.line([(0, i), (800, i)], fill=(r, g, b))
    text = "Dream Image (Demo)"
    draw.text((320, 210), text, fill=(209, 250, 229))
    buf = BytesIO()
    img.save(buf, format='PNG')
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    data_uri = f'data:image/png;base64,{img_base64}'
    return JSONResponse({
        "success": True,
        "image": data_uri,
        "type": "datauri_fallback",
        "message": "演示图像"
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


