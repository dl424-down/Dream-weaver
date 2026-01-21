#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
from datetime import datetime

# 加载 .env 文件
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import tempfile
from typing import Optional, List

class ComprehensiveAnalysisRequest(BaseModel):
    entry_ids: List[int]

# 确保项目根目录在导入路径中
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 复用现有的 DreamAnalyzer（位于 analyse_script 包内）
from analyse_script.dream_analyzer import DreamAnalyzer
from analyse_script.dream_video_generator import generate_dream_video
from db.database import (
    init_db,
    save_dream_entry,
    get_recent_entries,
    get_dream_entry_by_id,
    update_image_path,
)
from db.redis_cache import get_cache

init_db()

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
cache = get_cache()  # 初始化 Redis 缓存


@app.post("/analyze")
async def analyze(
    dream_text: str = Form(...),
    image: Optional[UploadFile] = File(None),
):
    image_path = None
    tmp_file = None
    saved_image_path = None
    has_image = image is not None
    
    try:
        # 处理图片上传
        if image is not None:
            suffix = os.path.splitext(image.filename or "")[1] or ".jpg"
            fd, tmp_file = tempfile.mkstemp(prefix="dream_img_", suffix=suffix)
            with os.fdopen(fd, "wb") as f:
                f.write(await image.read())
            image_path = tmp_file
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            saved_name = f"{timestamp}{suffix}"
            saved_image_path = os.path.join(UPLOAD_DIR, saved_name)
            shutil.copy(tmp_file, saved_image_path)

        # 先尝试从缓存获取分析结果
        cached_result = cache.get_analysis_cache(dream_text, has_image=has_image)
        
        if cached_result:
            # 缓存命中，使用缓存的结果
            result = cached_result.copy()  # 复制一份，避免修改原缓存
            print("[缓存] 使用缓存的分析结果，跳过 AI 模型调用")
        else:
            # 缓存未命中，执行实际分析
            print("[缓存] 缓存未命中，执行 AI 分析")
            result = analyzer.analyze_dream(dream_text, image_path=image_path)
            
            # 将分析结果写入缓存（不包含 entry_id，因为这是动态生成的）
            cache_result = {
                "text_analysis": result.get("text_analysis"),
                "combined_analysis": result.get("combined_analysis"),
                "visualization_prompt": result.get("visualization_prompt"),
                "image_caption": result.get("image_caption"),
            }
            cache.set_analysis_cache(dream_text, cache_result, has_image=has_image)

        # 无论是否使用缓存，都要保存到数据库（用于历史记录）
        try:
            entry_id = save_dream_entry(
                dream_text=dream_text,
                text_analysis=result.get("text_analysis"),
                combined_analysis=result.get("combined_analysis"),
                visualization_prompt=result.get("visualization_prompt"),
                image_caption=result.get("image_caption"),
                image_path=saved_image_path,
            )
            result['entry_id'] = entry_id  # 在响应中包含记录ID
            
            # 清除历史记录缓存，因为新增了记录
            cache.invalidate_history_cache()
        except Exception as db_error:
            print(f"[WARN] 保存梦境记录失败: {db_error}")
        
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
async def generate_image(
    dream_text: str = Form(...),
    entry_id: Optional[int] = Form(None),
):
    # 先尝试从缓存获取图像生成结果
    cached_result = cache.get_image_generation_cache(dream_text)
    
    if cached_result:
        # 缓存命中，使用缓存的结果
        print("[缓存] 使用缓存的图像生成结果，跳过 API 调用")
        # 如果提供了 entry_id，更新数据库
        if entry_id is not None:
            try:
                # 从缓存结果中获取图片路径（如果有）
                if 'saved_image_path' in cached_result and cached_result['saved_image_path']:
                    update_image_path(entry_id, cached_result['saved_image_path'])
            except Exception as db_err:
                print(f"[WARN] 更新图片路径失败: {db_err}")
        
        return JSONResponse(cached_result)
    
    # 缓存未命中，执行图像生成
    print("[缓存] 缓存未命中，执行图像生成")
    # 1. 优化英文 prompt
    prompt = f"Translate this dream description into a detailed English image generation prompt. Return ONLY the English prompt. Dream: {dream_text} Requirements: Cinematic, vivid, atmospheric, surreal, detailed visual descriptors."
    saved_image_path = None

    try:
        import dashscope
        from dashscope import Generation, ImageSynthesis
        import requests
        import os
        # 确保 API KEY 被设置（只从环境变量读取，不再使用硬编码密钥）
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY 未配置，请在项目根目录 .env 中设置该环境变量"
            )
        dashscope.api_key = api_key
        print("[DEBUG] DashScope API Key 已从环境变量加载")
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
        # 3. 下载图片并转为 data URI，同时保存到本地文件
        if image_url:
            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()
            img_bytes = img_response.content
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            content_type = img_response.headers.get('Content-Type', 'image/png')
            data_uri = f'data:{content_type};base64,{img_base64}'

            # 将生成的图片保存到 uploads 目录
            ext = ".png"
            if content_type == "image/jpeg":
                ext = ".jpg"
            elif content_type == "image/webp":
                ext = ".webp"

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"gen_{timestamp}{ext}"
            saved_image_path = os.path.join(UPLOAD_DIR, filename)
            with open(saved_image_path, "wb") as f:
                f.write(img_bytes)

            # 将图片路径写入数据库（如果有对应的 entry_id）
            try:
                if entry_id is not None:
                    update_image_path(entry_id, saved_image_path)
                    # 清除该记录的详情缓存
                    cache.invalidate_detail_cache(entry_id)
                else:
                    entry_id = save_dream_entry(
                        dream_text=dream_text,
                        image_path=saved_image_path,
                    )
                    # 清除历史记录缓存
                    cache.invalidate_history_cache()
            except Exception as db_err:
                print(f"[WARN] 保存生成图片路径到数据库失败: {db_err}")

            result = {
                "success": True,
                "image": data_uri,
                "type": "datauri_real",
                "optimized_prompt": optimized_prompt[:200],
                "message": "图像生成成功（qwen-image-plus）",
                "entry_id": entry_id,
                "saved_image_path": saved_image_path,  # 保存路径用于缓存
            }
            
            # 将结果写入缓存（不包含 entry_id，因为这是动态的）
            cache_result = {
                "success": True,
                "image": data_uri,
                "type": "datauri_real",
                "optimized_prompt": optimized_prompt[:200],
                "message": "图像生成成功（qwen-image-plus）",
                "saved_image_path": saved_image_path,
            }
            cache.set_image_generation_cache(dream_text, cache_result)
            
            return JSONResponse(result)
    except Exception as e:
        print(f"[ERROR] 图像生成失败: {e}")
    # 4. Fallback：生成演示 PNG，并保存到本地
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
    img_bytes = buf.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    data_uri = f'data:image/png;base64,{img_base64}'

    # 保存到 uploads 目录
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    fallback_filename = f"gen_fallback_{timestamp}.png"
    saved_image_path = os.path.join(UPLOAD_DIR, fallback_filename)
    with open(saved_image_path, "wb") as f:
        f.write(img_bytes)

    # 同样更新或创建数据库记录
    try:
        if entry_id is not None:
            update_image_path(entry_id, saved_image_path)
            # 清除该记录的详情缓存
            cache.invalidate_detail_cache(entry_id)
        else:
            entry_id = save_dream_entry(
                dream_text=dream_text,
                image_path=saved_image_path,
            )
            # 清除历史记录缓存
            cache.invalidate_history_cache()
    except Exception as db_err:
        print(f"[WARN] 保存演示图片路径到数据库失败: {db_err}")
    
    result = {
        "success": True,
        "image": data_uri,
        "type": "datauri_fallback",
        "message": "演示图像",
        "entry_id": entry_id,
        "saved_image_path": saved_image_path,
    }
    
    # 演示图像也缓存（虽然不常用，但保持一致性）
    cache_result = {
        "success": True,
        "image": data_uri,
        "type": "datauri_fallback",
        "message": "演示图像",
        "saved_image_path": saved_image_path,
    }
    cache.set_image_generation_cache(dream_text, cache_result)
    
    return JSONResponse(result)

@app.post("/generate-video")
async def generate_video(
    dream_text: str = Form(...),
    duration: int = Form(5),
    size: str = Form("832*480"),
    entry_id: Optional[int] = Form(None),
):
    """生成梦境视频"""
    try:
        print(f"[视频生成] 开始生成视频，梦境：{dream_text[:50]}...")
        
        # 调用视频生成函数
        result = generate_dream_video(
            dream_text=dream_text,
            duration=duration,
            size=size,
            max_poll=30,
            poll_interval=10
        )
        
        if not result.get("success"):
            return JSONResponse({
                "success": False,
                "error": result.get("error", "视频生成失败"),
                "video_url": None,
                "local_path": None
            }, status_code=500)
        
        video_url = result.get("video_url")
        local_path = result.get("local_path")
        
        # 如果提供了 entry_id，更新数据库（记录视频路径）
        if entry_id is not None and local_path:
            try:
                # 这里可以选择保存视频路径到数据库
                # 如果数据库有视频路径字段，可以添加 update_video_path 函数
                print(f"[视频生成] 视频已保存到：{local_path}")
            except Exception as db_err:
                print(f"[WARN] 更新视频路径失败: {db_err}")
        
        return JSONResponse({
            "success": True,
            "video_url": video_url,
            "local_path": local_path,
            "message": "视频生成成功",
            "entry_id": entry_id
        })
    
    except Exception as e:
        print(f"[ERROR] 视频生成失败: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "video_url": None,
            "local_path": None
        }, status_code=500)

@app.get("/dreams/history")
async def get_dream_history(limit: int = 20):
    """获取最近的梦境记录"""
    try:
        # 先尝试从缓存获取
        cached_result = cache.get_history_cache(limit=limit)
        
        if cached_result:
            print(f"[缓存] 使用缓存的历史记录列表: limit={limit}")
            return JSONResponse(cached_result)
        
        # 缓存未命中，从数据库查询
        print(f"[缓存] 缓存未命中，从数据库查询历史记录: limit={limit}")
        entries = get_recent_entries(limit=limit)
        # 为前端准备简略信息
        simplified_entries = []
        for entry in entries:
            simplified = {
                "id": entry.get("id"),
                "dream_text": entry.get("dream_text", ""),
                "preview": entry.get("dream_text", "")[:50] + ("..." if len(entry.get("dream_text", "")) > 50 else ""),
                "created_at": entry.get("created_at"),
                "has_image": bool(entry.get("image_path")),
                "has_analysis": bool(entry.get("combined_analysis"))
            }
            simplified_entries.append(simplified)
        
        result = {
            "success": True,
            "count": len(simplified_entries),
            "entries": simplified_entries
        }
        
        # 写入缓存
        cache.set_history_cache(limit, result)
        
        return JSONResponse(result)
    except Exception as e:
        print(f"[ERROR] 查询梦境记录失败: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/dreams/comprehensive-analysis")
async def comprehensive_analysis(request: ComprehensiveAnalysisRequest):
    """综合分析多个梦境记录"""
    try:
        entry_ids = request.entry_ids
        if not entry_ids:
            return JSONResponse({
                "success": False,
                "error": "请至少选择一个梦境记录"
            }, status_code=400)
        
        # 先尝试从缓存获取综合分析结果
        cached_result = cache.get_comprehensive_analysis_cache(entry_ids)
        
        if cached_result:
            print(f"[缓存] 使用缓存的综合分析结果: {len(entry_ids)} 条记录")
            return JSONResponse(cached_result)
        
        # 缓存未命中，执行综合分析
        print(f"[缓存] 缓存未命中，执行综合分析: {len(entry_ids)} 条记录")
        
        # 获取所有选中的梦境记录
        from db.database import get_dream_entry_by_id
        entries = []
        for entry_id in entry_ids:
            entry = get_dream_entry_by_id(entry_id)
            if entry:
                entries.append(entry)
        
        if not entries:
            return JSONResponse({
                "success": False,
                "error": "未找到有效的梦境记录"
            }, status_code=404)
        
        # 综合分析
        analysis_result = analyzer.analyze_comprehensive(entries)
        
        result = {
            "success": True,
            "analysis": analysis_result
        }
        
        # 写入缓存
        cache.set_comprehensive_analysis_cache(entry_ids, result)
        
        return JSONResponse(result)
    except Exception as e:
        print(f"[ERROR] 综合分析失败: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/dreams/{entry_id}")
async def get_dream_detail(entry_id: int):
    """获取单条梦境记录的完整详情"""
    try:
        # 先尝试从缓存获取
        cached_result = cache.get_detail_cache(entry_id)
        
        if cached_result:
            print(f"[缓存] 使用缓存的梦境详情: entry_id={entry_id}")
            return JSONResponse(cached_result)
        
        # 缓存未命中，从数据库查询
        print(f"[缓存] 缓存未命中，从数据库查询梦境详情: entry_id={entry_id}")
        entry = get_dream_entry_by_id(entry_id)
        if not entry:
            return JSONResponse({
                "success": False,
                "error": "记录不存在"
            }, status_code=404)
        
        # 处理图片路径，如果是本地文件，需要转换为可访问的URL
        image_path = entry.get("image_path")
        image_url = None
        if image_path and os.path.exists(image_path):
            # 读取图片并转换为base64
            try:
                import base64
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    # 根据文件扩展名确定MIME类型
                    ext = os.path.splitext(image_path)[1].lower()
                    mime_type = {
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.gif': 'image/gif',
                        '.webp': 'image/webp'
                    }.get(ext, 'image/jpeg')
                    image_url = f'data:{mime_type};base64,{img_base64}'
            except Exception as img_error:
                print(f"[WARN] 读取图片失败: {img_error}")
        
        result = {
            "success": True,
            "entry": {
                "id": entry.get("id"),
                "dream_text": entry.get("dream_text"),
                "text_analysis": entry.get("text_analysis", {}),
                "combined_analysis": entry.get("combined_analysis"),
                "visualization_prompt": entry.get("visualization_prompt"),
                "image_caption": entry.get("image_caption"),
                "image_url": image_url,
                "created_at": entry.get("created_at")
            }
        }
        
        # 写入缓存
        cache.set_detail_cache(entry_id, result)
        
        return JSONResponse(result)
    except Exception as e:
        print(f"[ERROR] 获取梦境详情失败: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/cache/status")
async def get_cache_status():
    """获取 Redis 缓存状态"""
    status = cache.get_status()
    return JSONResponse({
        "success": True,
        "cache": status
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


