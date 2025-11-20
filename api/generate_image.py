# -*- coding: utf-8 -*-
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import dashscope, requests, os
from dashscope import ImageSynthesis

app = FastAPI()

# ✅ 设置 DashScope API Key
API_KEY = "sk-1bb88c7976254a628f3fa470a25b83c0"
dashscope.api_key = API_KEY
os.environ["DASHSCOPE_API_KEY"] = API_KEY

@app.post("/generate_image")
def generate_image(prompt: str = Form(...)):
    """
    前端发送梦境文本，后端生成梦境图像并返回图片URL。
    """
    try:
        result = ImageSynthesis.call(
            model="qwen-image-plus",
            prompt=prompt,
            size="1328*1328",
            n=1
        )

        if result.output and "results" in result.output and len(result.output["results"]) > 0:
            image_url = result.output["results"][0]["url"]
            return JSONResponse(content={
                "success": True,
                "image_url": image_url
            })
        else:
            return JSONResponse(content={
                "success": False,
                "message": "模型未返回有效结果"
            })

    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        })