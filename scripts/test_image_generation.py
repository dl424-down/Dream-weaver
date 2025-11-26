# -*- coding: utf-8 -*-
import os
import requests
import dashscope
from dashscope import ImageSynthesis

# ✅ 设置 DashScope API Key（两种方式都写上最保险）
API_KEY = "sk-1bb88c7976254a628f3fa470a25b83c0"
os.environ["DASHSCOPE_API_KEY"] = API_KEY
dashscope.api_key = API_KEY   # ← 必须加这一行！！！

def generate_dream_image():
    prompt = (
        "我梦见自己在一片无边无际的森林里迷路了，天色越来越暗，我很害怕。"
        "要求：电影级质感，暗色调，树木高大密集，地面有落叶，远处有微弱光点，"
        "中央有模糊人影，氛围压抑恐惧。"
    )

    print("[信息] 正在生成梦境图片...")
    print(f"[提示词] {prompt[:50]}...")

    try:
        # ✅ 调用 Qwen Image Plus
        result = ImageSynthesis.call(
            model="qwen-image-plus",
            prompt=prompt,
            size="1328*1328",
            n=1
        )

        # 调试信息
        print("[调试] 模型返回：", result)

        # ✅ 提取返回图片 URL
        if result.output and "results" in result.output and len(result.output["results"]) > 0:
            image_url = result.output["results"][0]["url"]
            print("[成功] 图片生成成功，URL：", image_url)

            # 自动下载并保存
            response = requests.get(image_url)
            image_path = os.path.abspath("dream_qwen_image.png")
            with open(image_path, "wb") as f:
                f.write(response.content)
            print("[成功] 图片已保存到：", image_path)
        else:
            print("[错误] 模型未返回有效结果。")

    except Exception as e:
        print("[错误] 出错：", e)

if __name__ == "__main__":
    generate_dream_image()