import os
import requests
import time
from datetime import datetime
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "uploads"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MODEL_NAME = "wan2.5-t2v-preview"
BASE_CREATE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"


def generate_dream_video(
    dream_text: str,
    duration: int = 5,
    size: str = "832*480",
    max_poll: int = 30,
    poll_interval: int = 10
) -> dict:
    """
    根据梦境文本生成视频，返回包含视频URL和本地路径的字典
    """
    if not API_KEY:
        raise RuntimeError(
            "DASHSCOPE_API_KEY 未配置，请在项目根目录 .env 中设置该环境变量"
        )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-DashScope-Async": "enable",
        "Content-Type": "application/json"
    }

    # 构建提示词
    prompt = f"梦境场景：{dream_text}"

    payload = {
        "model": MODEL_NAME,
        "input": {"prompt": prompt},
        "parameters": {"size": size, "duration": duration}
    }

    try:
        # 创建视频生成任务
        create_resp = requests.post(BASE_CREATE_URL, headers=headers, json=payload, timeout=30).json()
    except Exception as e:
        print("请求失败：", e)
        return {
            "success": False,
            "error": str(e),
            "video_url": None,
            "local_path": None
        }

    task_id = create_resp.get("output", {}).get("task_id")
    if not task_id:
        print("任务创建失败，返回数据：", create_resp)
        return {
            "success": False,
            "error": "任务创建失败",
            "video_url": None,
            "local_path": None
        }

    print("任务创建成功，task_id:", task_id)

    # 轮询检查任务状态
    status_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    video_url = None

    for i in range(max_poll):
        try:
            status_resp = requests.get(status_url, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=30).json()
        except Exception as e:
            print(f"轮询请求失败，第{i+1}次：", e)
            time.sleep(poll_interval)
            continue

        task_status = status_resp.get("output", {}).get("task_status", "UNKNOWN")
        print(f"轮询第{i+1}次，状态：{task_status}")

        if task_status == "SUCCEEDED":
            output = status_resp.get("output", {})
            # 新版 API
            video_url = output.get("video_url")
            # 兼容旧版 API
            if not video_url:
                result = output.get("result", {})
                videos = result.get("videos", [])
                if videos:
                    video_url = videos[0].get("url")

            if video_url:
                print("视频生成完成，下载链接：", video_url)
            else:
                print("任务完成，但未返回视频 URL")
            break
        elif task_status == "FAILED":
            print("视频生成失败：", status_resp)
            return {
                "success": False,
                "error": "视频生成失败",
                "video_url": None,
                "local_path": None
            }

        time.sleep(poll_interval)

    if not video_url:
        print("任务未完成或视频 URL 获取失败")
        return {
            "success": False,
            "error": "任务超时或未完成",
            "video_url": None,
            "local_path": None
        }

    # 下载视频文件并保存到本地
    local_path = None
    try:
        video_response = requests.get(video_url, timeout=60)
        video_response.raise_for_status()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        local_filename = f"dream_video_{timestamp}.mp4"
        local_path = os.path.join(OUTPUT_DIR, local_filename)
        
        with open(local_path, 'wb') as f:
            f.write(video_response.content)
        
        print(f"视频已保存到本地：{local_path}")
    except Exception as e:
        print(f"下载视频失败：{e}")
        local_path = None

    return {
        "success": True,
        "error": None,
        "video_url": video_url,
        "local_path": local_path
    }
