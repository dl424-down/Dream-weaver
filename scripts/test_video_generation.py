import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY", "你的API_KEY")
MODEL_NAME = "wan2.5-t2v-preview"
BASE_CREATE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "X-DashScope-Async": "enable",
    "Content-Type": "application/json"
}

def generate_video(prompt, duration=5, size="832*480", max_poll=30, poll_interval=10):
    payload = {
        "model": MODEL_NAME,
        "input": {"prompt": prompt},
        "parameters": {"size": size, "duration": duration}
    }

    try:
        create_resp = requests.post(BASE_CREATE_URL, headers=HEADERS, json=payload).json()
    except Exception as e:
        print("请求失败：", e)
        return None

    task_id = create_resp.get("output", {}).get("task_id")
    if not task_id:
        print("任务创建失败，返回数据：", create_resp)
        return None

    print("任务创建成功，task_id:", task_id)

    status_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    video_url = None

    for i in range(max_poll):
        try:
            status_resp = requests.get(status_url, headers={"Authorization": f"Bearer {API_KEY}"}).json()
        except Exception as e:
            print(f"轮询请求失败，第{i+1}次：", e)
            time.sleep(poll_interval)
            continue

        task_status = status_resp.get("output", {}).get("task_status", "UNKNOWN")
        print(f"轮询第{i+1}次，状态：{task_status}")
        print("返回内容：", status_resp)

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
            break

        time.sleep(poll_interval)

    if not video_url:
        print("任务未完成或视频 URL 获取失败")
    return video_url

if __name__ == "__main__":
    prompt_text = "梦境场景：夜晚的森林中漂浮着发光的水晶球"
    video_link = generate_video(prompt_text, duration=5, size="832*480")
    print("最终生成视频链接：", video_link)
