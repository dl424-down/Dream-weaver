# 快速启动指南 - 视频生成功能

## 🎬 30秒快速开始

### 步骤 1: 检查配置
```bash
# 确认 .env 文件中有 API Key
cat D:\clone\Dream-weaver\.env
```

### 步骤 2: 启动后端
```bash
cd D:\clone\Dream-weaver
python api/main.py
# 预期：Uvicorn server running on http://0.0.0.0:8000
```

### 步骤 3: 启动前端
```bash
cd D:\clone\Dream-weaver\dream_weaver
npm run dev
# 预期：http://localhost:5173
```

### 步骤 4: 使用功能
1. 打开浏览器访问 `http://localhost:5173`
2. 输入梦境描述，如：`"星空下翩翩起舞"`
3. 点击蓝色的**"时光投影"**按钮
4. 等待 1-5 分钟
5. 视频自动显示！

## 🎯 核心改动

### 后端 (api/main.py)
```python
# 新增路由
@app.post("/generate-video")
async def generate_video(dream_text, duration=5, size="832*480", entry_id=None):
    result = generate_dream_video(dream_text, duration, size)
    return JSONResponse({
        "success": True,
        "video_url": result["video_url"],
        "local_path": result["local_path"],
        "message": "视频生成成功"
    })
```

### 前端 (dream_weaver/src/App.vue)
```javascript
// 新增函数
async function generateVideo() {
  const form = new FormData()
  form.append('dream_text', dreamText.value)
  form.append('duration', videoDuration.value)
  form.append('size', videoSize.value)
  
  const resp = await fetch('http://localhost:8000/generate-video', {
    method: 'POST',
    body: form
  })
  const data = await resp.json()
  
  // 设置视频 URL - 直接在前端播放！
  generatedVideo.value = {
    url: data.video_url,
    message: data.message
  }
}

// HTML 中
<video :src="generatedVideo.url" controls autoplay></video>
```

## 📱 用户界面

| 按钮位置 | 功能 | 颜色 |
|---------|------|------|
| 具象化梦境 | 生成梦境图片 | 粉红色 |
| **时光投影** | **✨ 生成视频** | **蓝色** |
| ▶ 视频参数 | 展开参数设置 | 灰色 |

## 🎥 视频播放

视频直接在前端播放，支持：
- ⏯️ 暂停/播放
- 🔊 音量控制
- ⏱️ 进度条跳转
- 🖥️ 全屏播放

## ⚙️ 参数说明

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| dream_text | - | 任意 | 梦境描述 |
| duration | 5 秒 | 1-60 | 视频时长 |
| size | 832*480 | 标准 | 视频分辨率 |

## 🔍 调试技巧

### 查看视频 URL
打开浏览器开发者工具 (F12) → Network 标签：
```
POST /generate-video
Response: {
  "video_url": "https://dashscope.aliyuncs.com/..."
}
```

### 测试 API
```bash
curl -X POST "http://localhost:8000/generate-video" \
  -F "dream_text=在梦中飞行" \
  -F "duration=5"
```

## ⚠️ 常见问题

**Q: 视频加载很慢?**  
A: 正常！视频生成需要 1-5 分钟，会显示加载动画。

**Q: 视频无法播放?**  
A: 检查浏览器是否支持 MP4 格式，或检查网络连接。

**Q: API Key 错误?**  
A: 确认 `.env` 文件中的 `DASHSCOPE_API_KEY` 正确有效。

**Q: 前端无法连接后端?**  
A: 确保后端已启动在 `http://localhost:8000`。

## 📂 文件位置

```
D:\clone\Dream-weaver\
├── api/
│   └── main.py                    # ← 后端主文件 (包含 /generate-video)
├── analyse_script/
│   └── dream_video_generator.py   # ← 视频生成模块
├── dream_weaver/
│   └── src/
│       └── App.vue                # ← 前端主文件 (包含视频 UI)
├── data/
│   └── uploads/
│       └── dream_video_*.mp4      # ← 保存的视频文件
└── .env                           # ← API Key 配置
```

## ✨ 功能演示

1. **输入梦境**: `"与彩虹一起在云间漂浮"`
2. **点击时光投影**
3. **等待... 正在编织时光...**
4. **✅ 视频已生成，现在播放！**

---

祝您使用愉快！🎬✨
