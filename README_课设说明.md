# Dream Weaver - 梦境分析系统

<div align="center">

![Dream Weaver](https://img.shields.io/badge/Dream-Weaver-purple?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Vue](https://img.shields.io/badge/Vue-3.5-green?style=for-the-badge&logo=vue.js)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal?style=for-the-badge&logo=fastapi)

**基于 AI 的多模态梦境分析与可视化系统**

[功能特性](#-核心功能) • [快速开始](#-快速开始) • [项目结构](#-项目结构) • [API 文档](#-api-接口文档)

</div>

---

## 📖 项目简介

Dream Weaver 是一个基于 AI 技术的梦境分析系统，结合了自然语言处理、计算机视觉和图像生成技术，为用户提供深度的梦境心理分析和可视化服务。

### ✨ 核心特性

- 🧠 **智能梦境分析** - 使用大语言模型（通义千问）进行情绪、主题、关键词识别
- 🖼️ **多模态理解** - 融合文本和图片信息，提供更全面的分析
- 🎨 **图像生成** - 基于梦境描述生成 AI 图像
- 📊 **历史记录** - 完整的梦境记录管理和查询系统
- 🔄 **智能缓存** - Redis 缓存加速，提升响应速度
- 💾 **数据持久化** - SQLite 数据库存储所有分析记录
- 📈 **综合分析** - 支持多梦境记录的批量分析和趋势分析

---

## 🎯 核心功能

### 1. 梦境文本分析

- **情绪识别**：自动识别梦境中的主要情绪（快乐、焦虑、恐惧、悲伤、愤怒、平静、困惑）
- **主题分类**：识别梦境主题（飞行、追逐、水、动物、人物、场所、考试等）
- **关键词提取**：提取梦境中的关键名词和重要概念
- **心理分析**：生成 200-400 字的详细心理分析报告

### 2. 图像理解与分析

- **图片上传**：支持上传与梦境相关的图片
- **BLIP 模型分析**：使用 BLIP 模型理解图片内容
- **深度融合**：图片信息自动融入到文本分析中，而非简单拼接
- **多模态分析**：同时考虑文本和图片，生成更准确的分析结果

### 3. 视觉化提示词生成

- **智能提示词**：生成 100-200 字的中文图像生成提示词
- **融合图片信息**：如果上传了图片，提示词会融合图片的视觉元素
- **详细描述**：包含场景、氛围、色彩、光影、构图等细节

### 4. AI 图像生成

- **梦境具象化**：根据梦境描述生成 AI 图像
- **模型支持**：使用阿里云 qwen-image-plus 模型
- **智能优化**：自动优化提示词，提升生成质量
- **缓存机制**：相同描述的图像会被缓存，避免重复生成

### 5. 历史记录管理

- **记录保存**：所有分析结果自动保存到数据库
- **历史查询**：支持查看历史梦境记录列表
- **详情查看**：点击记录可查看完整的分析详情
- **记录删除**：支持删除不需要的记录（前端删除）

### 6. 综合分析功能

- **批量分析**：选择多个梦境记录进行综合分析
- **状态评分**：提供综合状态、睡眠质量、情绪状态评分（0-100分）
- **情绪分布**：统计各种情绪的出现频率和强度
- **专业建议**：基于分析结果提供个性化的改善建议

### 7. 缓存系统

- **Redis 缓存**：使用 Redis 缓存分析结果，提升响应速度
- **智能缓存**：支持分析结果、图像生成、历史记录等多种缓存
- **自动失效**：新增记录时自动清除相关缓存
- **缓存状态**：提供缓存状态查询接口

---

## 🚀 快速开始

### 环境要求

- **Python**: 3.8+ （推荐 3.10+）
- **Node.js**: 18+ （用于前端构建）
- **Redis**: 可选，用于缓存功能（不安装也能运行，但无缓存）

### 安装步骤

#### 1. 克隆项目

```bash
git clone <repository-url>
cd 梦境
```

#### 2. 安装 Python 依赖

```bash
# 完整版（包含 BLIP 模型）
pip install -r requirements.txt

# 或最小版（不含 AI 模型，使用演示模式）
pip install -r requirements_minimal.txt
```

#### 3. 安装前端依赖

```bash
cd dream_weaver
npm install
cd ..
```

#### 4. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
# DashScope API Key（必需）
DASHSCOPE_API_KEY=your_api_key_here

# Redis 配置（可选）
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
```

#### 5. 启动 Redis（可选）

```bash
# Docker
docker run -d --name redis-dream -p 6379:6379 redis

# 或使用系统服务
# Ubuntu/Debian: sudo systemctl start redis-server
# macOS: brew services start redis
```

#### 6. 启动服务

**启动后端：**
```bash
python api/main.py
```
后端运行在 `http://localhost:8000`

**启动前端（新终端）：**
```bash
cd dream_weaver
npm run dev
```
前端运行在 `http://localhost:5173`

---

## 📁 项目结构

```
梦境/
├── api/                          # 后端 API 服务
│   └── main.py                  # FastAPI 主服务文件
│
├── analyse_script/               # 核心分析算法
│   ├── dream_analyzer.py        # 梦境分析核心类
│   └── requirements_cpu.txt     # Python 依赖（旧版）
│
├── dream_weaver/                 # 前端界面
│   ├── src/
│   │   ├── App.vue              # 主组件（核心）
│   │   ├── main.js              # Vue 入口
│   │   ├── style.css            # 全局样式
│   │   └── history.vue          # 历史记录组件
│   ├── index.html               # HTML 入口
│   ├── package.json             # 前端依赖
│   └── vite.config.js           # Vite 配置
│
├── db/                           # 数据库模块
│   ├── database.py              # SQLite 数据库操作
│   └── redis_cache.py           # Redis 缓存管理
│
├── data/                         # 数据目录
│   ├── dreams.db                # SQLite 数据库
│   └── uploads/                 # 上传的图片和生成的图像
│
├── scripts/                      # 工具脚本
│   ├── check_redis.py           # Redis 状态检查
│   ├── show_redis_storage.py    # Redis 存储查看
│   └── test_cache_with_db_data.py # 缓存测试
│
├── docs/                         # 项目文档
│   ├── api模块详细设计文档.md
│   ├── 项目架构说明.md
│   └── 系统逻辑流程说明.md
│
├── requirements.txt              # Python 完整依赖
├── package.json                  # Node.js 依赖（根目录）
└── README_课设说明.md           # 本文件
```

---

## 🔌 API 接口文档

### 基础信息

- **Base URL**: `http://localhost:8000`
- **API 文档**: `http://localhost:8000/docs` (Swagger UI)

### 主要接口

#### 1. 梦境分析

**POST** `/analyze`

分析梦境文本，可选上传图片。

**请求参数：**
- `dream_text` (string, required): 梦境文本描述
- `image` (file, optional): 相关图片文件

**响应示例：**
```json
{
  "text_analysis": {
    "emotions": ["平静", "困惑", "神秘"],
    "themes": ["天空", "光", "飞行"],
    "keywords": ["云海", "夕阳", "月亮", "光点"],
    "analysis": "详细的心理分析文本（200-400字）..."
  },
  "image_caption": "图片描述（如果有）",
  "combined_analysis": "融合了图片信息的综合分析...",
  "visualization_prompt": "详细的视觉化提示词（100-200字）...",
  "entry_id": 123
}
```

#### 2. 图像生成

**POST** `/generate-image`

根据梦境描述生成 AI 图像。

**请求参数：**
- `dream_text` (string, required): 梦境文本描述
- `entry_id` (int, optional): 关联的记录ID

**响应示例：**
```json
{
  "success": true,
  "image": "data:image/png;base64,...",
  "type": "datauri_real",
  "optimized_prompt": "优化后的英文提示词",
  "message": "图像生成成功",
  "entry_id": 123
}
```

#### 3. 历史记录列表

**GET** `/dreams/history?limit=20`

获取最近的梦境记录列表。

**查询参数：**
- `limit` (int, optional): 返回记录数量，默认 20

**响应示例：**
```json
{
  "success": true,
  "count": 5,
  "entries": [
    {
      "id": 123,
      "dream_text": "梦境描述...",
      "preview": "梦境描述前50字...",
      "created_at": "2024-12-06 12:00:00",
      "has_image": true,
      "has_analysis": true
    }
  ]
}
```

#### 4. 记录详情

**GET** `/dreams/{entry_id}`

获取单条梦境记录的完整详情。

**响应示例：**
```json
{
  "success": true,
  "entry": {
    "id": 123,
    "dream_text": "完整的梦境描述",
    "text_analysis": {
      "emotions": ["平静"],
      "themes": ["天空"],
      "keywords": ["云", "光"]
    },
    "combined_analysis": "详细分析...",
    "visualization_prompt": "视觉化提示词...",
    "image_url": "data:image/jpeg;base64,...",
    "created_at": "2024-12-06 12:00:00"
  }
}
```

#### 5. 综合分析

**POST** `/dreams/comprehensive-analysis`

对多个梦境记录进行综合分析。

**请求体：**
```json
{
  "entry_ids": [1, 2, 3, 4, 5]
}
```

**响应示例：**
```json
{
  "success": true,
  "analysis": {
    "overall_score": 75,
    "sleep_quality": 80,
    "emotion_score": 70,
    "summary": "综合分析总结（200字左右）",
    "emotion_breakdown": {
      "焦虑": 60,
      "平静": 30,
      "快乐": 10
    },
    "sleep_analysis": "睡眠质量详细分析（150字左右）",
    "suggestions": ["建议1", "建议2", "建议3"]
  }
}
```

#### 6. 缓存状态

**GET** `/cache/status`

获取 Redis 缓存状态信息。

**响应示例：**
```json
{
  "success": true,
  "cache": {
    "redis_library_installed": true,
    "cache_enabled": true,
    "redis_available": true,
    "connection_info": {
      "host": "localhost",
      "port": 6379,
      "db": 0
    },
    "redis_info": {
      "version": "7.0.0",
      "used_memory_human": "2.5M",
      "connected_clients": 1
    }
  }
}
```

---

## 🛠️ 技术栈

### 后端

- **FastAPI** - 现代、快速的 Web 框架
- **Uvicorn** - ASGI 服务器
- **DashScope API** - 阿里云通义千问（LLM 分析）
- **qwen-image-plus** - 阿里云图像生成模型
- **BLIP** - HuggingFace 图像理解模型
- **PyTorch** - 深度学习框架（用于 BLIP）
- **SQLite** - 轻量级数据库
- **Redis** - 内存缓存数据库

### 前端

- **Vue 3** - 渐进式 JavaScript 框架
- **Three.js** - 3D 图形库（背景效果）
- **Vite** - 下一代前端构建工具

### 开发工具

- **Python 3.8+** - 后端开发语言
- **Node.js 18+** - 前端开发环境
- **npm** - 包管理器

---

## 🎨 功能亮点

### 1. 多模态融合分析

- 文本和图片信息深度融合，而非简单拼接
- 图片描述自动翻译成中文
- LLM 同时考虑文本和图片，生成更准确的分析

### 2. 智能缓存系统

- Redis 缓存分析结果，大幅提升响应速度
- 支持多种缓存类型（分析结果、图像生成、历史记录等）
- 自动缓存失效机制，保证数据一致性

### 3. 3D 视觉效果

- Three.js 实现的动态 3D 星空背景
- 响应式设计，适配不同屏幕尺寸
- 现代化的 UI 设计

### 4. 完整的记录管理

- 自动保存所有分析记录
- 支持历史记录查询和详情查看
- 支持批量分析和趋势分析

---

## 📝 使用示例

### 1. 基础文本分析

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "dream_text=我梦见自己在天空中飞翔，周围是金色的云彩"
```

### 2. 带图片的分析

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "dream_text=我梦见自己在天空中飞翔" \
  -F "image=@/path/to/image.jpg"
```

### 3. 生成图像

```bash
curl -X POST "http://localhost:8000/generate-image" \
  -F "dream_text=我梦见自己在天空中飞翔，周围是金色的云彩"
```

### 4. 查询历史记录

```bash
curl "http://localhost:8000/dreams/history?limit=10"
```

---

## 🔧 配置说明

### 环境变量

在项目根目录创建 `.env` 文件：

```env
# DashScope API Key（必需）
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxx

# Redis 配置（可选）
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# 后端服务配置（可选）
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
```

### 获取 API Key

1. 访问 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/)
2. 注册/登录账号
3. 创建 API Key
4. 将 API Key 填入 `.env` 文件

---

## 🐛 常见问题

### 1. BLIP 模型加载失败

**问题**：首次运行需要下载模型，可能因为网络问题失败

**解决**：
- 确保网络连接稳定
- 检查磁盘空间（模型约 1GB）
- 如果不需要 BLIP，使用 `requirements_minimal.txt`

### 2. Redis 连接失败

**问题**：Redis 未运行或配置错误

**解决**：
- 检查 Redis 是否运行：`redis-cli ping`
- 检查 `.env` 文件中的 Redis 配置
- 如果不需要缓存，可以不安装 Redis（功能仍可用）

### 3. API 调用失败

**问题**：DashScope API 调用失败

**解决**：
- 检查 `.env` 文件中的 `DASHSCOPE_API_KEY` 是否正确
- 检查 API Key 是否有效
- 查看后端日志获取详细错误信息

### 4. 前端构建失败

**问题**：npm 安装或构建失败

**解决**：
- 确保 Node.js 版本 >= 18
- 删除 `node_modules` 和 `package-lock.json`，重新安装
- 检查网络连接（npm 需要下载包）

---

## 📚 相关文档

- [API 详细设计文档](docs/api模块详细设计文档.md)
- [项目架构说明](docs/项目架构说明.md)
- [系统逻辑流程说明](docs/系统逻辑流程说明.md)
- [安装指南](docs/install_guide.md)
- [Redis 检查指南](docs/redis_check_guide.md)

---

## 🚧 开发计划

- [ ] 支持更多情绪类型和主题
- [ ] 改进心理学解释的准确性
- [ ] 添加用户认证系统
- [ ] 支持梦境记录的导出和导入
- [ ] 添加数据可视化图表
- [ ] 支持多语言界面

---

## 📄 许可证

本项目为课程设计项目，仅供学习和研究使用。

---

## 🙏 致谢

本项目基于以下开源项目和技术：

- [BLIP](https://github.com/salesforce/BLIP) - Salesforce 的视觉-语言预训练模型
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Web 框架
- [Vue.js](https://vuejs.org/) - 渐进式 JavaScript 框架
- [Three.js](https://threejs.org/) - 3D 图形库
- [DashScope](https://dashscope.aliyun.com/) - 阿里云 AI 服务
- [PyTorch](https://pytorch.org/) - 深度学习框架

---

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件

---

<div align="center">

**Made with ❤️ for Dream Analysis**

⭐ 如果这个项目对你有帮助，请给个 Star！

</div>
