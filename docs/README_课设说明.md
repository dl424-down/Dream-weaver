# 梦境分析系统 - 课程设计项目

## 项目简介

本项目是基于BLIP（Bootstrapping Language-Image Pre-training）模型开发的梦境分析系统，专门为课程设计需求优化，支持CPU运行。

### 主要功能

1. **梦境文本分析** - 分析用户输入的梦境描述文本
2. **情绪识别** - 识别梦境中的主要情绪（快乐、焦虑、恐惧等）
3. **主题分类** - 识别梦境的主要主题（飞行、追逐、水等）
4. **图像理解** - 分析用户上传的相关图片
5. **心理解释** - 提供简单的心理学解释
6. **视觉化建议** - 生成用于AI绘画的提示词

## 系统架构

```
梦境分析系统/
├── dream_analyzer.py    # 核心分析模块
├── dream_ui.py         # 图形用户界面
├── requirements_cpu.txt # CPU版本依赖
├── README_课设说明.md   # 项目说明
└── BLIP/              # BLIP模型代码
    ├── models/        # 模型定义
    ├── configs/       # 配置文件
    └── ...
```

## 安装和运行

### 1. 环境准备

确保您的系统已安装Python 3.7+

### 2. 安装依赖

```bash
# 安装CPU版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install -r requirements_cpu.txt

# 安装BLIP依赖
cd BLIP
pip install -r requirements.txt
cd ..
```

### 3. 运行系统

```bash
# 启动图形界面
python dream_ui.py

# 或者直接测试核心功能
python dream_analyzer.py
```

## 使用说明

### 图形界面使用

1. **输入梦境描述** - 在左侧文本框中详细描述您的梦境
2. **上传相关图片**（可选）- 点击"选择图片"上传手绘图或相关照片
3. **开始分析** - 点击"🔍 开始分析"按钮
4. **查看结果** - 在右侧标签页中查看分析结果：
   - 情绪分析：主要情绪、主题、关键词
   - 梦境解析：心理学解释和综合分析
   - 视觉化建议：用于AI绘画的提示词
5. **保存结果** - 点击"💾 保存结果"将分析结果保存为文本文件

### 命令行使用

```python
from dream_analyzer import DreamAnalyzer

analyzer = DreamAnalyzer()
result = analyzer.analyze_dream("我梦见自己在飞翔...")
print(result)
```

## 技术特点

### CPU优化
- 使用较小的图像尺寸（224x224）减少计算量
- 优化模型加载和推理过程
- 支持纯CPU环境运行

### 模块化设计
- **dream_analyzer.py** - 核心分析逻辑
- **dream_ui.py** - 用户界面
- 清晰的模块分离，便于维护和扩展

### 用户友好
- 直观的图形界面
- 实时进度显示
- 结果保存功能
- 错误处理和用户提示

## 课设展示建议

### 1. 功能演示
- 准备几个典型的梦境描述案例
- 展示不同情绪类型的分析结果
- 演示图像上传和分析功能

### 2. 技术亮点
- BLIP模型的应用
- 多模态（文本+图像）分析
- CPU优化的实现
- 心理学知识的融入

### 3. 扩展可能
- 添加更多情绪类型
- 改进心理学解释的准确性
- 集成真正的图像生成功能
- 添加历史记录功能

## 注意事项

### 首次运行
- 首次运行时会自动下载BLIP预训练模型（约500MB）
- 请确保网络连接稳定
- 模型下载完成后会缓存到本地

### 性能说明
- CPU运行速度较慢，单次分析可能需要10-30秒
- 建议使用较小的图片（<2MB）
- 文本分析速度较快，图像分析相对较慢

### 系统要求
- Python 3.7+
- 至少4GB RAM
- 2GB可用磁盘空间（用于模型缓存）

## 故障排除

### 常见问题

1. **模型下载失败**
   - 检查网络连接
   - 尝试手动下载模型文件

2. **内存不足**
   - 关闭其他程序释放内存
   - 考虑使用更小的模型

3. **图片加载失败**
   - 检查图片格式（支持jpg, png, bmp等）
   - 确保图片文件未损坏

## 项目结构说明

```
梦境分析系统/
├── dream_analyzer.py      # 核心分析引擎
│   ├── DreamAnalyzer类    # 主要分析类
│   ├── 情绪关键词字典      # 情绪识别规则
│   ├── 梦境主题分类       # 主题识别规则
│   └── BLIP模型集成      # 图像理解功能
│
├── dream_ui.py           # 图形用户界面
│   ├── DreamAnalyzerGUI类 # 主界面类
│   ├── 输入区域设计       # 文本和图片输入
│   ├── 结果显示区域       # 多标签页结果展示
│   └── 文件操作功能       # 保存和清空功能
│
└── 配置和文档
    ├── requirements_cpu.txt  # 依赖包列表
    └── README_课设说明.md    # 项目文档
```

## 致谢

本项目基于以下开源项目：
- [BLIP](https://github.com/salesforce/BLIP) - Salesforce的视觉-语言预训练模型
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Tkinter](https://docs.python.org/3/library/tkinter.html) - Python GUI库