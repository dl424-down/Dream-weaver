# 梦境分析系统 - 安装指南

## 依赖安装问题解决方案

### 问题：tokenizers 编译失败

如果遇到 `Failed building wheel for tokenizers` 错误，请按以下步骤解决：

#### 方案1：使用预编译版本（推荐）

```bash
# 先安装预编译的tokenizers
pip install tokenizers --only-binary=all

# 然后安装其他依赖
pip install transformers==4.21.0  # 使用更新的版本
pip install timm==0.6.12         # 使用更新的版本
pip install Pillow requests
```

#### 方案2：使用更新的版本

```bash
# 使用更新且兼容的版本
pip install transformers>=4.20.0
pip install timm>=0.6.0
pip install tokenizers>=0.12.0
pip install Pillow requests numpy
```

#### 方案3：跳过有问题的依赖（演示模式）

如果仍然无法安装，可以只安装基础依赖：

```bash
pip install Pillow requests numpy
```

系统会自动进入演示模式，不使用真实的BLIP模型。

## 完整安装步骤

### 1. 创建虚拟环境（推荐）

```bash
python -m venv dream_env
dream_env\Scripts\activate  # Windows
```

### 2. 升级pip

```bash
python -m pip install --upgrade pip
```

### 3. 安装依赖

选择以下方案之一：

**完整版本（需要PyTorch）：**
```bash
# 安装CPU版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install transformers>=4.20.0 timm>=0.6.0 Pillow requests numpy
```

**轻量版本（演示模式）：**
```bash
pip install Pillow requests numpy tkinter
```

### 4. 验证安装

```bash
python dream_analyzer.py
```

## 系统要求

- Python 3.7+
- Windows 10/11
- 至少 4GB RAM
- 2GB 可用磁盘空间

## 故障排除

### 如果仍然遇到编译错误：

1. **安装 Microsoft C++ Build Tools**
   - 下载并安装 Visual Studio Build Tools
   - 或安装 Visual Studio Community

2. **使用conda替代pip**
   ```bash
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   conda install transformers timm -c conda-forge
   ```

3. **使用预编译轮子**
   ```bash
   pip install --only-binary=all transformers timm tokenizers
   ```

### 如果只是课程演示：

可以直接运行演示模式，不需要安装PyTorch相关依赖：

```bash
python dream_ui.py
```

系统会自动检测可用的依赖并相应调整功能。