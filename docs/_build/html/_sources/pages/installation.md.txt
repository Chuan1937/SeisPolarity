# 安装与配置

## 通过 pip 安装

SeisPolarity 可以通过两种方式安装。在这两种情况下，您可能需要考虑在虚拟环境中安装 SeisPolarity，例如使用 conda。

### 标准安装

SeisPolarity 可直接通过 pip 获取。要在本地安装，请运行：

```bash
pip install seispolarity
```

### 从源码安装

如果您想从源码安装最新版本，请克隆仓库并运行：

```bash
git clone https://github.com/Chuan1937/SeisPolarity.git
cd SeisPolarity
pip install -e .
```

## 安装文档依赖

要构建文档，请安装文档依赖：

```bash
pip install seispolarity[docs]
```

## 配置

### 数据集缓存目录

SeisPolarity 会自动下载数据集和模型。默认情况下，数据集缓存在 `~/.cache/seispolarity/datasets`，模型缓存在 `~/.cache/seispolarity/models`。

要配置自定义缓存目录：

```python
from seispolarity import configure_cache

configure_cache(cache_dir="/path/to/cache")
```

或者设置 `SEISPOLARITY_CACHE_DIR` 环境变量：

```bash
export SEISPOLARITY_CACHE_DIR=/path/to/cache
```

### 远程仓库

SeisPolarity 使用远程仓库来提供数据集和模型权重。您可以配置远程仓库：

```python
import seispolarity

# 查看当前的远程根目录
print(seispolarity.remote_root)      # 数据仓库
print(seispolarity.remote_model_root)  # 模型仓库
```

## GPU 支持

SeisPolarity 基于 PyTorch 构建，PyTorch 支持 CUDA 进行 GPU 加速。

### GPU 安装

要安装带 CUDA 支持的 PyTorch，请遵循[官方 PyTorch 安装指南](https://pytorch.org/)。

### ModelScope 访问（中国用户）

对于中国用户，SeisPolarity 支持 ModelScope 以实现更快的下载：

```python
from seispolarity import get_dataset_path

# 使用 ModelScope 而不是 Hugging Face
data_path = get_dataset_path("SCSN", use_hf=False)
```
