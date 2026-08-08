# 安装与配置

## 通过 pip 安装

SeisPolarity 可以通过两种方式安装。无论采用哪种方式，都建议在虚拟环境（例如 conda）中安装 SeisPolarity。

### 标准安装

SeisPolarity 可以通过 pip 直接安装。本地安装请运行：

```bash
pip install seispolarity
```

### 从源码安装

如需从源码安装最新版本，请克隆仓库并运行：

```bash
git clone https://github.com/Chuan1937/SeisPolarity.git
cd SeisPolarity
pip install -e .
```

## 文档依赖

如需构建文档，请安装文档相关依赖：

```bash
pip install seispolarity[docs]
```

## 配置

### 缓存目录配置

SeisPolarity 会自动下载数据集和模型。默认情况下，所有缓存内容存储在 `~/.seispolarity/` 目录中：

- 数据集：`~/.seispolarity/datasets`
- 模型：`~/.seispolarity/models`
- 波形数据：`~/.seispolarity/waveforms`

如需配置自定义缓存目录：

```python
from seispolarity import configure_cache

configure_cache(cache_root="/path/to/cache")
```

或设置 `SEISPOLARITY_CACHE_ROOT` 环境变量：

```bash
export SEISPOLARITY_CACHE_ROOT=/path/to/cache
```

### 远程仓库

SeisPolarity 使用远程仓库提供数据集和模型权重。您可以配置远程仓库：

```python
import seispolarity

# 查看当前的远程根目录
print(seispolarity.remote_root)      # 数据仓库
print(seispolarity.remote_model_root)  # 模型仓库
```

## GPU 支持

SeisPolarity 基于 PyTorch 构建，支持使用 CUDA 进行 GPU 加速。

### GPU 安装

如需安装带 CUDA 支持的 PyTorch，请参考 [PyTorch 官方安装指南](https://pytorch.org/)。

### ModelScope 访问（中国用户）

对于中国用户，SeisPolarity 支持使用 ModelScope 以获得更快的下载速度：

```python
from seispolarity import get_dataset_path

# 使用 ModelScope 而不是 Hugging Face
data_path = get_dataset_path("SCSN", use_hf=False)
```
