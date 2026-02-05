# 欢迎使用 SeisPolarity

:::{figure} _static/seispolarity_logo.svg
:align: center
:width: 200px
:::

**SeisPolarity** 是一个全面的地震初至极性拾取框架。

SeisPolarity 提供统一的 API，用于将深度学习模型应用于地震波形进行极性分类，支持多个数据集和模型。

## 功能特性

- **统一数据接口**：支持 SCSN、Txed、DiTing、Instance、PNW 数据集
- **多种模型**：Ross、Eqpolarity、APP、DiTingMotion、CFM、RPNet、PolarCAP
- **灵活数据加载**：RAM/磁盘流式传输，支持任意规模数据集
- **高级推理**：自动从 Hugging Face 下载预训练模型
- **数据增强**：支持多种增强技术的平衡采样
- **统一训练**：支持检查点保存和早停机制

## 快速开始

### 安装

```bash
git clone https://github.com/Chuan1937/SeisPolarity.git
cd SeisPolarity
pip install -e .
```

**系统要求**：Python ≥ 3.10、PyTorch ≥ 2.1、NumPy、h5py、ObsPy

### 推理

```python
from seispolarity import WaveformDataset, get_dataset_path

# 自动下载数据集
data_path = get_dataset_path("SCSN", "train", cache_dir="./datasets")

# 加载数据集
dataset = WaveformDataset(path=data_path, name="SCSN", preload=False)
loader = dataset.get_dataloader(batch_size=1024, num_workers=4)

# 使用预训练模型
from seispolarity.models import PPNet
model = PPNet(num_fm_classes=3)
# model.load_state_dict(...)  # 加载预训练权重
```

### 数据增强

```python
from seispolarity import GenericGenerator, Demean, Normalize, RandomTimeShift

generator = GenericGenerator(dataset)
generator.add_augmentations([
    Demean(),
    Normalize(amp_norm_type="peak"),
    RandomTimeShift(max_shift=10)
])
loader = generator.get_dataloader(batch_size=256)
```

## 文档

```{toctree}
:maxdepth: 1
:caption: 文档目录

pages/installation.md
pages/datasets/overview.md
pages/models/overview.md
pages/training/overview.md
pages/augmentation/overview.md
pages/api/overview.md
```

## 引用

如果您在研究中使用了 SeisPolarity，请引用：

```bibtex
@software{seispolarity,
  title = {SeisPolarity: A Framework for Seismic First-Motion Polarity Picking},
  author = {He, Xingchen},
  year = {2025},
  url = {https://github.com/Chuan1937/SeisPolarity}
}
```

## 相关链接

- GitHub: https://github.com/Chuan1937/SeisPolarity
- Hugging Face: https://huggingface.co/HeXingChen/SeisPolarity-Model
