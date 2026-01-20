# SeisPolarity

A comprehensive framework for seismic first-motion polarity picking.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)](https://pytorch.org/)

地震初动极性判读的综合框架

## Features / 特性

- Unified data interface for SCSN, Txed, DiTing, Instance, PNW datasets / 统一数据接口
- Multiple models: Ross, Eqpolarity, APP, DiTingMotion, CFM, RPNet, PolarCAP / 多模型支持
- RAM/Disk streaming for datasets of any size / 灵活的数据加载方式
- High-level `Predictor` with auto-download from Hugging Face / 高级推理接口
- Data augmentation with balanced sampling / 数据增强与平衡采样
- Unified `Trainer` with checkpointing & early stopping / 统一训练工具

## Installation / 安装

```bash
git clone https://github.com/Chuan1937/SeisPolarity.git
cd SeisPolarity
pip install -e .
```

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.1, NumPy, h5py, ObsPy

## Quick Start / 快速开始

### Inference / 推理

```python
from seispolarity.inference import Predictor
import numpy as np

predictor = Predictor("ross")  # Options: "ross", "eqpolarity", "diting_motion"
waveforms = np.random.randn(10, 400)  # (Batch, Length)
predictions = predictor.predict(waveforms)
# [0, 1, 2, ...]  (0: Up, 1: Down, 2: Unknown)
```

### Dataset / 数据集

```python
from seispolarity.data import WaveformDataset

# Disk streaming / 磁盘流式加载
dataset = WaveformDataset(path="data.hdf5", name="SCSN", preload=False)
loader = dataset.get_dataloader(batch_size=1024, num_workers=4)

# RAM preloading / 内存预加载
dataset_ram = WaveformDataset(path="data.hdf5", name="SCSN", preload=True)
```

### Training / 训练

```python
from seispolarity.models.scsn import SCSN
from seispolarity.training import Trainer, TrainingConfig

model = SCSN(num_fm_classes=3)
config = TrainingConfig(batch_size=256, epochs=50, learning_rate=1e-4)
trainer = Trainer(model=model, dataset=dataset, config=config)
trainer.train()
```

### Augmentation / 数据增强

```python
from seispolarity.generate import GenericGenerator
from seispolarity.generate.augmentation import Demean, Normalize, RandomTimeShift

generator = GenericGenerator(dataset)
generator.add_augmentations([
    Demean(),
    Normalize(amp_norm_type="peak"),
    RandomTimeShift(max_shift=10)
])
loader = generator.get_dataloader(batch_size=256)
```

## Models / 模型

| Model | Input Length | Classes | Best Accuracy |
|-------|--------------|---------|---------------|
| Ross (SCSN) | 400 | 3 (U/D/N) | 97%+ |
| Eqpolarity | 600 | 2 (U/D) | 97.24% |
| DiTingMotion | 128 | 3 (U/D/N) | - |
| CFM | 160 | 1 | - |
| RPNet | 400 | 2 (U/D) | - |
| PolarCAP | 64 | 2 (U/D) | - |

## Project Structure / 项目结构

```
SeisPolarity/
├── seispolarity/
│   ├── models/          # Neural network architectures / 模型
│   ├── data/            # Dataset implementations / 数据集
│   ├── training/        # Trainer, TrainingConfig / 训练工具
│   ├── generate/        # Data augmentation / 数据增强
│   └── inference.py     # Predictor class / 推理接口
├── tests/              # Example scripts / 示例脚本
└── pretrained_model/   # Pre-trained weights / 预训练权重
```

## License / 许可证

[License](LICENSE)

## Links / 链接

- GitHub: https://github.com/Chuan1937/SeisPolarity
- Hugging Face: https://huggingface.co/HeXingChen/SeisPolarity-Model
