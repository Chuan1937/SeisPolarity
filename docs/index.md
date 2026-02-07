# 欢迎使用 SeisPolarity

:::{figure} _static/seispolarity_logo_title.svg
:align: center
:::

**SeisPolarity** 是一个全面的地震初至极性拾取框架。

SeisPolarity 提供统一的 API，用于将深度学习模型应用于地震波形进行极性分类，支持多个数据集和模型。

## 功能特性

- **统一数据接口**：支持 SCSN、Txed、DiTing、Instance、PNW 数据集，自动下载
- **多种模型**：Ross、Eqpolarity、APP、DiTingMotion、CFM、RPNet、PolarCAP，含预训练权重
- **灵活数据加载**：RAM/磁盘流式传输，支持任意规模数据集
- **高级推理**：`Predictor` 类，自动从 Hugging Face/ModelScope 下载预训练模型
- **数据增强**：支持多种增强技术的平衡采样
- **统一训练**：支持检查点保存和早停机制
- **跨平台支持**：支持 Linux、macOS 和 Windows

## 快速开始

### 安装

```bash
pip install seispolarity
```

```bash
git clone https://github.com/Chuan1937/SeisPolarity.git
cd SeisPolarity
pip install -e .
```

### 推理

```python
from seispolarity.inference import Predictor
import numpy as np

predictor = Predictor("ross")  # 可选: "ross", "eqpolarity", "diting_motion"
waveforms = np.random.randn(10, 400)  # (Batch, Length)
predictions = predictor.predict(waveforms)
# 返回: [0, 1, 2, ...]  (0: Up, 1: Down, 2: Unknown)
```

### 数据加载

```python
from seispolarity.data import WaveformDataset

# 大数据集使用磁盘流式传输
dataset = WaveformDataset(path="data.hdf5", name="SCSN", preload=False)
loader = dataset.get_dataloader(batch_size=1024, num_workers=4)

# 小数据集使用 RAM 预加载
dataset_ram = WaveformDataset(path="data.hdf5", name="SCSN", preload=True)
```

### 自动下载数据集

```python
from seispolarity.data import PNW
from pathlib import Path

output_dir = Path('datasets/PNW')
processor = PNW(
    csv_path=str(output_dir / 'PNW.csv'),
    hdf5_path=str(output_dir / 'PNW.hdf5'),
    output_polarity=str(output_dir),
    auto_download=True,  # 自动下载缺失数据
    use_hf=False,         # 使用 ModelScope 而非 Hugging Face
    component='Z',        # 提取垂直分量
    sampling_rate=100     # 目标采样率 (Hz)
)
processor.process()
```

### 训练

```python
from seispolarity.models.scsn import SCSN
from seispolarity.training import Trainer, TrainingConfig

model = SCSN(num_fm_classes=3)
config = TrainingConfig(
    batch_size=256,
    epochs=50,
    learning_rate=1e-4,
    device='cuda'  # 或 'cpu'
)
trainer = Trainer(model=model, dataset=dataset, config=config)
trainer.train()
```

### 数据增强

```python
from seispolarity.generate import GenericGenerator
from seispolarity.generate.augmentation import (
    Demean, Normalize, RandomTimeShift, 
    BandpassFilter, PolarityInversion
)

generator = GenericGenerator(dataset)
generator.add_augmentations([
    Demean(),
    Normalize(amp_norm_type="peak"),
    RandomTimeShift(max_shift=10),
    BandpassFilter(freqmin=1, freqmax=20),
    PolarityInversion(prob=0.5)
])
loader = generator.get_dataloader(batch_size=256)
```

## 支持的数据集

| 数据集 | 描述 | 样本数 | 自动下载 |
|-------|------|--------|---------|
| SCSN | 南加州地震台网 | 100k+ | 是 |
| Txed | 德州地震数据 | 50k+ | 是 |
| DiTing | 中国地震台网 | 80k+ | 是 |
| Instance | 基于实例的数据集 | 30k+ | 否(需申请) |
| PNW | 太平洋西北地区 | 20k+ | 是 |

## 支持的模型

| 模型 | 输入长度 | 类别 |
|-------|---------|------|
| Ross (SCSN) | 400 | 3 (U/D/N) |
| Eqpolarity | 600 | 2 (U/D) |
| DiTingMotion | 128 | 3 (U/D/N) |
| CFM | 160 | 2 (U/D) |
| RPNet | 400 | 2 (U/D) |
| PolarCAP | 64 | 2 (U/D) |
| APP | 400 | 3 (U/D/N) |

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

## 示例

参见 `examples/` 目录中的完整示例：
- [数据集 API 使用](../examples/datasets_api.ipynb) - 数据集加载和使用示例
- [预测 API 使用](../examples/predict_api.ipynb) - 模型推理示例
- [模型 API 使用](../examples/model_api.ipynb) - 模型架构和初始化
- [训练 API 使用](../examples/train_api.ipynb) - 训练流程示例

## 引用

文章将在后续发布，敬请关注。

## 相关链接

- GitHub: https://github.com/Chuan1937/SeisPolarity
- Documentation: https://seispolarity.readthedocs.io/en/latest/index.html
- Hugging Face: https://huggingface.co/chuanjun1978/SeisPolarity-Model
- ModelScope: https://modelscope.cn/models/chuanjun1978/SeisPolarity-Model

## 联系方式

如有问题和支持，请在 GitHub 上提交 issue 或联系: [chuanjun1978@gmail.com]
