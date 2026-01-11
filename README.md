# SeisPolarity

A Toolbox for First-Motion Polarity Determination in Seismology.

SeisPolarity is a framework for seismic polarity picking, designed to provide unified data interfaces, model encapsulations, and evaluation tools to facilitate rapid experimentation and comparison of polarity determination models.

## Features

- **Unified Data Interface**: Seamless access to seismic datasets (e.g., SCSN) using a standardized API (`SCSNDataset`).
- **Dual Loading Modes**:
    - **RAM Mode**: Preload entire datasets into memory with progress tracking for maximum training speed.
    - **Disk Streaming Mode**: Efficient, parallelized streaming for datasets larger than memory.
- **High-Level Inference**: `Predictor` class for easy model inference locally or from pre-trained weights.
- **Remote Data Support**: Automatic downloading of datasets and models from Hugging Face / Remote Repositories.

## Installation

```bash
git clone https://github.com/Chuan1937/SeisPolarity.git
cd SeisPolarity
pip install -e .
```

## Quick Start

### 1. Inference with Pre-trained Model

Use the high-level `Predictor` to classify waveforms using models like Ross et al. (2018).

```python
from seispolarity.inference import Predictor
import numpy as np

# Initialize Predictor (Auto-downloads model if needed)
predictor = Predictor("ross")

# Predict on numpy array (Batch, Length)
# Input should be centered on P-arrival
waveforms = np.random.randn(10, 400) 
predictions = predictor.predict(waveforms)

print(predictions) # [0, 1, 2, ...] (0: Up, 1: Down, 2: Unknown)
```

### 2. Working with Datasets

The `SCSNDataset` handles large HDF5 seismic archives efficiently.

```python
from seispolarity.data import SCSNDataset

# Mode 1: Disk Streaming (Low RAM, suitable for huge files)
dataset = SCSNDataset(h5_path="path/to/data.hdf5", preload=False, split="test")
loader = dataset.get_dataloader(batch_size=1024, num_workers=4)

# Mode 2: RAM Preloading (High Speed, requires sufficient RAM)
# Shows a progress bar during loading
dataset_ram = SCSNDataset(h5_path="path/to/data.hdf5", preload=True)
```

### 3. Unified Inference on Large Datasets

Combine `SCSNDataset` and `Predictor` to run inference on millions of samples.

```python
# See tests/predict_ross_scsn.py for full example
loader = dataset.get_dataloader(...)
probs, labels = predictor.predict_from_loader(loader, return_probs=True)
```

## Project Structure

- `seispolarity/`: Core package source code.
    - `data/`: Dataset implementations (`scsn.py`, etc.).
    - `models/`: Neural network architectures.
    - `inference.py`: High-level inference interfaces (`Predictor`).
    - `training/`: Training loops and utilities.
- `tests/`: Example scripts and unit tests.
    - `predict_ross_scsn.py`: Example for large-scale inference.
    - `train_ross_scsn.py`: Example for training models.

---

# SeisPolarity (中文说明)

SeisPolarity 是一个地震极性拾取框架，旨在提供统一的数据接口、模型封装和评测工具，帮助快速试验和比较极性判读模型。

## 特性

- **统一数据接口**：通过标准 API (`SCSNDataset`) 无缝访问地震数据集（如 SCSN）。
- **双重加载模式**：
    - **内存模式 (RAM Mode)**：为了极致的训练速度，支持将整个数据集预加载到内存（带进度条）。
    - **磁盘流式模式 (Disk Streaming Mode)**：针对超大数据集，支持高效并发的磁盘流式读取，无需大量内存。
- **高级推理接口**：提供 `Predictor` 类，方便调用预训练模型进行预测。
- **远程支持**：支持从 Hugging Face 或远程仓库自动下载数据集和模型权重。

## 安装

```bash
git clone https://github.com/Chuan1937/SeisPolarity.git
cd SeisPolarity
pip install -e .
```

## 快速开始

### 1. 使用预训练模型预测

使用 `Predictor` 高级接口调用 Ross et al. (2018) 等模型。

```python
from seispolarity.inference import Predictor
import numpy as np

# 初始化预测器 (如模型不存在会自动下载)
predictor = Predictor("ross")

# 对 numpy 数组进行预测 (Batch, Length)
# 输入应以 P 波到达为中心
waveforms = np.random.randn(10, 400) 
predictions = predictor.predict(waveforms)

print(predictions) # [0, 1, 2, ...] (0: Up, 1: Down, 2: Unknown)
```

### 2. 使用数据集

`SCSNDataset` 可以高效处理大型 HDF5 地震数据归档。

```python
from seispolarity.data import SCSNDataset

# 模式 1: 磁盘流式 (低内存占用，适合海量数据)
dataset = SCSNDataset(h5_path="path/to/data.hdf5", preload=False, split="test")
loader = dataset.get_dataloader(batch_size=1024, num_workers=4)

# 模式 2: 内存预加载 (高速度，需足够内存)
# 加载过程会显示进度条
dataset_ram = SCSNDataset(h5_path="path/to/data.hdf5", preload=True)
```

### 3. 大规模数据集统一推理

结合 `SCSNDataset` 和 `Predictor` 对数百万样本进行推理。

```python
# 完整示例请参考 tests/predict_ross_scsn.py
loader = dataset.get_dataloader(...)
probs, labels = predictor.predict_from_loader(loader, return_probs=True)
```

## 项目结构

- `seispolarity/`: 核心代码库。
    - `data/`: 数据集实现 (`scsn.py` 等)。
    - `models/`: 神经网络模型结构。
    - `inference.py`: 高级推理接口 (`Predictor`)。
    - `training/`: 训练循环和工具。
- `tests/`: 示例脚本和测试。
    - `predict_ross_scsn.py`: 大规模推理示例脚本。
    - `train_ross_scsn.py`: 模型训练示例脚本。

