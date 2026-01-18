# SeisPolarity

A Toolbox for First-Motion Polarity Determination in Seismology.

SeisPolarity is a framework for seismic polarity picking, designed to provide unified data interfaces, model encapsulations, and evaluation tools to facilitate rapid experimentation and comparison of polarity determination models.

## Features

- **Unified Data Interface**: Seamless access to seismic datasets (e.g., SCSN, Txed) using standardized APIs (`SCSNDataset`).
- **Dual Loading Modes**:
    - **RAM Mode**: Preload entire datasets into memory with progress tracking for maximum training speed.
    - **Disk Streaming Mode**: Efficient, parallelized streaming for datasets larger than memory.
- **High-Level Inference**: `Predictor` class for easy model inference locally or from pre-trained weights.
- **Remote Data Support**: Automatic downloading of datasets and models from Hugging Face / Remote Repositories.
- **Multiple Model Architectures**: Support for Ross (2018), Eqpolarity, APP, DiTingMotion, and custom models.
- **Cross-Domain Training**: Tools for domain adaptation and transfer learning between different seismic datasets.

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

### 3. Training Models

Train Ross or Eqpolarity models on SCSN dataset:

```python
# See tests/train_ross_scsn.py for full example
from seispolarity.models.scsn import SCSN
from seispolarity.training import Trainer, TrainingConfig

# Initialize model
model = SCSN()

# Configure training
config = TrainingConfig(
    h5_path="path/to/data.hdf5",
    batch_size=256,
    epochs=50,
    learning_rate=1e-4,
    checkpoint_dir="./checkpoints"
)

# Create trainer and start training
trainer = Trainer(model=model, dataset=dataset, config=config)
trainer.train()
```

### 4. Unified Inference on Large Datasets

Combine `SCSNDataset` and `Predictor` to run inference on millions of samples.

```python
# See tests/predict_ross_scsn.py for full example
loader = dataset.get_dataloader(...)
probs, labels = predictor.predict_from_loader(loader, return_probs=True)
```

## Project Structure

```
SeisPolarity/
├── seispolarity/                    # Core package source code
│   ├── data/                        # Dataset implementations
│   │   ├── scsn.py                  # SCSN dataset loader
│   │   └── __init__.py
│   ├── models/                      # Neural network architectures
│   │   ├── base.py                  # Base model class
│   │   ├── scsn.py                  # Ross (2018) model
│   │   ├── eqpolarity.py            # Eqpolarity (Transformer) model
│   │   ├── app.py                   # APP (U-Net + Attention) model
│   │   ├── diting_motion.py         # DiTingMotion model
│   │   └── __init__.py
│   ├── training/                    # Training loops and utilities
│   ├── inference.py                 # High-level inference interfaces (Predictor)
│   ├── annotations.py               # Data annotation classes
│   ├── config.py                    # Configuration utilities
│   └── util/                        # Utility functions
├── tests/                           # Example scripts and unit tests
│   ├── train_ross_scsn.py           # Train Ross model on SCSN
│   ├── train_eqpolarity_scsn.py     # Train Eqpolarity model on SCSN
│   ├── train_diting_DitingScsn.py   # Train DiTingMotion model on SCSN
│   ├── predict_ross_scsn.py         # Large-scale inference example (Ross model)
│   ├── predict_eqpolarity_scsn.py   # Large-scale inference example (Eqpolarity model)
│   ├── predict_diting_scsn.py       # Large-scale inference example (DiTingMotion model with forced U/D output)
│   ├── test_models.py               # Model unit tests
│   └── test_data_structure.py       # Data structure tests
├── checkpoints_ross_scsn/           # Ross model checkpoints
├── checkpoints_eqpolarity_scsn/     # Eqpolarity model checkpoints
├── pretrained_model/                # Pre-trained model weights
│   ├── Eqpolarity/                  # Eqpolarity pre-trained models
│   ├── Ross/                        # Ross pre-trained models
│   └── DiTingMotion/                # DiTingMotion pre-trained models
├── datasets/                        # Dataset storage (optional)
├── Polarity-model/                  # Cross-domain evaluation notebooks
├── Methods Valid.ipynb              # Data validation and analysis notebook
├── Update_model.ipynb               # Model update notebook
├── pyproject.toml                   # Python project configuration
└── README.md                        # This file
```

## Model Architectures

### 1. Ross (2018) Model
- **Architecture**: CNN-based model from Ross et al. (2018)
- **Input**: 400 samples (200 before + 200 after P-wave arrival)
- **Dataset**: SCSN (Southern California Seismic Network)
- **Preprocessing**: Crop 200 samples before and after P-wave arrival at sample 300
- **Output**: 3 classes (Up, Down, Unknown)

### 2. Eqpolarity Model
- **Architecture**: Compact Convolutional Transformer (CCT)
- **Input**: 600 samples (full waveform, no cropping)
- **Dataset**: SCSN + Txed (for fine-tuning)
- **Preprocessing**: No cropping, uses full 600-sample waveforms
- **Output**: Binary classification (adapted for 2-class CrossEntropy)

### 3. APP Model
- **Architecture**: U-Net like with LSTM and Multi-Head Self-Attention
- **Features**: Dual output (segmentation mask + classification)
- **Applications**: Advanced polarity picking with attention mechanisms

### 4. DiTingMotion Model
- **Architecture**: Custom CNN with growth blocks
- **Input**: 128 samples with 2 channels
- **Applications**: Motion detection and polarity analysis

## Dataset Processing

### SCSN Dataset
- **Source**: Southern California Seismic Network (2000-2017)
- **Format**: HDF5 files
- **Waveform Length**: 600 samples (6 seconds)
- **P-wave Arrival**: Sample 300 (3 seconds)
- **Labels**: 0 (Up), 1 (Down), 2 (Unknown)
- **Files**:
  - `scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5` - Training data
  - `scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5` - Test data
  - `scsn_p_2000_2017_6sec_0.5r_pick_train.hdf5` - P-wave picking training

### Txed Dataset
- **Source**: Texas seismic data
- **Format**: HDF5 + CSV
- **Label Distribution**:
  - U (Up): 14,671 samples
  - D (Down): 8,663 samples
  - unknown: 288,897 samples
- **Fine-tuning**: Use 10% of data (9% train, 1% validation) for domain adaptation

---

# SeisPolarity (中文说明)

SeisPolarity 是一个地震极性拾取框架，旨在提供统一的数据接口、模型封装和评测工具，帮助快速试验和比较极性判读模型。

## 特性

- **统一数据接口**：通过标准 API (`SCSNDataset`) 无缝访问地震数据集（如 SCSN, Txed）。
- **双重加载模式**：
    - **内存模式 (RAM Mode)**：为了极致的训练速度，支持将整个数据集预加载到内存（带进度条）。
    - **磁盘流式模式 (Disk Streaming Mode)**：针对超大数据集，支持高效并发的磁盘流式读取，无需大量内存。
- **高级推理接口**：提供 `Predictor` 类，方便调用预训练模型进行预测。
- **远程支持**：支持从 Hugging Face 或远程仓库自动下载数据集和模型权重。
- **多模型架构**：支持 Ross (2018)、Eqpolarity、APP、DiTingMotion 等多种模型。
- **跨域训练**：提供不同地震数据集之间的域适应和迁移学习工具。

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

### 3. 训练模型

在 SCSN 数据集上训练 Ross 或 Eqpolarity 模型：

```python
# 完整示例请参考 tests/train_ross_scsn.py
from seispolarity.models.scsn import SCSN
from seispolarity.training import Trainer, TrainingConfig

# 初始化模型
model = SCSN()

# 配置训练参数
config = TrainingConfig(
    h5_path="path/to/data.hdf5",
    batch_size=256,
    epochs=50,
    learning_rate=1e-4,
    checkpoint_dir="./checkpoints"
)

# 创建训练器并开始训练
trainer = Trainer(model=model, dataset=dataset, config=config)
trainer.train()
```

### 4. 大规模数据集统一推理

结合 `SCSNDataset` 和 `Predictor` 对数百万样本进行推理。

```python
# 完整示例请参考 tests/predict_ross_scsn.py
loader = dataset.get_dataloader(...)
probs, labels = predictor.predict_from_loader(loader, return_probs=True)
```

## 模型架构

### 1. Ross (2018) 模型
- **架构**: 基于 Ross et al. (2018) 的 CNN 模型
- **输入**: 400 个样本点（P波到达点前后各200点）
- **数据集**: SCSN（南加州地震网络）
- **预处理**: 在300样本点处截取前后各200个样本点
- **输出**: 3个类别（正向、负向、不确定）

### 2. Eqpolarity 模型
- **架构**: 紧凑卷积变换器 (Compact Convolutional Transformer)
- **输入**: 600 个样本点（完整波形，不截取）
- **数据集**: SCSN + Txed（用于微调）
- **预处理**: 不截取，使用完整的600个样本点波形
- **输出**: 二元分类（适配为2类CrossEntropy）

### 3. APP 模型
- **架构**: U-Net 风格，包含 LSTM 和多头自注意力机制
- **特点**: 双输出（分割掩码 + 分类）
- **应用**: 带注意力机制的高级极性拾取

### 4. DiTingMotion 模型
- **架构**: 自定义 CNN 带增长块
- **输入**: 128个样本点，2个通道
- **应用**: 运动检测和极性分析

## 数据集处理

### SCSN 数据集
- **来源**: 南加州地震网络 (2000-2017)
- **格式**: HDF5 文件
- **波形长度**: 600个样本点（6秒）
- **P波到达点**: 第300个样本点（3秒处）
- **标签**: 0（正向）、1（负向）、2（不确定）
- **文件**:
  - `scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5` - 训练数据
  - `scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5` - 测试数据
  - `scsn_p_2000_2017_6sec_0.5r_pick_train.hdf5` - P波拾取训练数据

### Txed 数据集
- **来源**: 德克萨斯州地震数据
- **格式**: HDF5 + CSV
- **标签分布**:
  - U（正向）: 14,671 个样本
  - D（负向）: 8,663 个样本
  - unknown（不确定）: 288,897 个样本
- **微调策略**: 使用10%的数据（9%训练，1%验证）进行域适应

## 实验结果

### Eqpolarity 在 SCSN 上的训练结果
根据训练日志 (`pretrained_model/Eqpolarity/Eqpolarity_SCSN.txt`):
- **最佳验证准确率**: 97.24% (第5个epoch)
- **最佳验证损失**: 0.0860
- **训练准确率**: 97.16%
- **训练损失**: 0.0883

### 跨域评估
项目包含多个跨域评估笔记本：
- `Polarity-model/` 目录下的跨域评估脚本
- 支持 SCSN、Txed、DiTing、Instance 等数据集间的迁移学习

## 项目结构

```
SeisPolarity/
├── seispolarity/                    # 核心代码库
│   ├── data/                        # 数据集实现
│   │   ├── scsn.py                  # SCSN 数据集加载器
│   │   └── __init__.py
│   ├── models/                      # 神经网络模型结构
│   │   ├── base.py                  # 基础模型类
│   │   ├── scsn.py                  # Ross (2018) 模型
│   │   ├── eqpolarity.py            # Eqpolarity (Transformer) 模型
│   │   ├── app.py                   # APP (U-Net + Attention) 模型
│   │   ├── diting_motion.py         # DiTingMotion 模型
│   │   └── __init__.py
│   ├── training/                    # 训练循环和工具
│   ├── inference.py                 # 高级推理接口 (Predictor)
│   ├── annotations.py               # 数据标注类
│   ├── config.py                    # 配置工具
│   └── util/                        # 工具函数
├── tests/                           # 示例脚本和单元测试
│   ├── train_ross_scsn.py           # 在 SCSN 上训练 Ross 模型
│   ├── train_eqpolarity_scsn.py     # 在 SCSN 上训练 Eqpolarity 模型
│   ├── train_diting_DitingScsn.py   # 在 SCSN 上训练 DiTingMotion 模型
│   ├── predict_ross_scsn.py         # 大规模推理示例
│   ├── predict_eqpolarity_scsn.py   # Eqpolarity 模型推理示例
│   ├── predict_diting_scsn.py       # 大规模推理示例 (DiTingMotion 模型，强制输出 U/D)
│   ├── test_models.py               # 模型单元测试
│   └── test_data_structure.py       # 数据结构测试
├── checkpoints_ross_scsn/           # Ross 模型检查点
├── checkpoints_eqpolarity_scsn/     # Eqpolarity 模型检查点
├── checkpoints_diting_DitingScsn/   # DiTingMotion 模型检查点
├── pretrained_model/                # 预训练模型权重
│   ├── Eqpolarity/                  # Eqpolarity 预训练模型
│   ├── Ross/                        # Ross 预训练模型
│   └── DiTingMotion/                # DiTingMotion 预训练模型
├── datasets/                        # 数据集存储（可选）
├── Polarity-model/                  # 跨域评估笔记本
├── Methods Valid.ipynb              # 数据验证和分析笔记本
├── Update_model.ipynb               # 模型更新笔记本
├── pyproject.toml                   # Python 项目配置
└── README.md                        # 本文档
```

