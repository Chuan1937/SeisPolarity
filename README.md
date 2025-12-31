# SeisPolarity

SeisPolarity is a framework for seismic polarity picking, designed to provide unified data interfaces, model encapsulations, and evaluation tools to facilitate rapid experimentation and comparison of polarity determination models. It builds upon robust data handling capabilities for seismic data.

## Features

- **Unified Data Interface**: Seamless access to various seismic datasets (e.g., SCEDC, STEAD, INSTANCE) using a standardized API.
- **Modular Design**: Clear separation between data loading (`data`), model definitions (`models`), and annotation structures (`annotations`).
- **Extensible**: Easy to add new models or datasets.
- **Remote Data Support**: Built-in support for downloading datasets from Hugging Face.

## Installation

```bash
git clone https://github.com/Chuan1937/SeisPolarity.git
cd SeisPolarity
pip install -e .
```

## Quick Start

### Loading a Dataset

SeisPolarity provides direct access to many standard datasets.

```python
import seispolarity.data as sbd

# Load the SCEDC dataset
# This will automatically download metadata and waveforms if not present
dataset = sbd.SCEDC(sampling_rate=100)
print(dataset)

# Access waveforms and metadata
waveforms, metadata = dataset.get_sample(0)
print(f"Waveform shape: {waveforms.shape}")
print(f"Metadata: {metadata}")
```

## Project Structure

- `seispolarity/`: Core package source code.
    - `data/`: Dataset implementations and base classes.
    - `models/`: Model base classes and wrappers.
    - `annotations.py`: Data structures for picks and polarity labels.
    - `config.py`: Configuration management.
- `tests/`: Unit tests.

---

# SeisPolarity (中文说明)

SeisPolarity 是一个地震极性拾取框架，旨在提供统一的数据接口、模型封装和评测工具，帮助快速试验和比较极性判读模型。本项目基于成熟的地震数据处理能力构建。

## 特性

- **统一数据接口**：使用标准 API 无缝访问各种地震数据集（如 SCEDC, STEAD, INSTANCE）。
- **模块化设计**：数据加载 (`data`)、模型定义 (`models`) 和标注结构 (`annotations`) 清晰分离。
- **可扩展性**：易于添加新模型或数据集。
- **远程数据支持**：内置支持从 Hugging Face 下载数据集。

## 安装

```bash
git clone https://github.com/Chuan1937/SeisPolarity.git
cd SeisPolarity
pip install -e .
```

## 快速开始

### 加载数据集

SeisPolarity 提供了对许多标准数据集的直接访问。

```python
import seispolarity.data as sbd

# 加载 SCEDC 数据集
# 如果本地不存在，将自动下载元数据和波形
dataset = sbd.SCEDC(sampling_rate=100)
print(dataset)

# 获取波形和元数据
waveforms, metadata = dataset.get_sample(0)
print(f"波形形状: {waveforms.shape}")
print(f"元数据: {metadata}")
```

## 项目结构

- `seispolarity/`: 核心包源代码。
    - `data/`: 数据集实现和基类。
    - `models/`: 模型基类和封装。
    - `annotations.py`: 拾取和极性标签的数据结构。
    - `config.py`: 配置管理。
- `tests/`: 单元测试。
