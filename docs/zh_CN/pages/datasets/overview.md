# 数据集总览

SeisPolarity 提供统一的接口来访问和处理多个地震极性数据集。所有数据集均以 HDF5 格式存储，以实现高效存储和流式加载。

## 可用数据集

| 数据集 | 来源 | 规模 | 类别 | 格式 |
|---------|--------|------|---------|--------|
| SCSN | 南加州地震台网 | 大 | U/D/N | HDF5 |
| Txed | 得克萨斯地震数据集 | 中 | U/D | HDF5 |
| DiTing | 中国地震台网中心 | 中 | U/D/N | HDF5 |
| Instance | 全球地震数据 | 中 | U/D/N | HDF5 |
| PNW | 太平洋西北地区 | 大 | U/D/N | HDF5 |

## 数据集

### SCSN

南加州地震台网（SCSN）数据集包含来自南加州地震台网、带有极性标注的地震波形。该数据集涵盖 2000-2020 年的地震事件，是一个具有人工标注的高质量数据集。

```{image} ../datasets/SCSN.png
:alt: SCSN
:align: center
```

#### 注意

数据集大小：waveforms.hdf5 约 660Gb，metadata.csv 约 2.2Gb
极性子集 SCSN 约 15 GB

#### 引用

Cheng, Y., Ross, Z. E., Hauksson, E., Ben-Zion, Y. (2023). Refined earthquake focal mechanism catalog for southern California derived with deep learning algorithms. Journal of Geophysical Research: Solid Earth, 128, e2022JB025975. https://doi.org/10.1029/2022JB025975

Ross, Z. E., Meier, M.-A., Hauksson, E. (2018). P wave arrival picking and first-motion polarity determination with deep learning. Journal of Geophysical Research: Solid Earth, 123, 5405-5416. https://doi.org/10.1029/2018JB015510

---

### Txed

得克萨斯地震数据集（TXED）是来自得克萨斯州的区域地震信号基准数据集。该数据集包含大量地震事件和噪声波形，是地震学机器学习的重要数据资源。

```{image} ../datasets/TXED.png
:alt: Txed
:align: center
```

#### 平衡策略

推荐策略：**极性反转 (1:1:1)**

该策略构建一个上、下、未知样本等比例分布的平衡数据集：
- 每个 Up 和 Down 样本生成两个样本（原始 + 极性反转）
- 补充 Unknown 样本，使其数量与 (Up + Down) 样本总数一致
- 最终分布：Up = 1/3，Down = 1/3，Unknown = 1/3

```python
from seispolarity import BalancedPolarityGenerator

generator = BalancedPolarityGenerator(
    dataset,
    strategy="polarity_inversion"
)
```

#### 注意

数据集大小：waveforms.hdf5 约 70Gb，metadata.csv 120Mb

#### 引用

Chen, Y., Savvaidis, A., Saad, O. M., Huang, G.-C. D., Siervo, D., O'Sullivan, V., McCabe, C., Uku, B., Fleck, P., Burke, G., Alvarez, N. L., Domino, J., & Grigoratos, I. (2024). TXED: The Texas Earthquake Dataset for AI. Seismological Research Letters, 95(6), 1-13. https://doi.org/10.1785/0220230327

---

### DiTing

DiTing 数据集是专为人工智能地震学研究设计的大规模中国地震基准数据集。该数据集包含超过 640,000 个高质量的 P 波初动极性标签，覆盖中国 1,300 多个宽带和短周期地震台站。

```{image} ../datasets/DITING.jpg
:alt: Diting
:align: center
```

#### 引用

Zhao, M., Xiao, Z., Chen, S., & Fang, L. (2023). DiTing: A large-scale Chinese seismic benchmark dataset for artificial intelligence in seismology. Earthquake Science, 36(2), 84-94. https://doi.org/10.1016/j.eqs.2022.01.022

#### 数据下载

可在以下地址申请下载该数据集：https://data.earthquake.cn/

---

### Instance

INSTANCE 数据集是由意大利国家地球物理与火山学研究所（INGV）编制的意大利地震波形数据集，专为机器学习应用设计。该数据集包含近 120 万条三分量波形记录，是地震学研究的重要资源。

```{image} ../datasets/INSTANCE.png
:alt: Instance
:align: center
```

#### 平衡策略

推荐策略：**极性反转 (1:1:1)**

该策略构建一个上、下、未知样本等比例分布的平衡数据集：
- 每个 Up 和 Down 样本生成两个样本（原始 + 极性反转）
- 补充 Unknown 样本，使其数量与 (Up + Down) 样本总数一致
- 最终分布：Up = 1/3，Down = 1/3，Unknown = 1/3

```python
from seispolarity import BalancedPolarityGenerator

generator = BalancedPolarityGenerator(
    dataset,
    strategy="polarity_inversion"
)
```

#### 注意

数据集大小：
- waveforms (counts) 约 160Gb
- waveforms (ground motion units) 约 310Gb

#### 引用

Michelini, A., Cianetti, S., Gaviano, S., Giunchi, C., Jozinović, D., & Lauciani, V. (2021). INSTANCE – The Italian Seismic Dataset For Machine Learning. Earth System Science Data, 13, 5509–5542. https://doi.org/10.5194/essd-13-5509-2021

---

### PNW

太平洋西北地区（PNW）数据集是面向机器学习整理的数据集，包含来自太平洋西北地区的多种地震信号。该数据集由太平洋西北地震台网编制，涵盖地震、爆破和噪声等多种地震事件类型。

```{image} ../datasets/PNW.png
:alt: PNW
:align: center
```

#### 平衡策略

推荐策略：**基于最小类 (1:1:1)**

该策略从所有类别中按最小类别数量等量采样，构建平衡数据集：
- 统计每个极性类别（Up、Down、Unknown）的样本数
- 确定所有类别中的最小数量
- 从每个类别中等量采样至最小数量
- 最终分布：Up = 1/3，Down = 1/3，Unknown = 1/3

```python
from seispolarity import BalancedPolarityGenerator

generator = BalancedPolarityGenerator(
    dataset,
    strategy="min_based"
)
```

#### 引用

Ni, Y., Hutko, A., Skene, F., Denolle, M., Malone, S., Bodin, P., Hartog, R., & Wright, A. (2023). Curated Pacific Northwest AI-ready Seismic Dataset. Seismica, 2(1), 368. https://doi.org/10.26443/seismica.v2i1.368

## 加载数据集

### 自动下载

SeisPolarity 可以自动下载数据集：

```python
from seispolarity import get_dataset_path, WaveformDataset

# 从 Hugging Face 下载（默认）
data_path = get_dataset_path("SCSN", "train", cache_dir="./datasets")

# 或使用 ModelScope（推荐中国用户使用）
data_path = get_dataset_path("SCSN", "train", use_hf=False)
```

### 从本地文件加载

```python
from seispolarity import WaveformDataset

# 磁盘流式加载（适用于大型数据集）
dataset = WaveformDataset(
    path="data/scsn_train.hdf5",
    name="SCSN_Train",
    preload=False
)

# RAM 预加载（适用于小型数据集）
dataset = WaveformDataset(
    path="data/scsn_train.hdf5",
    name="SCSN_Train",
    preload=True
)
```

## 数据集 API

### WaveformDataset

加载波形数据的主要类。

```python
from seispolarity import WaveformDataset

dataset = WaveformDataset(
    path="data.hdf5",          # HDF5 文件路径
    name="SCSN",               # 数据集名称
    preload=False,             # 是否预加载到内存
    data_key="X",              # 波形数据的 HDF5 键
    label_key="Y",             # 标签的 HDF5 键
    p_pick_position=300,      # P 波到时位置
    pick_key="p_pick",        # 使用 p_pick 作为 P 波到时点
    crop_left=200,             # P 波到时前的采样点数
    crop_right=200,            # P 波到时后的采样点数
    allowed_labels=[0, 1, 2]   # 允许的标签 (0: Up, 1: Down, 2: Unknown)
)
```

### 数据格式

波形以 HDF5 文件存储，结构如下：

```
waveforms.hdf5
├── X                # 波形数据 (N_samples, N_channels)
├── Y                # P 值标签 (N_samples,)
├── Z                # 清晰度（仅 ditingmotion 需要）
├── metadata         # 附加元数据（可选）
└── ...
```

### 标签编码

- **0**: Up（正极性）
- **1**: Down（负极性）
- **2**: Unknown（未知）

### DataLoader

创建用于训练的 PyTorch DataLoader：

```python
loader = dataset.get_dataloader(
    batch_size=1024,
    num_workers=4,
    shuffle=True,
    pin_memory=True
)
```

## 数据检查

### 基本统计

```python
from seispolarity import WaveformDataset

dataset = WaveformDataset(path="data.hdf5", name="SCSN")

# 获取数据集统计信息
print(f"Total samples: {len(dataset)}")
print(f"Label distribution: {dataset.label_distribution}")
print(f"Waveform shape: {dataset.waveform_shape}")
```

## 多数据集训练

组合多个数据集：

```python
from seispolarity import MultiWaveformDataset

# 创建多个数据集
dataset1 = WaveformDataset(path="scsn.hdf5", name="SCSN")
dataset2 = WaveformDataset(path="txed.hdf5", name="Txed")

# 组合它们
combined = MultiWaveformDataset([dataset1, dataset2])
```

## 平衡采样

对于标签不平衡的数据集，可以使用平衡采样：

```python
from seispolarity import BalancedPolarityGenerator

generator = BalancedPolarityGenerator(
    dataset,
    strategy="polarity_inversion"  # 或 "min_based"
)
loader = generator.get_dataloader(batch_size=256)
```

## 下载地址

数据集可以从以下地址下载：

- **Hugging Face**: `https://huggingface.co/datasets/chuanjun1978/Seismic-AI-Data`
- **ModelScope**: `https://www.modelscope.cn/datasets/chuanjun/Seismic-AI-Data/`（推荐中国用户使用）

更多详情请参见[安装指南](../installation.md)。
