# 数据增强总览

SeisPolarity 提供灵活的数据增强系统，包含多种技术以提高模型鲁棒性并处理不平衡数据集。

## 基本用法

### 使用 GenericGenerator

```python
from seispolarity import WaveformDataset, GenericGenerator
from seispolarity import Demean, Normalize, RandomTimeShift

# 加载数据集
dataset = WaveformDataset(path="data.hdf5", name="SCSN", preload=False)

# 创建带增强的生成器
generator = GenericGenerator(dataset)
generator.add_augmentations([
    Demean(),
    Normalize(amp_norm_type="peak"),
    RandomTimeShift(max_shift=10)
])

# 获取 dataloader
loader = generator.get_dataloader(batch_size=256, num_workers=4)
```

### 使用 BalancedPolarityGenerator

用于带极性标签的不平衡数据集：

```python
from seispolarity import BalancedPolarityGenerator
from seispolarity import Demean, Normalize

generator = BalancedPolarityGenerator(
    dataset,
    strategy="polarity_inversion"  # 或 "min_based"
)
generator.add_augmentations([
    Demean(),
    Normalize()
])
```

## 可用的增强方法

### 1. Demean（去均值）

去除波形的均值。

```python
from seispolarity import Demean

augmentation = Demean()
```

**参数**：无

### 2. Normalize（归一化）

按振幅对波形进行归一化。

```python
from seispolarity import Normalize

# 按峰值振幅归一化
augmentation = Normalize(amp_norm_type="peak")

# 按 RMS 归一化
augmentation = Normalize(amp_norm_type="rms")

# 按最大绝对值归一化
augmentation = Normalize(amp_norm_type="max")
```

**参数**：
- `amp_norm_type`：归一化类型（"peak"、"rms"、"max"）

### 3. RandomTimeShift（随机时间平移）

对波形进行随机时间平移。

```python
from seispolarity import RandomTimeShift

# 最多平移 10 个采样点
augmentation = RandomTimeShift(max_shift=10)
```

**参数**：
- `max_shift`：最大平移采样点数（默认：10）

### 4. RandomPPickShift（随机 P 波到时平移）

随机移动 P 波相位拾取位置。

```python
from seispolarity import RandomPPickShift

# 最多平移 5 个采样点
augmentation = RandomPPickShift(max_shift=5)
```

**参数**：
- `max_shift`：最大平移采样点数（默认：5）

### 5. BandpassFilter（带通滤波）

对波形应用带通滤波。

```python
from seispolarity import BandpassFilter

# 应用 1-20 Hz 带通滤波
augmentation = BandpassFilter(freqmin=1.0, freqmax=20.0)
```

**参数**：
- `freqmin`：最小频率（Hz）
- `freqmax`：最大频率（Hz）
- `corners`：滤波器阶数（默认：4）
- `zerophase`：是否使用零相位滤波（默认：True）

### 6. Detrend（去趋势）

去除波形的线性趋势。

```python
from seispolarity import Detrend

augmentation = Detrend()
```

**参数**：
- `type`：去趋势类型（"linear" 或 "constant"）

### 7. PolarityInversion（极性反转）

随机反转波形极性。

```python
from seispolarity import PolarityInversion

# 50% 的反转概率
augmentation = PolarityInversion(p=0.5)
```

**参数**：
- `p`：极性反转概率（默认：0.5）

### 8. DifferentialFeatures（差分特征）

从波形计算差分特征。

```python
from seispolarity import DifferentialFeatures

augmentation = DifferentialFeatures()
```

**参数**：无

### 9. ChangeDtype（数据类型转换）

改变波形的数据类型。

```python
from seispolarity import ChangeDtype

# 转换为 float32
augmentation = ChangeDtype(dtype="float32")
```

**参数**：
- `dtype`：目标数据类型（"float32"、"float64" 等）

### 10. Stretching（拉伸）

随机拉伸或压缩波形。

```python
from seispolarity import Stretching

# 最多拉伸 10%
augmentation = Stretching(max_stretch=0.1)
```

**参数**：
- `max_stretch`：最大拉伸因子（默认：0.1）

### 11. DitingMotionLoss

用于 DiTing 基于运动模型的损失函数。

```python
from seispolarity import DitingMotionLoss

loss_fn = DitingMotionLoss()
```

## 平衡采样

### 极性反转策略

该策略构建一个上、下、未知样本等比例分布的平衡数据集。

```python
from seispolarity import BalancedPolarityGenerator

generator = BalancedPolarityGenerator(
    dataset,
    strategy="polarity_inversion"
)
```

**工作原理**：
1. 每个 Up 和 Down 样本生成两个样本（原始 + 极性反转）
2. 补充 Unknown 样本，使其数量与 (Up + Down) 样本总数一致
3. 结果：均衡分布 - Up = 1/3，Down = 1/3，Unknown = 1/3

该策略推荐用于 Instance 和 Txed 数据集。

### 基于最小类策略

该策略从少数类中等量采样。

```python
from seispolarity import BalancedPolarityGenerator

generator = BalancedPolarityGenerator(
    dataset,
    strategy="min_based"
)
```

**工作原理**：
1. 统计每个类别的样本数
2. 确定最小数量
3. 从每个类别中等量采样至最小数量

## 自定义增强

通过继承基类创建自定义增强：

```python
from seispolarity.generate.augmentation import BaseAugmentation

class CustomAugmentation(BaseAugmentation):
    def __call__(self, waveform, label):
        # 应用您的自定义变换
        augmented_waveform = self._apply_transformation(waveform)
        return augmented_waveform, label

    def _apply_transformation(self, waveform):
        # 您的变换逻辑
        return waveform

# 使用它
generator = GenericGenerator(dataset)
generator.add_augmentations([
    CustomAugmentation()
])
```

## 增强流水线

组合多种增强：

```python
from seispolarity import (
    Demean,
    Normalize,
    RandomTimeShift,
    BandpassFilter,
    PolarityInversion
)

generator = GenericGenerator(dataset)
generator.add_augmentations([
    Demean(),
    Normalize(amp_norm_type="peak"),
    BandpassFilter(freqmin=1.0, freqmax=20.0),
    RandomTimeShift(max_shift=10),
    PolarityInversion(p=0.5)
])
```

## 数据预处理

### 标准预处理流水线

```python
from seispolarity import (
    Demean,
    Detrend,
    Normalize,
    BandpassFilter
)

generator = GenericGenerator(dataset)
generator.add_augmentations([
    Detrend(type="linear"),
    Demean(),
    BandpassFilter(freqmin=1.0, freqmax=20.0),
    Normalize(amp_norm_type="peak")
])
```

### 训练与验证

```python
# 训练：包含数据增强
train_generator = GenericGenerator(train_dataset)
train_generator.add_augmentations([
    Demean(),
    Normalize(),
    RandomTimeShift(max_shift=10),
    PolarityInversion(p=0.5)
])

# 验证：仅基础预处理
val_generator = GenericGenerator(val_dataset)
val_generator.add_augmentations([
    Demean(),
    Normalize()
])
```

## 可视化

### 可视化增强后的样本

```python
import matplotlib.pyplot as plt
import numpy as np

# 获取原始和增强后的样本
original_waveform, label = dataset[0]
augmented_waveform, _ = generator[0]

# 绘图
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(original_waveform[0])
axes[0].set_title(f"Original (Label: {label})")
axes[1].plot(augmented_waveform[0])
axes[1].set_title("Augmented")
plt.tight_layout()
plt.show()
```

## 性能建议

1. **顺序很重要**：在其他增强之后应用归一化
2. **谨慎使用**：并非所有增强都适用于所有任务
3. **验证**：始终在未增强的数据上进行验证
4. **监控损失**：留意过度增强的迹象
5. **数据集规模**：小型数据集应使用更多增强

## 示例：带增强的完整训练

```python
from seispolarity import WaveformDataset, GenericGenerator
from seispolarity.models import PPNet
from seispolarity.training import Trainer, TrainingConfig
from seispolarity import (
    Demean,
    Detrend,
    Normalize,
    BandpassFilter,
    RandomTimeShift,
    PolarityInversion
)

# 加载数据集
dataset = WaveformDataset(path="data.hdf5", name="SCSN")

# 创建带增强的生成器
generator = GenericGenerator(dataset)
generator.add_augmentations([
    Detrend(type="linear"),
    Demean(),
    BandpassFilter(freqmin=1.0, freqmax=20.0),
    Normalize(amp_norm_type="peak"),
    RandomTimeShift(max_shift=10),
    PolarityInversion(p=0.3)
])

# 创建模型和训练器
model = PPNet(num_fm_classes=3)
config = TrainingConfig(
    batch_size=256,
    epochs=50,
    learning_rate=1e-4,
    device="cuda"
)

trainer = Trainer(model=model, dataset=generator, config=config)
trainer.train()
```

详细的 API 文档请参见 [API 参考](../api/augmentation.md)。
