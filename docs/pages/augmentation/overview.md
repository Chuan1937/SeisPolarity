# 数据增强概述

SeisPolarity 提供了灵活的数据增强系统，包含多种技术以提高模型鲁棒性并处理不平衡数据集。

## 基本用法

### 使用 GenericGenerator

```python
from seispolarity import WaveformDataset, GenericGenerator
from seispolarity import Demean, Normalize, RandomTimeShift

# Load dataset
dataset = WaveformDataset(path="data.hdf5", name="SCSN", preload=False)

# Create generator with augmentations
generator = GenericGenerator(dataset)
generator.add_augmentations([
    Demean(),
    Normalize(amp_norm_type="peak"),
    RandomTimeShift(max_shift=10)
])

# Get dataloader
loader = generator.get_dataloader(batch_size=256, num_workers=4)
```

### 使用 BalancedPolarityGenerator

对于具有极性标签的不平衡数据集：

```python
from seispolarity import BalancedPolarityGenerator
from seispolarity import Demean, Normalize

generator = BalancedPolarityGenerator(
    dataset,
    strategy="polarity_inversion"  # or "min_based"
)
generator.add_augmentations([
    Demean(),
    Normalize()
])
```

## 可用的增强方法

### 1. Demean

从波形中移除均值。

```python
from seispolarity import Demean

augmentation = Demean()
```

**Parameters**: None

### 2. Normalize

按振幅归一化波形。

```python
from seispolarity import Normalize

# 按峰值振幅归一化
augmentation = Normalize(amp_norm_type="peak")

# 按 RMS 归一化
augmentation = Normalize(amp_norm_type="rms")

# 按最大绝对值归一化
augmentation = Normalize(amp_norm_type="max")
```

**参数**:
- `amp_norm_type`: 归一化类型（"peak"、"rms"、"max"）

### 3. RandomTimeShift

随机在时间上平移波形。

```python
from seispolarity import RandomTimeShift

# Shift by up to 10 samples
augmentation = RandomTimeShift(max_shift=10)
```

**参数**:
- `max_shift`: 平移的最大样本数（默认：10）

### 4. RandomPPickShift

随机平移 P 震相拾取位置。

```python
from seispolarity import RandomPPickShift

# Shift P-pick by up to 5 samples
augmentation = RandomPPickShift(max_shift=5)
```

**参数**:
- `max_shift`: 平移的最大样本数（默认：5）

### 5. BandpassFilter

对波形应用带通滤波器。

```python
from seispolarity import BandpassFilter

# Apply 1-20 Hz bandpass filter
augmentation = BandpassFilter(freqmin=1.0, freqmax=20.0)
```

**参数**:
- `freqmin`: 最小频率（Hz）
- `freqmax`: 最大频率（Hz）
- `corners`: 滤波器角点（默认：4）
- `zerophase`: 是否使用零相位滤波（默认：True）

### 6. Detrend

从波形中移除线性趋势。

```python
from seispolarity import Detrend

augmentation = Detrend()
```

**参数**:
- `type`: 去趋势类型（"linear" 或 "constant"）

### 7. PolarityInversion

随机反转波形的极性。

```python
from seispolarity import PolarityInversion

# 50% probability of inversion
augmentation = PolarityInversion(p=0.5)
```

**参数**:
- `p`: 反转极性的概率（默认：0.5）

### 8. DifferentialFeatures

从波形计算微分特征。

```python
from seispolarity import DifferentialFeatures

augmentation = DifferentialFeatures()
```

**Parameters**: None

### 9. ChangeDtype

Change the data type of waveforms.

```python
from seispolarity import ChangeDtype

# Convert to float32
augmentation = ChangeDtype(dtype="float32")
```

**Parameters**:
- `dtype`: Target data type ("float32", "float64", etc.)

### 10. Stretching

Randomly stretch or compress waveforms.

```python
from seispolarity import Stretching

# Stretch by up to 10%
augmentation = Stretching(max_stretch=0.1)
```

**Parameters**:
- `max_stretch`: Maximum stretch factor (default: 0.1)

### 11. DitingMotionLoss

Custom loss function for DiTing motion-based model.

```python
from seispolarity import DitingMotionLoss

loss_fn = DitingMotionLoss()
```

## Balanced Sampling

### Polarity Inversion Strategy

This strategy creates a balanced dataset by inverting polarities of Up and Down samples.

```python
from seispolarity import BalancedPolarityGenerator

generator = BalancedPolarityGenerator(
    dataset,
    strategy="polarity_inversion"
)
```

**How it works**:
1. For each Up sample, create a Down sample by inversion
2. For each Down sample, create an Up sample by inversion
3. Result: Equal number of Up, Down, and Unknown samples

### Min-Based Strategy

This strategy samples equally from the minority class.

```python
from seispolarity import BalancedPolarityGenerator

generator = BalancedPolarityGenerator(
    dataset,
    strategy="min_based"
)
```

**How it works**:
1. Count samples in each class
2. Determine the minimum count
3. Sample equally from each class up to the minimum count

## Custom Augmentation

Create custom augmentation by subclassing the base class:

```python
from seispolarity.generate.augmentation import BaseAugmentation

class CustomAugmentation(BaseAugmentation):
    def __call__(self, waveform, label):
        # Apply your custom transformation
        augmented_waveform = self._apply_transformation(waveform)
        return augmented_waveform, label

    def _apply_transformation(self, waveform):
        # Your transformation logic here
        return waveform

# Use it
generator = GenericGenerator(dataset)
generator.add_augmentations([
    CustomAugmentation()
])
```

## Augmentation Pipeline

Combine multiple augmentations:

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

## Data Preprocessing

### Standard Preprocessing Pipeline

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

### Training vs Validation

```python
# Training: include data augmentation
train_generator = GenericGenerator(train_dataset)
train_generator.add_augmentations([
    Demean(),
    Normalize(),
    RandomTimeShift(max_shift=10),
    PolarityInversion(p=0.5)
])

# Validation: only basic preprocessing
val_generator = GenericGenerator(val_dataset)
val_generator.add_augmentations([
    Demean(),
    Normalize()
])
```

## Visualization

### Visualize Augmented Samples

```python
import matplotlib.pyplot as plt
import numpy as np

# Get original and augmented samples
original_waveform, label = dataset[0]
augmented_waveform, _ = generator[0]

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(original_waveform[0])
axes[0].set_title(f"Original (Label: {label})")
axes[1].plot(augmented_waveform[0])
axes[1].set_title("Augmented")
plt.tight_layout()
plt.show()
```

## Performance Tips

1. **Order matters**: Apply normalization after other augmentations
2. **Use carefully**: Not all augmentations are appropriate for all tasks
3. **Validate**: Always validate on unaugmented data
4. **Monitor loss**: Watch for signs of over-augmentation
5. **Dataset size**: Use more augmentation for smaller datasets

## Example: Complete Training with Augmentation

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

# Load dataset
dataset = WaveformDataset(path="data.hdf5", name="SCSN")

# Create generator with augmentations
generator = GenericGenerator(dataset)
generator.add_augmentations([
    Detrend(type="linear"),
    Demean(),
    BandpassFilter(freqmin=1.0, freqmax=20.0),
    Normalize(amp_norm_type="peak"),
    RandomTimeShift(max_shift=10),
    PolarityInversion(p=0.3)
])

# Create model and trainer
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

See [API Reference](../api/augmentation.md) for detailed API documentation.
