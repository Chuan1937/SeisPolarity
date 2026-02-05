# 模型概述

SeisPolarity 提供了一系列用于地震极性分类的深度学习模型。所有模型均使用 PyTorch 实现，并支持 GPU 加速。

## 可用模型

| 模型 | 输入长度 | 类别 |
|-------|---------|---------|
| Ross (SCSN) | 400 | 3 (U/D/N) |
| Eqpolarity | 600 | 2 (U/D) |
| DiTingMotion | 128 | 3 (U/D/N) |
| CFM | 160 | 2 (U/D) |
| RPNet | 400 | 2 (U/D) |
| PolarCAP | 64 | 2 (U/D) |
| APP | 400 | 3 (U/D/N) |

## 加载模型

### 预训练模型

从 Hugging Face 加载预训练模型：

```python
from seispolarity.models import PPNet, RossNet, EqpolarityNet

# Ross 模型
model = RossNet(num_fm_classes=3)

# Eqpolarity 模型
model = EqpolarityNet()

# 通用 PPNet（用于 SCSN）
model = PPNet(num_fm_classes=3)
```

### 加载自定义权重

```python
import torch
from seispolarity.models import PPNet

model = PPNet(num_fm_classes=3)

# 从检查点加载
checkpoint = torch.load("checkpoints/model.pth")
model.load_state_dict(checkpoint["model_state_dict"])

# 或直接加载
model.load_state_dict(torch.load("model_weights.pth"))
```

## 模型 API

SeisPolarity 中的所有模型都暴露统一的 PyTorch 接口：

```python
import torch
from seispolarity.models import PPNet

# 创建模型
model = PPNet(num_fm_classes=3)
model.eval()

# 准备输入
waveforms = torch.randn(10, 1, 400)  # (批量, 通道, 长度)

# 前向传播
with torch.no_grad():
    logits = model(waveforms)
    predictions = logits.argmax(dim=1)  # 获取预测类别
```

## 模型架构

### Ross (SCSN)

Ross 模型是针对 SCSN 数据优化的基于 CNN 的架构。

**输入**：400 个采样点（100 Hz 采样率下 4 秒）

**架构**：
- 带批归一化的卷积层
- 最大池化
- 全连接层
- 用于正则化的 Dropout

```python
from seispolarity.models import PPNet

model = PPNet(num_fm_classes=3)
```

### Eqpolarity

Eqpolarity 是用于极性分类的深度 CNN 模型。

**输入**：600 个采样点（100 Hz 采样率下 6 秒）

```python
from seispolarity.models import EqpolarityNet

model = EqpolarityNet()
```

### DiTingMotion

基于运动的极性分类模型。

**输入**：128 个采样点（100 Hz 采样率下 1.28 秒）

```python
from seispolarity.models import DiTingMotionNet

model = DiTingMotionNet()
```

### CFM

用于极性检测的自定义架构。

**输入**：160 个采样点

```python
from seispolarity.models import CFM

model = CFM()
```

### RPNet

残差极性网络。

**输入**：400 个采样点（100 Hz 采样率下 4 秒）

```python
from seispolarity.models import RPNet

model = RPNet()
```

### PolarCAP

用于极性分类的轻量级模型。

**输入**：64 个采样点（100 Hz 采样率下 0.64 秒）

```python
from seispolarity.models import PolarCAP, PolarCAPLoss

model = PolarCAP()
loss_fn = PolarCAPLoss()
```

### APP

自适应极性预测器。

```python
from seispolarity.models import APP

model = APP()
```

## 推理

### 批量推理

```python
import torch
from seispolarity.models import PPNet
from seispolarity import WaveformDataset

# 加载模型
model = PPNet(num_fm_classes=3)
model.eval()
model.to("cuda")

# 加载数据集
dataset = WaveformDataset(path="data.hdf5", name="SCSN")
loader = dataset.get_dataloader(batch_size=1024)

# 推理
all_predictions = []
with torch.no_grad():
    for waveforms, _ in loader:
        waveforms = waveforms.to("cuda")
        logits = model(waveforms)
        predictions = logits.argmax(dim=1)
        all_predictions.append(predictions.cpu())

predictions = torch.cat(all_predictions)
```

### 单样本推理

```python
import numpy as np
from seispolarity.models import PPNet
import torch

# 加载模型
model = PPNet(num_fm_classes=3)
model.eval()

# 单样本
waveform = np.random.randn(400)  # 单个波形
waveform = torch.FloatTensor(waveform).unsqueeze(0).unsqueeze(0)  # 添加批量和通道维度

# 推理
with torch.no_grad():
    logits = model(waveform)
    prediction = logits.argmax(dim=1).item()

# 解释结果
label_map = {0: "Up", 1: "Down", 2: "Unknown"}
print(f"预测极性: {label_map[prediction]}")
```

## 模型输出

模型输出每个类别的 logits：

```python
# 原始 logits
logits = model(waveforms)  # 形状: (batch_size, num_classes)

# 概率（softmax）
import torch.nn.functional as F
probabilities = F.softmax(logits, dim=1)

# 预测类别
predictions = logits.argmax(dim=1)
```

## 模型下载

预训练模型可在 Hugging Face 上获取：

```python
from huggingface_hub import hf_hub_download
import torch

# 下载模型权重
model_path = hf_hub_download(
    repo_id="HeXingChen/SeisPolarity-Model",
    filename="ross_scsn.pth"
)

# 加载权重
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
```

## 自定义模型

要创建自定义模型：

```python
import torch.nn as nn
from seispolarity.models.base import BasePolarityModel

class CustomModel(BasePolarityModel):
    def __init__(self, num_fm_classes=3):
        super().__init__(num_fm_classes)
        # 在此处定义您的架构
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.fc = nn.Linear(32 * 396, num_fm_classes)

    def forward(self, x):
        # 前向传播
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

详细的 API 文档请参阅 [API 参考](../api/models.md)。
