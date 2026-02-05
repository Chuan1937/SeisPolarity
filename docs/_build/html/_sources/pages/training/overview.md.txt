# 训练概述

SeisPolarity 提供统一的训练接口，支持检查点保存、早停机制和灵活配置。

## 基本训练

### 快速开始

```python
from seispolarity.models import PPNet
from seispolarity.training import Trainer, TrainingConfig
from seispolarity import WaveformDataset

# 加载数据集
dataset = WaveformDataset(path="data.hdf5", name="SCSN", preload=True)

# 创建模型
model = PPNet(num_fm_classes=3)

# 配置训练
config = TrainingConfig(
    batch_size=256,
    epochs=50,
    learning_rate=1e-4,
    device="cuda"
)

# 创建训练器
trainer = Trainer(model=model, dataset=dataset, config=config)

# 训练
trainer.train()
```

## TrainingConfig

`TrainingConfig` 类提供所有训练参数：

```python
from seispolarity.training import TrainingConfig

config = TrainingConfig(
    # 数据参数
    batch_size=256,
    num_workers=4,
    pin_memory=True,

    # 优化参数
    epochs=50,
    learning_rate=1e-4,
    optimizer="adam",           # 或 "adamw", "sgd"
    weight_decay=1e-5,

    # 学习率调度器
    lr_scheduler=None,          # 或 "step", "cosine", "reduce_on_plateau"
    lr_scheduler_params=None,

    # 训练行为
    gradient_clip_value=None,   # 梯度裁剪
    early_stopping_patience=10,
    early_stopping_min_delta=1e-4,

    # 检查点
    save_dir="./checkpoints",
    save_every=5,               # 每 N 个 epoch 保存一次
    save_best_only=True,

    # 验证
    validation_split=0.1,       # 用于验证的数据比例
    validation_every=1,         # 每 N 个 epoch 验证一次

    # 设备
    device="cuda",              # 或 "cpu", "mps"

    # 日志记录
    log_every=100,              # 每 N 个批次记录一次
    tensorboard_dir=None,       # TensorBoard 日志目录
)
```

## 优化器

SeisPolarity 支持多种优化器：

```python
from seispolarity.training import TrainingConfig

# Adam（默认）
config = TrainingConfig(optimizer="adam", learning_rate=1e-4)

# AdamW
config = TrainingConfig(optimizer="adamw", learning_rate=1e-4, weight_decay=1e-5)

# SGD
config = TrainingConfig(
    optimizer="sgd",
    learning_rate=1e-3,
    weight_decay=1e-4,
    momentum=0.9
)
```

## 学习率调度器

### Step LR

```python
config = TrainingConfig(
    lr_scheduler="step",
    lr_scheduler_params={
        "step_size": 10,
        "gamma": 0.5
    }
)
```

### Cosine Annealing

```python
config = TrainingConfig(
    lr_scheduler="cosine",
    lr_scheduler_params={
        "T_max": 50
    }
)
```

### Reduce on Plateau

```python
config = TrainingConfig(
    lr_scheduler="reduce_on_plateau",
    lr_scheduler_params={
        "mode": "min",
        "factor": 0.5,
        "patience": 5
    }
)
```

## 自定义训练循环

如需更多对训练过程的控制：

```python
from seispolarity.models import PPNet
from seispolarity import WaveformDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 设置
device = "cuda"
model = PPNet(num_fm_classes=3).to(device)
dataset = WaveformDataset(path="data.hdf5", name="SCSN")
loader = DataLoader(dataset, batch_size=256, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练循环
model.train()
for epoch in range(50):
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/50")

    for waveforms, labels in pbar:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(waveforms)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
```

## 验证

### 手动验证

```python
from torch.utils.data import random_split

# 分割数据集
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for waveforms, labels in val_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy
```

## 检查点

### 保存检查点

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
    'accuracy': accuracy
}, f'checkpoint_epoch_{epoch}.pth')
```

### 加载检查点

```python
checkpoint = torch.load('checkpoint_epoch_50.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

## 早停

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 停止训练
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

# 使用
early_stopping = EarlyStopping(patience=10)

for epoch in range(epochs):
    # ... 训练代码 ...

    val_loss, _ = validate(model, val_loader, criterion, device)
    if early_stopping(val_loss):
        print(f"在 epoch {epoch} 早停")
        break
```

## TensorBoard 日志

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./logs")

# 训练期间记录 loss
for batch_idx, (waveforms, labels) in enumerate(train_loader):
    # ... 训练代码 ...
    writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

# 记录验证指标
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/val', accuracy, epoch)

writer.close()
```

## 多 GPU 训练

```python
import torch.multiprocessing as mp
import torch.distributed as dist

def train_worker(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model = PPNet(num_fm_classes=3)
    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # ... 训练代码 ...

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)
```

## 训练技巧

1. **数据加载**：使用 `num_workers > 0` 和 `pin_memory=True` 加快数据加载
2. **批量大小**：从小批量（256-512）开始，如果内存允许则增加
3. **学习率**：微调时使用较小的学习率（1e-4 到 1e-5）
4. **验证**：始终使用验证集来监控过拟合
5. **检查点**：定期保存检查点以避免丢失进度

## 示例：完整训练脚本

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from seispolarity.models import PPNet
from seispolarity import WaveformDataset

# 配置
DEVICE = "cuda"
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.1

# 加载数据集
dataset = WaveformDataset(path="data.hdf5", name="SCSN", preload=True)

# 分割训练/验证
train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 模型
model = PPNet(num_fm_classes=3).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# TensorBoard
writer = SummaryWriter("./logs")

best_accuracy = 0

for epoch in range(EPOCHS):
    # 训练
    model.train()
    train_loss = 0
    for waveforms, labels in train_loader:
        waveforms = waveforms.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 验证
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for waveforms, labels in val_loader:
            waveforms = waveforms.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # 指标
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    accuracy = 100. * correct / total

    # 日志
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.2f}%")

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)

    # 保存最佳模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model.pth')

writer.close()
```

详细的 API 文档请参阅 [API 参考](../api/training.md)。
