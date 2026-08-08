# Training Overview

SeisPolarity provides a unified training interface with checkpoint saving, early stopping mechanisms, and flexible configuration.

## Basic Training

### Quick Start

```python
from seispolarity.models import PPNet
from seispolarity.training import Trainer, TrainingConfig
from seispolarity import WaveformDataset

# Load dataset
dataset = WaveformDataset(path="data.hdf5", name="SCSN", preload=True)

# Create model
model = PPNet(num_fm_classes=3)

# Configure training
config = TrainingConfig(
    batch_size=256,
    epochs=50,
    learning_rate=1e-4,
    device="cuda"
)

# Create trainer
trainer = Trainer(model=model, dataset=dataset, config=config)

# Train
trainer.train()
```

## TrainingConfig

The `TrainingConfig` class provides all training parameters:

```python
from seispolarity.training import TrainingConfig

config = TrainingConfig(
    # Data parameters
    batch_size=256,
    num_workers=4,
    pin_memory=True,

    # Optimization parameters
    epochs=50,
    learning_rate=1e-4,
    optimizer="adam",           # or "adamw", "sgd"
    weight_decay=1e-5,

    # Learning rate scheduler
    lr_scheduler=None,          # or "step", "cosine", "reduce_on_plateau"
    lr_scheduler_params=None,

    # Training behavior
    gradient_clip_value=None,   # Gradient clipping
    early_stopping_patience=10,
    early_stopping_min_delta=1e-4,

    # Checkpoints
    save_dir="./checkpoints",
    save_every=5,               # Save every N epochs
    save_best_only=True,

    # Validation
    validation_split=0.1,       # Fraction of data for validation
    validation_every=1,         # Validate every N epochs

    # Device
    device="cuda",              # or "cpu", "mps"

    # Logging
    log_every=100,              # Log every N batches
    tensorboard_dir=None,       # TensorBoard log directory
)
```

## Optimizers

SeisPolarity supports multiple optimizers:

```python
from seispolarity.training import TrainingConfig

# Adam (default)
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

## Learning Rate Scheduler

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

## Custom Training Loop

For greater control over the training process:

```python
from seispolarity.models import PPNet
from seispolarity import WaveformDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Setup
device = "cuda"
model = PPNet(num_fm_classes=3).to(device)
dataset = WaveformDataset(path="data.hdf5", name="SCSN")
loader = DataLoader(dataset, batch_size=256, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
model.train()
for epoch in range(50):
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/50")

    for waveforms, labels in pbar:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(waveforms)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
```

## Validation

### Manual Validation

```python
from torch.utils.data import random_split

# Split dataset
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Validation function
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

## Checkpoints

### Saving Checkpoints

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
    'accuracy': accuracy
}, f'checkpoint_epoch_{epoch}.pth')
```

### Loading Checkpoints

```python
checkpoint = torch.load('checkpoint_epoch_50.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

## Early Stopping

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
                return True  # Stop training
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

# Usage
early_stopping = EarlyStopping(patience=10)

for epoch in range(epochs):
    # ... training code ...

    val_loss, _ = validate(model, val_loader, criterion, device)
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

## TensorBoard Logging

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./logs")

# Log loss during training
for batch_idx, (waveforms, labels) in enumerate(train_loader):
    # ... training code ...
    writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

# Log validation metrics
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/val', accuracy, epoch)

writer.close()
```

## Multi-GPU Training

```python
import torch.multiprocessing as mp
import torch.distributed as dist

def train_worker(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model = PPNet(num_fm_classes=3)
    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # ... training code ...

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)
```

## Training Tips

1. **Data Loading**: Use `num_workers > 0` and `pin_memory=True` to speed up data loading
2. **Batch Size**: Start with small batches (256-512) and increase if memory allows
3. **Learning Rate**: Use smaller learning rates (1e-4 to 1e-5) for fine-tuning
4. **Validation**: Always use a validation set to monitor overfitting
5. **Checkpoints**: Save checkpoints regularly to avoid losing progress

## Example: Complete Training Script

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from seispolarity.models import PPNet
from seispolarity import WaveformDataset

# Configuration
DEVICE = "cuda"
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.1

# Load dataset
dataset = WaveformDataset(path="data.hdf5", name="SCSN", preload=True)

# Split train/validation
train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = PPNet(num_fm_classes=3).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# TensorBoard
writer = SummaryWriter("./logs")

best_accuracy = 0

for epoch in range(EPOCHS):
    # Training
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

    # Validation
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

    # Metrics
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    accuracy = 100. * correct / total

    # Logging
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.2f}%")

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)

    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model.pth')

writer.close()
```

For detailed API documentation, please refer to [API Reference](../api/training.md).
