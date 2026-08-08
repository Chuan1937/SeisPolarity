# Models Overview

SeisPolarity provides a series of deep learning models for seismic polarity classification. All models are implemented using PyTorch and support GPU acceleration.

## Available Models

| Model | Input Length | Classes |
|-------|-------------|---------|
| Ross (SCSN) | 400 | 3 (U/D/N) |
| Eqpolarity | 600 | 2 (U/D) |
| DiTingMotion | 128 | 3 (U/D/N) |
| CFM | 160 | 2 (U/D) |
| RPNet | 400 | 2 (U/D) |
| PolarCAP | 64 | 2 (U/D) |
| APP | 400 | 3 (U/D/N) |

## Loading Models

### Pre-trained Models

Load pre-trained models from Hugging Face:

```python
from seispolarity.models import PPNet, RossNet, EqpolarityNet

# Ross model
model = RossNet(num_fm_classes=3)

# Eqpolarity model
model = EqpolarityNet()

# Generic PPNet (for SCSN)
model = PPNet(num_fm_classes=3)
```

### Loading Custom Weights

```python
import torch
from seispolarity.models import PPNet

model = PPNet(num_fm_classes=3)

# Load from checkpoint
checkpoint = torch.load("checkpoints/model.pth")
model.load_state_dict(checkpoint["model_state_dict"])

# Or load directly
model.load_state_dict(torch.load("model_weights.pth"))
```

## Model API

All models in SeisPolarity expose a unified PyTorch interface:

```python
import torch
from seispolarity.models import PPNet

# Create model
model = PPNet(num_fm_classes=3)
model.eval()

# Prepare input
waveforms = torch.randn(10, 1, 400)  # (batch, channels, length)

# Forward pass
with torch.no_grad():
    logits = model(waveforms)
    predictions = logits.argmax(dim=1)  # Get predicted classes
```

## Model Architectures

### Ross (SCSN)

```{image} ross.png
:align: center
:width: 80%
```

The Ross model is a CNN-based architecture optimized for SCSN data.

**Input**: 400 samples (4 seconds at 100 Hz sampling rate)

**Architecture**:
- Convolutional layers with batch normalization
- Max pooling
- Fully connected layers
- Dropout for regularization

```python
from seispolarity.models import PPNet

model = PPNet(num_fm_classes=3)
```

### Eqpolarity

```{image} eqpolarity.png
:align: center
:width: 80%
```

Eqpolarity is a deep CNN model for polarity classification.

**Input**: 600 samples (6 seconds at 100 Hz sampling rate)

```python
from seispolarity.models import EqpolarityNet

model = EqpolarityNet()
```

### DiTingMotion

```{image} ditingmotion.png
:align: center
:width: 60%
```

Motion-based polarity classification model.

**Input**: 128 samples (1.28 seconds at 100 Hz sampling rate)

```python
from seispolarity.models import DiTingMotionNet

model = DiTingMotionNet()
```

### CFM

```{image} cfm.png
:align: center
:width: 80%
```

Custom architecture for polarity detection.

**Input**: 160 samples

```python
from seispolarity.models import CFM

model = CFM()
```

### RPNet

```{image} rpnet.png
:align: center
:width: 80%
```

Residual Polarity Network.

**Input**: 400 samples (4 seconds at 100 Hz sampling rate)

```python
from seispolarity.models import RPNet

model = RPNet()
```

### PolarCAP

```{image} polarcap.png
:align: center
:width: 80%
```

Lightweight model for polarity classification.

**Input**: 64 samples (0.64 seconds at 100 Hz sampling rate)

```python
from seispolarity.models import PolarCAP, PolarCAPLoss

model = PolarCAP()
loss_fn = PolarCAPLoss()
```

### APP

```{image} app.png
:align: center
:width: 80%
```

Adaptive Polarity Predictor.

```python
from seispolarity.models import APP

model = APP()
```

## Inference

### Batch Inference

```python
import torch
from seispolarity.models import PPNet
from seispolarity import WaveformDataset

# Load model
model = PPNet(num_fm_classes=3)
model.eval()
model.to("cuda")

# Load dataset
dataset = WaveformDataset(path="data.hdf5", name="SCSN")
loader = dataset.get_dataloader(batch_size=1024)

# Inference
all_predictions = []
with torch.no_grad():
    for waveforms, _ in loader:
        waveforms = waveforms.to("cuda")
        logits = model(waveforms)
        predictions = logits.argmax(dim=1)
        all_predictions.append(predictions.cpu())

predictions = torch.cat(all_predictions)
```

### Single Sample Inference

```python
import numpy as np
from seispolarity.models import PPNet
import torch

# Load model
model = PPNet(num_fm_classes=3)
model.eval()

# Single sample
waveform = np.random.randn(400)  # Single waveform
waveform = torch.FloatTensor(waveform).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

# Inference
with torch.no_grad():
    logits = model(waveform)
    prediction = logits.argmax(dim=1).item()

# Interpret result
label_map = {0: "Up", 1: "Down", 2: "Unknown"}
print(f"Predicted polarity: {label_map[prediction]}")
```

## Model Output

Models output logits for each class:

```python
# Raw logits
logits = model(waveforms)  # Shape: (batch_size, num_classes)

# Probabilities (softmax)
import torch.nn.functional as F
probabilities = F.softmax(logits, dim=1)

# Predicted classes
predictions = logits.argmax(dim=1)
```

## Model Download

Pre-trained models are available on Hugging Face:

```python
from huggingface_hub import hf_hub_download
import torch

# Download model weights
model_path = hf_hub_download(
    repo_id="HeXingChen/SeisPolarity-Model",
    filename="ross_scsn.pth"
)

# Load weights
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
```

## Custom Models

To create a custom model:

```python
import torch.nn as nn
from seispolarity.models.base import BasePolarityModel

class CustomModel(BasePolarityModel):
    def __init__(self, num_fm_classes=3):
        super().__init__(num_fm_classes)
        # Define your architecture here
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.fc = nn.Linear(32 * 396, num_fm_classes)

    def forward(self, x):
        # Forward pass
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

For detailed API documentation, see the [API Reference](../api/models.md).
