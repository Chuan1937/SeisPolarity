# Welcome to SeisPolarity

:::{figure} _static/seispolarity_logo_title.svg
:align: center
:::

**SeisPolarity** is a comprehensive seismic first-motion polarity picking framework.

SeisPolarity provides a unified API for applying deep learning models to seismic waveforms for polarity classification, supporting multiple datasets and models.

## Features

- **Unified Data Interface**: Support for SCSN, Txed, DiTing, Instance, PNW datasets with automatic downloading
- **Multiple Models**: Ross, Eqpolarity, APP, DiTingMotion, CFM, RPNet, PolarCAP with pretrained weights
- **Flexible Data Loading**: RAM/disk streaming support for datasets of any scale
- **Advanced Inference**: `Predictor` class with automatic pretrained model downloading from Hugging Face/ModelScope
- **Data Augmentation**: Balanced sampling with various augmentation techniques
- **Unified Training**: Checkpoint saving and early stopping mechanisms
- **Cross-platform Support**: Linux, macOS, and Windows

## Quick Start

### Installation

```bash
pip install seispolarity
```

```bash
git clone https://github.com/Chuan1937/SeisPolarity.git
cd SeisPolarity
pip install -e .
```

### Inference

```python
from seispolarity.inference import Predictor
import numpy as np

predictor = Predictor("ROSS_SCSN")  
waveforms = np.random.randn(10, 400)  # (Batch, Length)
predictions = predictor.predict(waveforms)
```

### Data Loading

```python
from seispolarity.data import WaveformDataset

# Use disk streaming for large datasets
dataset = WaveformDataset(path="data.hdf5", name="SCSN", preload=False)
loader = dataset.get_dataloader(batch_size=1024, num_workers=4)

# Use RAM preloading for small datasets
dataset_ram = WaveformDataset(path="data.hdf5", name="SCSN", preload=True)
```

### Automatic Dataset Download

```python
from seispolarity.data import PNW
from pathlib import Path

output_dir = Path('datasets/PNW')
processor = PNW(
    csv_path=str(output_dir / 'PNW.csv'),
    hdf5_path=str(output_dir / 'PNW.hdf5'),
    output_polarity=str(output_dir),
    auto_download=True,  # Automatically download missing data
    use_hf=False,         # Use ModelScope instead of Hugging Face
    component='Z',        # Extract vertical component
    sampling_rate=100     # Target sampling rate (Hz)
)
processor.process()
```

### Training

```python
from seispolarity.models.scsn import SCSN
from seispolarity.training import Trainer, TrainingConfig

model = SCSN(num_fm_classes=3)
config = TrainingConfig(
    batch_size=256,
    epochs=50,
    learning_rate=1e-4,
    device='cuda'  # or 'cpu'
)
trainer = Trainer(model=model, dataset=dataset, config=config)
trainer.train()
```

### Data Augmentation

```python
from seispolarity.generate import GenericGenerator
from seispolarity.generate.augmentation import (
    Demean, Normalize, RandomTimeShift, 
    BandpassFilter, PolarityInversion
)

generator = GenericGenerator(dataset)
generator.add_augmentations([
    Demean(),
    Normalize(amp_norm_type="peak"),
    RandomTimeShift(max_shift=10),
    BandpassFilter(freqmin=1, freqmax=20),
    PolarityInversion(prob=0.5)
])
loader = generator.get_dataloader(batch_size=256)
```

## Supported Datasets

| Dataset | Description | Samples | Auto Download |
|---------|-------------|---------|---------------|
| SCSN | Southern California Seismic Network | 100k+ | Yes |
| Txed | Texas Earthquake Dataset | 50k+ | Yes |
| DiTing | China Earthquake Networks Center | 80k+ | No (request required) |
| Instance | Instance-based Dataset | 30k+ | Yes |
| PNW | Pacific Northwest | 20k+ | Yes |

## Supported Models

| Model | Input Length | Classes |
|-------|-------------|---------|
| Ross (SCSN) | 400 | 3 (U/D/N) |
| Eqpolarity | 600 | 2 (U/D) |
| DiTingMotion | 128 | 3 (U/D/N) |
| CFM | 160 | 2 (U/D) |
| RPNet | 400 | 2 (U/D) |
| PolarCAP | 64 | 2 (U/D) |
| APP | 400 | 3 (U/D/N) |

## Documentation

```{toctree}
:maxdepth: 1
:caption: Documentation Contents

pages/installation.md
pages/datasets/overview.md
pages/models/overview.md
pages/training/overview.md
pages/augmentation/overview.md
pages/api/overview.md
```

## Examples

See complete examples in the `examples/` directory:
- [Dataset API Usage](../examples/datasets_api.ipynb) - Dataset loading and usage examples
- [Prediction API Usage](../examples/predict_api.ipynb) - Model inference examples
- [Model API Usage](../examples/model_api.ipynb) - Model architecture and initialization
- [Training API Usage](../examples/train_api.ipynb) - Training workflow examples

## Citation

The paper will be published subsequently, please stay tuned.

## Related Links

- GitHub: https://github.com/Chuan1937/SeisPolarity
- Documentation: https://seispolarity.readthedocs.io/en/latest/index.html
- Hugging Face: https://huggingface.co/chuanjun1978/SeisPolarity-Model
- ModelScope: https://modelscope.cn/models/chuanjun1978/SeisPolarity-Model

## Contact

For questions and support, please submit an issue on GitHub or contact: [chuanjun1978@gmail.com]
