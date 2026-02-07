# SeisPolarity

<p align="center">
  <img src="https://raw.githubusercontent.com/Chuan1937/SeisPolarity/main/docs/logo/seispolarity_logo_title.svg" alt="SeisPolarity Logo" width="400">
</p>

A comprehensive framework for seismic first-motion polarity picking with unified APIs.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-BSD%203-Clause-blue.svg)](LICENSE)

## Documentation

Full documentation is available at: https://seispolarity.readthedocs.io/

Build documentation locally:
```bash
cd docs
make html
```

## Installation

### From Source

```bash
git clone https://github.com/Chuan1937/SeisPolarity.git
cd SeisPolarity
pip install -e .
```

## Features

- **Unified Data Interface**: Support for SCSN, Txed, DiTing, Instance, PNW datasets with automatic download
- **Multiple Models**: Ross, Eqpolarity, APP, DiTingMotion, CFM, RPNet, PolarCAP with pre-trained weights
- **Flexible Data Loading**: RAM/Disk streaming for datasets of any size
- **High-level Inference**: `Predictor` class with auto-download from Hugging Face/ModelScope
- **Data Augmentation**: Comprehensive augmentation pipeline with balanced sampling
- **Unified Training**: `Trainer` with checkpointing, early stopping, and logging
- **Cross-platform Support**: Works on Linux, macOS, and Windows

## Quick Start

### Inference

```python
from seispolarity.inference import Predictor
import numpy as np

predictor = Predictor("ross")  # Options: "ross", "eqpolarity", "diting_motion"
waveforms = np.random.randn(10, 400)  # (Batch, Length)
predictions = predictor.predict(waveforms)
# Returns: [0, 1, 2, ...]  (0: Up, 1: Down, 2: Unknown)
```

### Dataset Loading

```python
from seispolarity.data import WaveformDataset

# Disk streaming for large datasets
dataset = WaveformDataset(path="data.hdf5", name="SCSN", preload=False)
loader = dataset.get_dataloader(batch_size=1024, num_workers=4)

# RAM preloading for faster access
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
    auto_download=True,  # Auto-download missing data
    use_hf=False,         # Use ModelScope instead of Hugging Face
    component='Z',        # Extract vertical component
    sampling_rate=100     # Target sampling rate in Hz
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

| Dataset | Description | Samples | Auto-download |
|---------|-------------|---------|---------------|
| SCSN | Southern California Seismic Network | 100k+ | Yes |
| Txed | Texas Earthquake Data | 50k+ | Yes |
| DiTing | Chinese seismic network | 80k+ | Yes |
| Instance | Instance-based dataset | 30k+ | Yes |
| PNW | Pacific Northwest | 20k+ | Yes |

## Supported Models

| Model | Input Length | Classes |
|-------|---------|---------|
| Ross (SCSN) | 400 | 3 (U/D/N) |
| Eqpolarity | 600 | 2 (U/D) |
| DiTingMotion | 128 | 3 (U/D/N) |
| CFM | 160 | 2 (U/D) |
| RPNet | 400 | 2 (U/D) |
| PolarCAP | 64 | 2 (U/D) |
| APP | 400 | 3 (U/D/N) |

## Project Structure

```
SeisPolarity/
├── seispolarity/
│   ├── models/           # Neural network architectures
│   │   ├── base.py       # Base model class
│   │   ├── scsn.py       # SCSN/Ross models
│   │   ├── eqpolarity.py # Eqpolarity model
│   │   ├── diting_motion.py
│   │   ├── app.py
│   │   ├── cfm.py
│   │   ├── rpnet.py
│   │   └── polarCAP.py
│   ├── data/             # Dataset implementations
│   │   ├── base.py       # WaveformDataset base class
│   │   ├── pnw.py
│   │   ├── txed.py
│   │   ├── diting.py
│   │   ├── instance.py
│   │   └── download.py   # Download utilities
│   ├── training/         # Training utilities
│   │   └── trainer.py    # Trainer and TrainingConfig
│   ├── generate/         # Data augmentation
│   │   ├── generator.py  # Generator classes
│   │   └── augmentation.py # Augmentation operations
│   ├── inference.py      # Predictor class
│   ├── config.py         # Configuration management
│   └── annotations.py    # Data structures
├── examples/             # Example notebooks and scripts
├── tests/                # Unit tests
├── docs/                 # Documentation
└── pyproject.toml        # Project configuration
```

## Advanced Usage

### Multi-dataset Training

```python
from seispolarity.data import MultiWaveformDataset

datasets = {
    'SCSN': 'datasets/scsn.hdf5',
    'Txed': 'datasets/txed.hdf5'
}
multi_dataset = MultiWaveformDataset(datasets)
```

### Custom Augmentation

```python
from seispolarity.generate.augmentation import AugmentationBase

class CustomAugmentation(AugmentationBase):
    def __call__(self, waveform, metadata):
        # Your custom augmentation logic
        return waveform, metadata
```

### Model Zoo

Pre-trained models are automatically downloaded from:
- **Hugging Face**: https://huggingface.co/chuanjun1978/SeisPolarity-Model
- **ModelScope**: https://modelscope.cn/models/chuanjun1978/SeisPolarity-Model

## Examples

See the `examples/` directory for complete notebooks:
- [Dataset API Usage](examples/datasets_api.ipynb) - PNW dataset example
- More examples coming soon...

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation
Paper will come soon...

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SeisBench framework for inspiration
- Hugging Face and ModelScope for model hosting
- All dataset providers

## Links

- **GitHub**: https://github.com/Chuan1937/SeisPolarity
- **Hugging Face**: https://huggingface.co/chuanjun1978/SeisPolarity-Model
- **ModelScope**: https://modelscope.cn/models/chuanjun1978/SeisPolarity-Model
- **Documentation**: https://seispolarity.readthedocs.io/

## Contact

For questions and support, please open an issue on GitHub or contact: [chuanjun1978@gmail.com]
