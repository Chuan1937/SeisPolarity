# API Reference

This section provides detailed API documentation for SeisPolarity modules.

## Modules

- [Dataset API](dataset.md) - WaveformDataset and data loading utilities
- [Model API](model.md) - Pretrained models and model utilities
- [Training API](training.md) - Training utilities and configuration
- [Augmentation API](augmentation.md) - Data augmentation classes

## Package Structure

```
seispolarity/
├── __init__.py              # Main package entry point
├── dataset/                 # Dataset loading and utilities
│   ├── __init__.py
│   ├── dataset.py          # WaveformDataset class
│   ├── utils.py            # Dataset utilities
│   └── huggingface.py      # Hugging Face integration
├── models/                  # Model architectures
│   ├── __init__.py
│   ├── ross.py             # Ross model
│   ├── eqpolarity.py       # Eqpolarity model
│   ├── ditingmotion.py     # DiTingMotion model
│   ├── cfm.py              # CFM model
│   ├── rpnet.py            # RPNet model
│   ├── polarcap.py         # PolarCAP model
│   └── app.py              # APP model
├── training/                # Training utilities
│   ├── __init__.py
│   ├── trainer.py          # Trainer class
│   └── config.py           # TrainingConfig class
└── generate/                # Data generation and augmentation
    ├── __init__.py
    ├── generator.py        # Generator classes
    └── augmentation.py     # Augmentation classes
```

## Quick Links

### Data Loading

```{eval-rst}
.. autoclass:: seispolarity.dataset.WaveformDataset
   :members:
```

### Models

```{eval-rst}
.. autoclass:: seispolarity.models.PPNet
   :members:
```

```{eval-rst}
.. autoclass:: seispolarity.models.EqpolarityNet
   :members:
```

### Training

```{eval-rst}
.. autoclass:: seispolarity.training.Trainer
   :members:
```

```{eval-rst}
.. autoclass:: seispolarity.training.TrainingConfig
   :members:
```

### Augmentation

```{eval-rst}
.. autoclass:: seispolarity.generate.GenericGenerator
   :members:
```

```{eval-rst}
.. autoclass:: seispolarity.generate.BalancedPolarityGenerator
   :members:
```

## Utility Functions

### get_dataset_path

```{eval-rst}
.. autofunction:: seispolarity.get_dataset_path
```

### configure_cache

```{eval-rst}
.. autofunction:: seispolarity.configure_cache
```

### get_hf_path

```{eval-rst}
.. autofunction:: seispolarity.get_hf_path
```

## Exceptions

```{eval-rst}
.. autoclass:: seispolarity.dataset.DatasetError
   :members:
```

```{eval-rst}
.. autoclass:: seispolarity.models.ModelError
   :members:
```

## Version Information

```python
import seispolarity

print(seispolarity.__version__)
```
