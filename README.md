# SeisPolarity

<p align="center">
  <img src="https://raw.githubusercontent.com/Chuan1937/SeisPolarity/main/docs/logo/seispolarity_logo_title.svg" alt="SeisPolarity Logo">
</p>

A comprehensive framework for seismic first-motion polarity picking with unified APIs.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)](https://pytorch.org/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Documentation

Full documentation is available at: https://seispolarity.readthedocs.io/

Build documentation locally:
```bash
cd docs
make html
```

## Installation

### Install via pip (Recommended)

```bash
pip install seispolarity
```

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

## Supported Datasets

| Dataset | Description | Samples | Auto-download |
|---------|-------------|---------|---------------|
| SCSN | Southern California Seismic Network | 100k+ | Yes |
| Txed | Texas Earthquake Data | 50k+ | Yes |
| DiTing | Chinese seismic network | 80k+ | No(must apply) |
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


### Model Zoo

Pre-trained models are automatically downloaded from:
- **Hugging Face**: https://huggingface.co/chuanjun1978/SeisPolarity-Model
- **ModelScope**: https://modelscope.cn/models/chuanjun1978/SeisPolarity-Model

## Examples

See the `examples/` directory for complete notebooks:
- [Dataset API Usage](examples/datasets_api.ipynb) - Dataset loading and usage examples
- [Predict API Usage](examples/predict_api.ipynb) - Model inference examples
- [Model API Usage](examples/model_api.ipynb) - Model architecture and initialization
- [Train API Usage](examples/train_api.ipynb) - Training pipeline examples

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
