# Installation and Configuration

## Installation via pip

SeisPolarity can be installed through two methods. In either case, it is recommended to install SeisPolarity in a virtual environment, such as using conda.

### Standard Installation

SeisPolarity is directly available via pip. To install locally, run:

```bash
pip install seispolarity
```

### Installation from Source

To install the latest version from source, clone the repository and run:

```bash
git clone https://github.com/Chuan1937/SeisPolarity.git
cd SeisPolarity
pip install -e .
```

## Documentation Dependencies

To build the documentation, install the documentation dependencies:

```bash
pip install seispolarity[docs]
```

## Configuration

### Cache Directory Configuration

SeisPolarity automatically downloads datasets and models. By default, all cached content is stored in the `~/.seispolarity/` directory:

- Datasets: `~/.seispolarity/datasets`
- Models: `~/.seispolarity/models`
- Waveform data: `~/.seispolarity/waveforms`

To configure a custom cache directory:

```python
from seispolarity import configure_cache

configure_cache(cache_root="/path/to/cache")
```

Or set the `SEISPOLARITY_CACHE_ROOT` environment variable:

```bash
export SEISPOLARITY_CACHE_ROOT=/path/to/cache
```

### Remote Repositories

SeisPolarity uses remote repositories to provide datasets and model weights. You can configure the remote repositories:

```python
import seispolarity

# View current remote root directories
print(seispolarity.remote_root)      # Data repository
print(seispolarity.remote_model_root)  # Model repository
```

## GPU Support

SeisPolarity is built on PyTorch, which supports CUDA for GPU acceleration.

### GPU Installation

To install PyTorch with CUDA support, follow the [official PyTorch installation guide](https://pytorch.org/).

### ModelScope Access (China Users)

For users in China, SeisPolarity supports ModelScope for faster downloads:

```python
from seispolarity import get_dataset_path

# Use ModelScope instead of Hugging Face
data_path = get_dataset_path("SCSN", use_hf=False)
```
