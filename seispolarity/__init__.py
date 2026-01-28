"""SeisPolarity: polarity picking toolkit."""
from importlib.metadata import PackageNotFoundError, version

try:  # Best effort; falls back during editable development
    __version__ = version("seispolarity")
except PackageNotFoundError:  # pragma: no cover - during local dev without install
    __version__ = "0.0.0"

from .config import configure_cache, settings, config  # noqa: E402,F401
from .annotations import Pick, PickList, PolarityLabel, PolarityOutput  # noqa: E402,F401
from urllib.parse import urljoin as _urljoin
import logging as _logging

remote_root = "https://huggingface.co/datasets/chuanjun1978/Seismic-AI-Data/resolve/main/"
remote_data_root = remote_root
remote_model_root = "https://huggingface.co/chuanjun1978/SeisPolarity-Model/resolve/main/"

logger = _logging.getLogger("seispolarity")
_ch = _logging.StreamHandler()
_ch.setFormatter(
    _logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
)
logger.addHandler(_ch)

__all__ = [
    "__version__",
    "configure_cache",
    "settings",
    "config",
    "logger",
    "Pick",
    "PickList",
    "PolarityLabel",
    "PolarityOutput",
]

# 延迟导入以避免循环依赖
def __getattr__(name):
    if name == "WaveformDataset":
        from .data.base import WaveformDataset
        return WaveformDataset
    elif name == "MultiWaveformDataset":
        from .data.base import MultiWaveformDataset
        return MultiWaveformDataset
    elif name == "GenericGenerator":
        from .generate.generator import GenericGenerator
        return GenericGenerator
    elif name == "BalancedPolarityGenerator":
        from .generate.generator import BalancedPolarityGenerator
        return BalancedPolarityGenerator
    elif name == "PolarityInversion":
        from .generate.augmentation import PolarityInversion
        return PolarityInversion
    elif name == "Demean":
        from .generate.augmentation import Demean
        return Demean
    elif name == "Normalize":
        from .generate.augmentation import Normalize
        return Normalize
    elif name == "RandomTimeShift":
        from .generate.augmentation import RandomTimeShift
        return RandomTimeShift
    elif name == "BandpassFilter":
        from .generate.augmentation import BandpassFilter
        return BandpassFilter
    elif name == "get_dataset_path":
        from .data.download import get_dataset_path
        return get_dataset_path
    else:
        raise AttributeError(f"module 'seispolarity' has no attribute '{name}'")
