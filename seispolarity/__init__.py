"""SeisPolarity: polarity picking toolkit."""
from importlib.metadata import PackageNotFoundError, version

try:  # Best effort; falls back during editable development
    __version__ = version("seispolarity")
except PackageNotFoundError:  # pragma: no cover - during local dev without install
    __version__ = "0.0.0"

from .config import configure_cache, settings  # noqa: E402,F401
from .annotations import Pick, PickList, PolarityLabel, PolarityOutput  # noqa: E402,F401

__all__ = [
    "__version__",
    "configure_cache",
    "settings",
    "Pick",
    "PickList",
    "PolarityLabel",
    "PolarityOutput",
]
