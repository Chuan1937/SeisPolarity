from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

_DEFAULT_CACHE = Path(os.getenv("SEISPOLARITY_CACHE_ROOT", Path.home() / ".seispolarity"))
_DEFAULT_REMOTE = os.getenv(
    "SEISPOLARITY_REMOTE_ROOT",
    "https://huggingface.co/datasets/chuanjun1978/Seismic-AI-Data/resolve/main/",
)

# Load or create configuration dictionary
_config_path = _DEFAULT_CACHE / "config.json"
if not _DEFAULT_CACHE.is_dir():
    _DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)

if not _config_path.is_file():
    config = {"dimension_order": "NCW", "component_order": "ZNE"}
    with open(_config_path, "w") as _fconfig:
        json.dump(config, _fconfig, indent=4, sort_keys=True)
else:
    with open(_config_path, "r") as _fconfig:
        config = json.load(_fconfig)


@dataclass
class Settings:
    cache_root: Path = _DEFAULT_CACHE
    remote_root: str = _DEFAULT_REMOTE
    model_registry: dict[str, Callable] = field(default_factory=dict)

    @property
    def cache_waveforms(self) -> Path:
        return self.cache_root / "waveforms"

    @property
    def cache_models(self) -> Path:
        return self.cache_root / "models"

    @property
    def cache_datasets(self) -> Path:
        return self.cache_root / "datasets"


settings = Settings()


def configure_cache(cache_root: str | Path) -> Settings:
    """Update cache root and ensure required subfolders exist."""
    root = Path(cache_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    (root / "waveforms").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "datasets").mkdir(parents=True, exist_ok=True)

    settings.cache_root = root

    # Update config path if cache root changes
    global config
    _config_path = root / "config.json"
    if not _config_path.is_file():
        config = {"dimension_order": "NCW", "component_order": "ZNE"}
        with open(_config_path, "w") as _fconfig:
            json.dump(config, _fconfig, indent=4, sort_keys=True)
    else:
        with open(_config_path, "r") as _fconfig:
            config = json.load(_fconfig)

    return settings


def register_model(name: str, factory: Callable) -> None:
    """Register a model factory callable under a short name."""
    settings.model_registry[name.lower()] = factory


def get_model(name: str):
    key = name.lower()
    if key not in settings.model_registry:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(settings.model_registry)}")
    return settings.model_registry[key]()
