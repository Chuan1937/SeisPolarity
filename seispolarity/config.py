from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

_DEFAULT_CACHE = Path(os.getenv("SEISPOLARITY_CACHE_ROOT", Path.home() / ".seispolarity"))
_DEFAULT_REMOTE = os.getenv(
    "SEISPOLARITY_REMOTE_ROOT",
    "https://example.com/seispolarity/",  # placeholder for future model hosting
)


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
    return settings


def register_model(name: str, factory: Callable) -> None:
    """Register a model factory callable under a short name."""
    settings.model_registry[name.lower()] = factory


def get_model(name: str):
    key = name.lower()
    if key not in settings.model_registry:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(settings.model_registry)}")
    return settings.model_registry[key]()
