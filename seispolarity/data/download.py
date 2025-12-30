from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Optional, Sequence

import requests

from seispolarity.config import settings

_CHUNK_SIZE = 1024 * 1024  # 1MB


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    url: str,
    filename: Optional[str] = None,
    dest_dir: Path | str | None = None,
    expected_sha256: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Download a file to the dataset cache.

    - If filename is None, derive it from the URL.
    - If dest_dir is None, use settings.cache_datasets.
    - If file exists and matches checksum (when provided), it is reused unless overwrite=True.
    """

    dest_dir = Path(dest_dir) if dest_dir else settings.cache_datasets
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = filename or url.rstrip("/").split("/")[-1]
    target = dest_dir / fname

    if target.exists() and not overwrite:
        if expected_sha256 is None or _sha256(target) == expected_sha256:
            return target

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with target.open("wb") as f:
            for chunk in r.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    if expected_sha256 and _sha256(target) != expected_sha256:
        target.unlink(missing_ok=True)
        raise ValueError("Downloaded file checksum mismatch")

    return target


def maybe_extract(archive_path: Path, target_dir: Path | str | None = None, overwrite: bool = False) -> Path:
    """Extract common archive formats into a directory named after the archive stem.

    Returns the extraction directory path.
    """

    target_dir = Path(target_dir) if target_dir else archive_path.parent / archive_path.stem
    if target_dir.exists():
        if overwrite:
            shutil.rmtree(target_dir)
        else:
            return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive_path), extract_dir=str(target_dir))
    return target_dir


def fetch_and_extract(
    name: str,
    url: str,
    expected_sha256: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Convenience: download an archive to cache/datasets/{name} and extract it."""

    archive = download_file(url, filename=None, dest_dir=settings.cache_datasets, expected_sha256=expected_sha256, overwrite=overwrite)
    return maybe_extract(archive, target_dir=settings.cache_datasets / name, overwrite=overwrite)


def fetch_hf_dataset(
    repo_id: str,
    revision: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    ignore_patterns: Sequence[str] | None = None,
    token: str | None = None,
    local_name: str | None = None,
    use_symlinks: bool = True,
) -> Path:
    """Download a Hugging Face dataset repository snapshot into the datasets cache.

    Parameters
    ----------
    repo_id: owner/name on Hugging Face, e.g. "chuanjun1978/Seismic-AI-Data".
    revision: branch/tag/commit; default None means main.
    allow_patterns / ignore_patterns: glob patterns to filter which files to download.
    token: optional HF token for private repos.
    local_name: optional folder name under cache_datasets; defaults to repo_id with slashes replaced by "__".
    use_symlinks: whether to let huggingface_hub create symlinks to its cache (saves space). Set False to copy files.
    """

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - user environment issue
        raise ImportError("huggingface-hub is required for fetch_hf_dataset; pip install huggingface-hub") from exc

    target_dir = settings.cache_datasets / (local_name or repo_id.replace("/", "__"))
    target_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        local_dir=target_dir,
        local_dir_use_symlinks=use_symlinks,
        token=token,
    )

    return target_dir


def fetch_hf_file(
    repo_id: str,
    repo_path: str,
    revision: str | None = None,
    token: str | None = None,
    local_name: str | None = None,
    use_symlinks: bool = True,
) -> Path:
    """Download a single file from a Hugging Face dataset repo into cache_datasets.

    Parameters
    ----------
    repo_id: owner/name, e.g. "chuanjun1978/Seismic-AI-Data".
    repo_path: path to file inside the repo, e.g. "SCEDC/scsn_p_2000_2017_6sec_0.5r_fm_combined.hdf5".
    revision: branch/tag/commit; default main.
    token: optional token for private repos.
    local_name: optional folder name under cache_datasets; default repo_id with slashes replaced.
    use_symlinks: whether to keep HF cache symlinks.
    """

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError("huggingface-hub is required for fetch_hf_file; pip install huggingface-hub") from exc

    target_dir = settings.cache_datasets / (local_name or repo_id.replace("/", "__"))
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=repo_path,
        revision=revision,
        token=token,
        local_dir=target_dir,
        local_dir_use_symlinks=use_symlinks,
    )

    return Path(file_path)
