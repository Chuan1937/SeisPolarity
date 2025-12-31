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

    """下载文件到数据集缓存。
    - 如果 filename 为 None，则从 URL 派生文件名。
    - 如果 dest_dir 为 None，则使用 settings.cache_datasets。
    - 如果文件存在且与校验和匹配（当提供时），则重用它，除非 overwrite=True。
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

    """将常见的归档格式提取到以归档主干命名的目录中。
    返回提取目录路径。
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
    """方便：将归档下载到 cache/datasets/{name} 并提取它。"""
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
    repo_type: str = "dataset",
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
    repo_type: "dataset" (default), "model", or "space".
    """

    """将 Hugging Face 数据集存储库快照下载到数据集缓存中。
    参数
    ----------
    repo_id：Hugging Face 上的所有者/名称，例如 "chuanjun1978/Seismic-AI-Data"。
    revision：分支/标签/提交；默认 None 表示 main。
    allow_patterns / ignore_patterns：用于过滤要下载的文件的全局模式。
    token：用于私有存储库的可选 HF 令牌。
    local_name：cache_datasets 下的可选文件夹名称；默认为用 "__" 替换斜杠的 repo_id。
    use_symlinks：是否允许 huggingface_hub 创建其缓存的符号链接（节省空间）。设置为 False 以复制文件。
    repo_type：“dataset”（默认）、“model”或“space”。
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
        repo_type=repo_type,
    )

    return target_dir


def fetch_hf_file(
    repo_id: str,
    repo_path: str,
    revision: str | None = None,
    token: str | None = None,
    local_name: str | None = None,
    use_symlinks: bool = True,
    repo_type: str = "dataset",
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
    repo_type: "dataset" (default), "model", or "space".
    """

    """将 Hugging Face 数据集存储库中的单个文件下载到 cache_datasets 中。
    参数
    ----------
    repo_id：所有者/名称，例如 "chuanjun1978/Seismic-AI-Data"。
    repo_path：存储库内文件的路径，例如 "SCEDC/scsn_p_2000_2017_6sec_0.5r_fm_combined.hdf5"。
    revision：分支/标签/提交；默认 main。
    token：用于私有存储库的可选令牌。
    local_name：cache_datasets 下的可选文件夹名称；默认替换斜杠的 repo_id。
    use_symlinks：是否保留 HF 缓存符号链接。
    repo_type：“dataset”（默认）、“model”或“space”。
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
        repo_type=repo_type,
    )

    return Path(file_path)

