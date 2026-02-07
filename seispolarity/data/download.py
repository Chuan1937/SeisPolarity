from __future__ import annotations

import hashlib
import os
import shutil
import socket
from pathlib import Path
from typing import Optional, Sequence

import h5py
import requests
from seispolarity.config import settings

_CHUNK_SIZE = 1024 * 1024  # 1MB

# Dataset source configuration
HF_REPO = "HeXingChen/Seismic-AI-Data"
# ModelScope repository ID (for dataset download)
MODELSCOPE_DATASET_REPO = "chuanjun/Seismic-AI-Data"
# ModelScope model repository ID (for model download)
MODELSCOPE_MODEL_REPO = "chuanjun/HeXingChen"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_hdf5_integrity(path: Path) -> bool:
    """Check if HDF5 file is complete and readable.

    Parameters
    ----------
    path: Path to the HDF5 file.

    Returns
    -------
    bool: True if file is complete and readable, False otherwise.
    """
    try:
        # Try to open the file in read-only mode
        with h5py.File(path, 'r') as f:
            # Try to access root group, this will raise an exception if file is corrupted
            _ = f.keys()
        return True
    except (OSError, KeyError) as e:
        print(f"  HDF5 file integrity check failed: {e}")
        return False


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

    """Download file to dataset cache.
    - If filename is None, derive it from URL.
    - If dest_dir is None, use settings.cache_datasets.
    - If file exists and matches checksum (when provided), reuse it unless overwrite=True.
    """

    dest_dir = Path(dest_dir) if dest_dir else settings.cache_datasets
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = filename or url.rstrip("/").split("/")[-1]
    target = dest_dir / fname

    if target.exists() and not overwrite:
        if expected_sha256 is None or _sha256(target) == expected_sha256:
            return target

    # Get total file size for progress bar
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))

        # If file size is available, show progress bar
        if total_size > 0:
            from tqdm import tqdm
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {fname}")
            with target.open("wb") as f:
                for chunk in r.iter_content(chunk_size=_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            progress_bar.close()
        else:
            # No file size information, download directly
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
        # local_dir_use_symlinks=use_symlinks, # Deprecated
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
    repo_path: path to file inside the repo, e.g. "SCSN/scsn_p_2000_2017_6sec_0.5r_fm_combined.hdf5".
    revision: branch/tag/commit; default main.
    token: optional token for private repos.
    local_name: optional folder name under cache_datasets; default repo_id with slashes replaced.
    use_symlinks: whether to keep HF cache symlinks.
    repo_type: "dataset" (default), "model", or "space".
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
        # local_dir_use_symlinks=use_symlinks, # Deprecated
        repo_type=repo_type,
    )

    downloaded_file = Path(file_path)

    # Verify the downloaded file is complete (especially for HDF5 files)
    filename = Path(repo_path).name
    if filename.endswith('.hdf5') or filename.endswith('.h5'):
        print("  Verifying downloaded HDF5 file integrity...")
        if not _verify_hdf5_integrity(downloaded_file):
            print("  ERROR: Downloaded HDF5 file is corrupted or incomplete!")
            # Remove the corrupted file
            if downloaded_file.exists():
                downloaded_file.unlink()
            raise RuntimeError(f"Downloaded HDF5 file is corrupted: {downloaded_file}")
        print("  Downloaded HDF5 file is intact.")

    return downloaded_file


def fetch_modelscope_dataset(
    repo_id: str,
    subset_name: str = "default",
    split: str = "train",
    cache_dir: str | Path | None = None,
) -> Path:
    """Download a dataset from ModelScope using MsDataset.load API.

    This is the preferred method for downloading datasets from ModelScope.
    It uses MsDataset.load which handles dataset downloads differently from model downloads.

    Parameters
    ----------
    repo_id: owner/name on ModelScope, e.g. "chuanjun/Seismic-AI-Data".
    subset_name: dataset subset name, default 'default'.
    split: dataset split, e.g. "train", "test". Default is "train".
    cache_dir: directory to cache the downloaded dataset. If None, uses settings.cache_datasets.

    Returns
    -------
    Path to the downloaded dataset directory.
    """

    try:
        from modelscope.msdatasets import MsDataset
    except ImportError as exc:
        raise ImportError("modelscope is required for fetch_modelscope_dataset; pip install modelscope") from exc

    # Setup cache directory
    if cache_dir is None:
        cache_dir = settings.cache_datasets
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a dataset-specific cache directory
    target_dir = cache_dir / f"{repo_id.replace('/', '__')}__{subset_name}__{split}"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from ModelScope using MsDataset.load...")
    print(f"Repo ID: {repo_id}")
    print(f"Subset: {subset_name}")
    print(f"Split: {split}")
    print(f"Cache directory: {target_dir}")

    # Load the dataset using MsDataset.load
    ds = MsDataset.load(
        repo_id,
        subset_name=subset_name,
        split=split,
        cache_dir=str(target_dir),
    )

    print(f"Dataset downloaded successfully!")
    print(f"Dataset info: {ds}")

    return target_dir


def fetch_modelscope_file(
    repo_id: str,
    repo_path: str,
    revision: str | None = None,
    local_name: str | None = None,
    subset_name: str | None = None,
    split: str = "train",
) -> Path:
    """Download a single file from a ModelScope dataset repo into cache_datasets using model_file_download API.

    Note: For datasets, prefer using fetch_modelscope_dataset instead.
    This function is kept for backward compatibility and for downloading individual files.

    Parameters
    ----------
    repo_id: owner/name on ModelScope, e.g. "chuanjun/Seismic-AI-Data".
    repo_path: path to file inside the repo, e.g. "SCSN/scsn_p_2000_2017_6sec_0.5r_fm_combined.hdf5".
    revision: branch/tag/commit; default None means master.
    local_name: optional folder name under cache_datasets; default repo_id with slashes replaced.
    subset_name: dataset subset name, default None means 'default'. (Not used in model_file_download)
    split: dataset split, e.g. "train", "test". Default is "train". (Not used in model_file_download)
    """

    try:
        from modelscope.hub.file_download import model_file_download
    except ImportError as exc:
        raise ImportError("modelscope is required for fetch_modelscope_file; pip install modelscope") from exc

    target_dir = settings.cache_datasets / (local_name or repo_id.replace("/", "__"))
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(repo_path).name
    target_path = target_dir / filename

    # Check if file already exists
    if target_path.exists():
        print(f"Found existing local file: {target_path}")
        # Check if file is complete and readable
        if filename.endswith('.hdf5') or filename.endswith('.h5'):
            print("  Verifying HDF5 file integrity...")
            if not _verify_hdf5_integrity(target_path):
                print("  HDF5 file is corrupted or incomplete. Re-downloading...")
                # Remove the corrupted file
                target_path.unlink()
            else:
                print("  HDF5 file is intact.")
                return target_path
        else:
            # For non-HDF5 files, just return if exists
            return target_path

    print(f"Downloading file from ModelScope using model_file_download...")
    print(f"Repo ID: {repo_id}")
    print(f"File path: {repo_path}")

    # Use ModelScope's model_file_download API to download the file
    file_path = model_file_download(
        model_id=repo_id,
        file_path=repo_path,
        revision=revision,
        local_dir=str(target_dir),
    )

    # Copy the file to target_path if it's not already there
    downloaded_file = Path(file_path)
    if downloaded_file != target_path:
        shutil.copy2(downloaded_file, target_path)
        print(f"Copied to: {target_path}")

    # Verify the downloaded file is complete (especially for HDF5 files)
    if filename.endswith('.hdf5') or filename.endswith('.h5'):
        print("  Verifying downloaded HDF5 file integrity...")
        if not _verify_hdf5_integrity(target_path):
            print("  ERROR: Downloaded HDF5 file is corrupted or incomplete!")
            # Remove the corrupted file
            if target_path.exists():
                target_path.unlink()
            raise RuntimeError(f"Downloaded HDF5 file is corrupted: {target_path}")
        print("  Downloaded HDF5 file is intact.")

    print(f"Downloaded to: {target_path}")
    return target_path


def _check_hf_network_access(timeout: float = 1.0) -> bool:
    """Check if Hugging Face is accessible.

    Parameters
    ----------
    timeout: timeout in seconds for the connection test.

    Returns
    -------
    True if huggingface.co is accessible, False otherwise.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("huggingface.co", 443))
        return True
    except Exception:
        return False
    finally:
        socket.setdefaulttimeout(None)


def fetch_dataset_from_remote(
    dataset_name: str,
    repo_path: str,
    hf_repo_id: str = HF_REPO,
    modelscope_repo_id: str = MODELSCOPE_DATASET_REPO,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    revision: str | None = None,
    use_hf: bool = False,
) -> Path:
    """Download a dataset file from ModelScope or Hugging Face.

    Priority: ModelScope (default) > Hugging Face (if use_hf=True).

    Parameters
    ----------
    dataset_name: name of the dataset (e.g., "SCSN", "TXED", "PNW").
    repo_path: path to file inside the repo (e.g., "SCSN/SCSN_P_2000_2017_6SEC_0.5R_FM_TRAIN.hdf5").
    hf_repo_id: Hugging Face repository ID (default: HeXingChen/Seismic-AI-Data).
    modelscope_repo_id: ModelScope repository ID (default: chuanjun/Seismic-AI-Data).
    cache_dir: directory to cache the downloaded files. If None, uses settings.cache_datasets.
    force_download: if True, force re-download even if file exists.
    use_hf: if True, use Hugging Face instead of ModelScope.

    Returns
    -------
    Path to the downloaded file.

    Examples
    --------
    >>> # Download SCSN training data from ModelScope (default)
    >>> path = fetch_dataset_from_remote(
    ...     dataset_name="SCSN",
    ...     repo_path="SCSN/SCSN_P_2000_2017_6SEC_0.5R_FM_TRAIN.hdf5"
    ... )
    >>> # Use in WaveformDataset
    >>> dataset = WaveformDataset(path=path, name="SCSN_Train", ...)
    """

    # Setup cache directory
    if cache_dir is None:
        cache_dir = settings.cache_datasets
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine target file path
    filename = Path(repo_path).name
    target_path = cache_dir / filename

    # Check if file already exists
    if target_path.exists() and not force_download:
        print(f"Found existing local file: {target_path}", flush=True)
        # Check if file is complete and readable
        if filename.endswith('.hdf5') or filename.endswith('.h5'):
            print("  Verifying HDF5 file integrity...", flush=True)
            if not _verify_hdf5_integrity(target_path):
                print("  HDF5 file is corrupted or incomplete. Re-downloading...", flush=True)
                # Remove the corrupted file
                target_path.unlink()
            else:
                print("  HDF5 file is intact.", flush=True)
                return target_path
        else:
            # For non-HDF5 files, just return if exists
            return target_path

    print(f"Downloading dataset '{dataset_name}' from {'Hugging Face' if use_hf else 'ModelScope'}...", flush=True)
    print(f"Target file: {filename}", flush=True)

    if use_hf:
        # Try Hugging Face if network is accessible
        if _check_hf_network_access():
            try:
                print("Hugging Face network is accessible. Attempting download from Hugging Face...")
                file_path = fetch_hf_file(
                    repo_id=hf_repo_id,
                    repo_path=repo_path,
                    local_name=dataset_name,
                )
                print(f"Dataset downloaded from Hugging Face: {file_path}")
                return file_path
            except Exception as e:
                error_msg = (
                    f"\n{'='*60}\n"
                    f"Failed to download dataset from Hugging Face.\n"
                    f"{'='*60}\n"
                    f"Dataset: {dataset_name}\n"
                    f"File: {filename}\n"
                    f"Hugging Face: {hf_repo_id}/{repo_path}\n\n"
                    f"Error: {e}\n\n"
                    f"Solutions:\n"
                    f"1. Check your network connection (huggingface.co)\n"
                    f"2. Manually download from Hugging Face:\n"
                    f"   https://huggingface.co/datasets/{hf_repo_id}/tree/main\n"
                    f"3. Use ModelScope instead (set use_hf=False)\n"
                    f"4. Specify local path manually in WaveformDataset\n"
                    f"5. Use VPN if Hugging Face is blocked in your region\n"
                    f"{'='*60}"
                )
                raise RuntimeError(error_msg) from e
        else:
            error_msg = (
                f"\n{'='*60}\n"
                f"Cannot access Hugging Face network.\n"
                f"{'='*60}\n"
                f"Dataset: {dataset_name}\n"
                f"File: {filename}\n"
                f"Hugging Face: {hf_repo_id}/{repo_path}\n\n"
                f"Solutions:\n"
                f"1. Check your network connection (huggingface.co)\n"
                f"2. Manually download from Hugging Face:\n"
                f"   https://huggingface.co/datasets/{hf_repo_id}/tree/main\n"
                f"3. Use ModelScope instead (set use_hf=False)\n"
                f"4. Specify local path manually in WaveformDataset\n"
                f"5. Use VPN if Hugging Face is blocked in your region\n"
                f"{'='*60}"
            )
            raise RuntimeError(error_msg)
    else:
        # Try ModelScope (default)
        try:
            from modelscope.hub.file_download import dataset_file_download

            print(f"Attempting download from ModelScope: {modelscope_repo_id}...")
            file_path = dataset_file_download(
                dataset_id=modelscope_repo_id,
                file_path=repo_path,
                cache_dir=str(cache_dir)
            )
            print(f"Dataset downloaded from ModelScope: {file_path}")
            return Path(file_path)
        except ImportError as exc:
            error_msg = (
                f"\n{'='*60}\n"
                f"ModelScope not installed.\n"
                f"{'='*60}\n"
                f"Please install modelscope:\n"
                f"  pip install modelscope\n"
                f"Or use Hugging Face instead (set use_hf=True)\n"
                f"{'='*60}"
            )
            raise RuntimeError(error_msg) from exc
        except Exception as e:
            error_msg = (
                f"\n{'='*60}\n"
                f"Failed to download dataset from ModelScope.\n"
                f"{'='*60}\n"
                f"Dataset: {dataset_name}\n"
                f"File: {filename}\n"
                f"ModelScope: {modelscope_repo_id}/{repo_path}\n\n"
                f"Error: {e}\n\n"
                f"Solutions:\n"
                f"1. Check your network connection (modelscope.cn)\n"
                f"2. Manually download from ModelScope:\n"
                f"   https://www.modelscope.cn/datasets/{modelscope_repo_id}/tree/master\n"
                f"3. Use Hugging Face instead (set use_hf=True)\n"
                f"4. Specify local path manually in WaveformDataset\n"
                f"{'='*60}"
            )
            raise RuntimeError(error_msg) from e


def fetch_dataset_folder(
    dataset_name: str,
    folder_path: str,
    hf_repo_id: str = HF_REPO,
    modelscope_repo_id: str = MODELSCOPE_DATASET_REPO,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    revision: str | None = None,
    use_hf: bool = False,
) -> Path:
    """Download an entire dataset folder from ModelScope or Hugging Face.

    Parameters
    ----------
    dataset_name: name of the dataset (e.g., "SCSN", "TXED", "PNW").
    folder_path: path to folder inside the repo (e.g., "SCSN/", "TXED/").
    hf_repo_id: Hugging Face repository ID (default: HeXingChen/Seismic-AI-Data).
    modelscope_repo_id: ModelScope repository ID (default: chuanjun/Seismic-AI-Data).
    cache_dir: directory to cache the downloaded files. If None, uses settings.cache_datasets.
    force_download: if True, force re-download even if files exist.
    use_hf: if True, use Hugging Face instead of ModelScope.

    Returns
    -------
    Path to the downloaded dataset folder.
    """
    # Setup cache directory
    if cache_dir is None:
        cache_dir = settings.cache_datasets
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Remove trailing slash from folder_path
    folder_name = folder_path.rstrip('/')
    target_dir = cache_dir / folder_name

    # Check if folder already exists and is not empty
    if target_dir.exists() and not force_download:
        files = list(target_dir.iterdir())
        if files:
            print(f"Found existing dataset folder: {target_dir}", flush=True)
            print(f"  Folder contains {len(files)} files/directories.", flush=True)
            return target_dir

    print(f"Downloading dataset folder '{dataset_name}' from {'Hugging Face' if use_hf else 'ModelScope'}...", flush=True)
    print(f"Target folder: {folder_name}", flush=True)

    if use_hf:
        # Try Hugging Face if network is accessible
        if _check_hf_network_access():
            try:
                print("Hugging Face network is accessible. Attempting download from Hugging Face...")
                # Use allow_patterns to download only the specified folder
                target_dir = fetch_hf_dataset(
                    repo_id=hf_repo_id,
                    allow_patterns=[f"{folder_name}/*"],
                    local_name=folder_name,
                )
                print(f"Dataset folder downloaded from Hugging Face: {target_dir}")
                return target_dir
            except Exception as e:
                error_msg = (
                    f"\n{'='*60}\n"
                    f"Failed to download dataset from Hugging Face.\n"
                    f"{'='*60}\n"
                    f"Dataset: {dataset_name}\n"
                    f"Folder: {folder_name}\n"
                    f"Hugging Face: {hf_repo_id}/{folder_name}\n\n"
                    f"Error: {e}\n\n"
                    f"Solutions:\n"
                    f"1. Check your network connection (huggingface.co)\n"
                    f"2. Manually download from Hugging Face:\n"
                    f"   https://huggingface.co/datasets/{hf_repo_id}/tree/main\n"
                    f"3. Use ModelScope instead (set use_hf=False)\n"
                    f"4. Specify local path manually in WaveformDataset\n"
                    f"5. Use VPN if Hugging Face is blocked in your region\n"
                    f"{'='*60}"
                )
                raise RuntimeError(error_msg) from e
        else:
            error_msg = (
                f"\n{'='*60}\n"
                f"Cannot access Hugging Face network.\n"
                f"{'='*60}\n"
                f"Dataset: {dataset_name}\n"
                f"Folder: {folder_name}\n"
                f"Hugging Face: {hf_repo_id}/{folder_name}\n\n"
                f"Solutions:\n"
                f"1. Check your network connection (huggingface.co)\n"
                f"2. Manually download from Hugging Face:\n"
                f"   https://huggingface.co/datasets/{hf_repo_id}/tree/main\n"
                f"3. Use ModelScope instead (set use_hf=False)\n"
                f"4. Specify local path manually in WaveformDataset\n"
                f"5. Use VPN if Hugging Face is blocked in your region\n"
                f"{'='*60}"
            )
            raise RuntimeError(error_msg)
    else:
        # Try ModelScope (default)
        try:
            from modelscope.hub.file_download import dataset_file_download

            print(f"Attempting download from ModelScope: {modelscope_repo_id}...")
            print(f"Note: Downloading files one by one from the folder...", flush=True)
            
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Known files for each dataset folder
            folder_files_map = {
                "SCSN": [
                    "SCSN/SCSN_P_2000_2017_6SEC_0.5R_FM_TRAIN.hdf5",
                    "SCSN/SCSN_P_2000_2017_6SEC_0.5R_FM_TEST.hdf5",
                ],
                "TXED": [
                    "TXED/TXED.hdf5",
                    "TXED/TXED.csv",
                ],
                "PNW": [
                    "PNW/PNW.hdf5",
                    "PNW/PNW.csv",
                ],
                # DiTing is not supported due to national security concerns.
                "DiTing": [
                ],
                "Instance": [
                    "Instance/Instance.hdf5",
                    "Instance/Instance.csv",
                ],
            }
            
            files_to_download = folder_files_map.get(folder_name, [])
            if not files_to_download:
                print(f"Warning: No known files for folder '{folder_name}'", flush=True)
                print(f"Trying to discover files in the folder...", flush=True)
                # For unknown folders, we'll need to use a different approach
                # For now, just return the empty directory
                return target_dir
            
            downloaded_files = []
            for file_path in files_to_download:
                try:
                    print(f"  Downloading: {file_path}...", flush=True)
                    local_file = dataset_file_download(
                        dataset_id=modelscope_repo_id,
                        file_path=file_path,
                        cache_dir=str(target_dir)
                    )
                    # ModelScope downloads to ._____temp/{dataset_id}/{file_path}
                    # We need to move the file to the target location
                    local_path = Path(local_file)
                    target_file = target_dir / Path(file_path).name
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(local_file, target_file)
                    downloaded_files.append(target_file)
                    print(f"    ✓ Downloaded and copied: {target_file.name}", flush=True)
                except Exception as e:
                    print(f"    ✗ Failed to download {file_path}: {e}", flush=True)
                    # Continue with other files
                    continue
            
            if downloaded_files:
                print(f"\nDataset folder downloaded from ModelScope: {target_dir}", flush=True)
                print(f"  Downloaded {len(downloaded_files)} files.", flush=True)
            else:
                print(f"\nWarning: No files were downloaded!", flush=True)
            
            return target_dir
        except ImportError as exc:
            error_msg = (
                f"\n{'='*60}\n"
                f"ModelScope not installed.\n"
                f"{'='*60}\n"
                f"Please install modelscope:\n"
                f"  pip install modelscope\n"
                f"Or use Hugging Face instead (set use_hf=True)\n"
                f"{'='*60}"
            )
            raise RuntimeError(error_msg) from exc
        except Exception as e:
            error_msg = (
                f"\n{'='*60}\n"
                f"Failed to download dataset from ModelScope.\n"
                f"{'='*60}\n"
                f"Dataset: {dataset_name}\n"
                f"Folder: {folder_name}\n"
                f"ModelScope: {modelscope_repo_id}/{folder_name}\n\n"
                f"Error: {e}\n\n"
                f"Solutions:\n"
                f"1. Check your network connection (modelscope.cn)\n"
                f"2. Manually download from ModelScope:\n"
                f"   https://www.modelscope.cn/datasets/{modelscope_repo_id}/tree/master\n"
                f"3. Use Hugging Face instead (set use_hf=True)\n"
                f"4. Specify local path manually in WaveformDataset\n"
                f"{'='*60}"
            )
            raise RuntimeError(error_msg) from e


# Dataset configuration registry for easy access
DATASET_REGISTRY = {
    "SCSN": {
        "reference": "Ross, Z. E., Meier, M. & Hauksson, E. P Wave Arrival Picking and First-Motion Polarity Determination With Deep Learning. JGR Solid Earth 123, 5120-5129 (2018).",
        "train": {
            "repo_path": "SCSN/SCSN_P_2000_2017_6SEC_0.5R_FM_TRAIN.hdf5",
        },
        "test": {
            "repo_path": "SCSN/SCSN_P_2000_2017_6SEC_0.5R_FM_TEST.hdf5",
        },
        "default": {
            "repo_path": "SCSN/",  # Download entire SCSN folder
        },
    },
    "TXED": {
        "reference": "Chen, Y. et al. TXED: The Texas Earthquake Dataset for AI. Seismological Research Letters 95, 2013-2022 (2024).",
        "hdf5": {
            "repo_path": "TXED/TXED.hdf5",
        },
        "csv": {
            "repo_path": "TXED/TXED.csv",
        },
        "default": {
            "repo_path": "TXED/",  # Download entire TXED folder
        },
    },
    "PNW": {
        "reference": "Ni, Y. et al. Curated Pacific Northwest AI-ready seismic dataset. https://eartharxiv.org/repository/view/5049/ (2023).",
        "hdf5": {
            "repo_path": "PNW/PNW.hdf5",
        },
        "csv": {
            "repo_path": "PNW/PNW.csv",
        },
        "default": {
            "repo_path": "PNW/",  # Download entire PNW folder
        },
    },
}


def get_dataset_path(
    dataset_name: str,
    subset: str = "train",
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    use_hf: bool = False,
) -> Path:
    """Get the path to a dataset file, downloading it if necessary.

    This is a convenience wrapper around fetch_dataset_from_remote that uses
    the predefined DATASET_REGISTRY to look up dataset paths.

    Parameters
    ----------
    dataset_name: name of the dataset (e.g., "SCSN", "TXED", "PNW").
    subset: which subset to download. For SCSN: "train" or "test".
             For TXED/PNW: "hdf5" or "csv" or "default".
             Defaults to "train" for SCSN, "default" for others.
    cache_dir: directory to cache the downloaded files. If None, uses settings.cache_datasets.
    force_download: if True, force re-download even if file exists.
    use_hf: if True, use Hugging Face instead of ModelScope.

    Returns
    -------
    Path to the dataset file.

    Examples
    --------
    >>> # Get SCSN training dataset path from ModelScope (default)
    >>> path = get_dataset_path("SCSN", subset="train")
    >>> dataset = WaveformDataset(path=path, name="SCSN_Train", ...)
    >>> # Get SCSN training dataset path from Hugging Face
    >>> path = get_dataset_path("SCSN", subset="train", use_hf=True)
    >>> dataset = WaveformDataset(path=path, name="SCSN_Train", ...)
    >>> # Get TXED HDF5 dataset path
    >>> path = get_dataset_path("TXED", subset="hdf5")
    """

    dataset_name = dataset_name.upper()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY.keys())}")

    if subset not in DATASET_REGISTRY[dataset_name]:
        raise ValueError(f"Unknown subset '{subset}' for dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY[dataset_name].keys())}")

    subset_config = DATASET_REGISTRY[dataset_name][subset]
    repo_path = subset_config["repo_path"]

    # If repo_path ends with '/', download entire folder
    if repo_path.endswith('/'):
        return fetch_dataset_folder(
            dataset_name=dataset_name,
            folder_path=repo_path,
            cache_dir=cache_dir,
            force_download=force_download,
            use_hf=use_hf,
        )
    else:
        return fetch_dataset_from_remote(
            dataset_name=dataset_name,
            repo_path=repo_path,
            cache_dir=cache_dir,
            force_download=force_download,
            use_hf=use_hf,
        )

