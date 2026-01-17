from __future__ import annotations
import copy
import logging
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, List, Union, Tuple, Literal, Dict
from urllib.parse import urljoin
import os

import h5py
import numpy as np
import pandas as pd
import scipy.signal
from tqdm import tqdm

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    DataLoader = None
    Dataset = object

import seispolarity.util as util
from seispolarity.util import pad_packed_sequence
from seispolarity.config import settings

# Configure logging
logger = logging.getLogger("seispolarity.data")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class LoadingContext:
    """
    The LoadingContext is a dict of pointers to the hdf5 files for the chunks.
    It is an easy way to manage opening and closing of file pointers when required.
    """

    def __init__(self, chunks, waveform_paths):
        self.chunk_dict = {
            chunk: waveform_path for chunk, waveform_path in zip(chunks, waveform_paths)
        }
        self.file_pointers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for file in self.file_pointers.values():
            file.close()
        self.file_pointers = {}

    def __getitem__(self, chunk):
        if chunk not in self.chunk_dict:
            raise KeyError(f'Unknown chunk "{chunk}"')

        if chunk not in self.file_pointers:
            self.file_pointers[chunk] = h5py.File(self.chunk_dict[chunk], "r")
        return self.file_pointers[chunk]

class WaveformDataset(Dataset):
    """
    Unified Base Dataset for SeisPolarity.
    Supports:
    1. Metadata (CSV) + Waveforms (HDF5 Group/Bucket structure)
    2. Flat Data (Single HDF5 with X/Y datasets, virtually indexed)
    """


    def __init__(
        self,
        path=None,
        name=None,
        dimension_order="NCW",
        component_order="ZNE",
        sampling_rate=None,
        cache=None,
        chunks=None,
        missing_components="pad",
        metadata_cache=False,
        split=None,
        preload=False,
        allowed_labels=None,
        **kwargs,
    ):
        if name is None:
            self._name = "Unnamed dataset"
        else:
            self._name = name

        self.cache = cache
        self._path = path if path else None
        
        # Internal handle for flat HDF5 streaming
        self._h5_handle = None 

        self._chunks = chunks
        if self._path:
             self._chunks = sorted(chunks) if chunks is not None else self.available_chunks(self._path)
        else:
             self._chunks = []

        self._missing_components = None
        self._trace_identification_warning_issued = False

        self._dimension_order = None
        self._dimension_mapping = None
        self._component_order = None
        self._component_mapping = None
        self._metadata_lookup = None
        self._chunks_with_paths_cache = None
        self.sampling_rate = sampling_rate

        # --- New parameters for unified loading ---
        self.preload = preload
        self.allowed_labels = allowed_labels
        self.data_cache = None  # Storage for RAM mode
        self._indices = None    # Logical to physical index mapping
        
        # --- Key mapping configuration ---
        # 默认使用SCSN格式的键名：X为数据，Y为标签
        self.data_key = kwargs.get('data_key', 'X')
        self.label_key = kwargs.get('label_key', 'Y')
        self.clarity_key = kwargs.get('clarity_key', 'Z')
        self.pick_key = kwargs.get('pick_key', 'p_pick')
        
        # 其他可选元数据键
        self.metadata_keys = kwargs.get('metadata_keys', ['snr', 'evids', 'split'])
        
        # --- Initialize Data ---
        # 1. Try Loading Metadata 
        # 2. If fails or not applicable, try loading Flat Index 
        self._metadata = self._load_metadata()
        
        # If metadata is empty, we might be in a Flat HDF5 scenario without CSV
        if self._metadata.empty and self._path and Path(self._path).is_file():
             self._scan_flat_hdf5()

        self._data_format = self._read_data_format()

        self._unify_sampling_rate()
        self._unify_component_order()
        self._build_trace_name_to_idx_dict()

        self.dimension_order = dimension_order
        self.component_order = component_order
        self.missing_components = missing_components
        self.metadata_cache = metadata_cache

        self._waveform_cache = defaultdict(dict)
        self.grouping = None
        
        # --- Build index and handle filtering ---
        # Ensure _chunks is set before building index
        if not self._chunks and self._path:
            self._chunks = self.available_chunks(self._path)
        
        self._build_index()
        
        # --- Conditional Preloading ---
        if self.preload:
            self._opt_load_ram()
        else:
            logger.info(f"Operating in Disk Streaming Mode for {self._name}")
        
        if split:
            # Handle split filtering if metadata has 'split' col or if we implement split logic
            if 'split' in self._metadata.columns:
                 self.filter(self._metadata['split'] == split)

    def _load_metadata(self):
        """
        Load metadata from CSV files corresponding to chunks.
        """
        if not self._path: return pd.DataFrame()
        
        metadatas = []
        chunks, metadata_paths, _ = self._chunks_with_paths()
        
        if not metadata_paths: return pd.DataFrame()

        for chunk, metadata_path in zip(chunks, metadata_paths):
            if metadata_path and metadata_path.is_file() and metadata_path.stat().st_size > 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
                    try:
                        tmp = pd.read_csv(metadata_path, low_memory=False)
                        tmp["trace_chunk"] = chunk
                        metadatas.append(tmp)
                    except Exception as e:
                        logger.warning(f"Failed to read metadata {metadata_path}: {e}")
        
        if metadatas:
            df = pd.concat(metadatas)
            df.reset_index(inplace=True, drop=True)
            return df
        return pd.DataFrame()

    def _scan_flat_hdf5(self):
        """
        Populate metadata from HDF5 if no CSV exists (SCSN Style).
        """
        fpath = Path(self._path)
        if not fpath.is_file(): return

        with h5py.File(fpath, 'r') as f:
            if self.data_key in f:
                N = f[self.data_key].shape[0]
                # Create virtual metadata
                meta = {'trace_chunk': [''] * N}
                
                # If label exists, add it
                if self.label_key in f:
                    meta['label'] = f[self.label_key][:]
                
                # If clarity exists, add it
                if self.clarity_key and self.clarity_key in f:
                    meta['clarity'] = f[self.clarity_key][:]
                
                # If pick exists, add it
                if self.pick_key and self.pick_key in f:
                    meta['p_pick'] = f[self.pick_key][:]
                
                # Add other metadata fields
                for k in self.metadata_keys:
                    if k in f:
                        meta[k] = f[k][:]
                
                self._metadata = pd.DataFrame(meta)
                logger.info(f"Initialized Flat Dataset from {fpath.name} with {N} samples.")

    def _chunks_with_paths(self):
        if self._chunks_with_paths_cache is None:
            if not self._path: return [], [], []
            
            p = Path(self._path)
            chunks = self._chunks
            
            # Case 1: Path is HDF5 file (SCSN)
            if p.is_file() and p.suffix == '.hdf5':
                self._chunks_with_paths_cache = ([''], [None], [p])
            
            # Case 2: Path is directory 
            else:
                meta_paths = [p / f"metadata_{c}.csv" if c != "" else p / "metadata.csv" for c in chunks]
                wave_paths = [p / f"waveforms_{c}.hdf5" if c != "" else p / "waveforms.hdf5" for c in chunks]
                self._chunks_with_paths_cache = (chunks, meta_paths, wave_paths)
                
        return self._chunks_with_paths_cache

    def available_chunks(self, path):
        path = Path(path)
        if path.is_file(): return [""]
        # Simplified directory scan
        if (path / "waveforms.hdf5").exists(): return [""]
        return []

    def __len__(self):
        return self._length if hasattr(self, '_length') else len(self._metadata)

    def __getitem__(self, idx):
        """
        Unified Access Point:
        - If RAM Mode (preload=True): Returns from self.data_cache
        - If Disk Mode (preload=False): Reads from HDF5 via persistent handle
        
        Returns (waveform, metadata_dict)
        """
        if isinstance(idx, (slice, np.ndarray, list)):
            # Handle slicing or array indexing
            if isinstance(idx, slice):
                indices = list(range(*idx.indices(len(self))))
            else:
                indices = idx
            
            results = []
            for i in indices:
                results.append(self._get_single_item(i))
            return results
        elif isinstance(idx, str):
            # String access (metadata column)
            return self._metadata[idx]
        else:
            # Single integer index
            return self._get_single_item(idx)
    
    def _get_single_item(self, idx):
        """Get single sample by logical index."""
        if self.preload and self.data_cache:
            # --- RAM Path ---
            waveform = self.data_cache[self.data_key][idx]
            # Fast dict comprehension for metadata
            metadata = {k: v[idx] for k, v in self.data_cache.items() if k != self.data_key}
            
        else:
            # --- Disk Path ---
            # Get physical index
            physical_idx = self._indices[idx] if self._indices is not None else idx
            
            # Get file path
            chunks, _, wave_paths = self._chunks_with_paths()
            if not wave_paths:
                return np.zeros((1, 100), dtype=np.float32), {}
            
            # Lazy Open: Only open file handle on first access
            if self._h5_handle is None:
                self._h5_handle = h5py.File(wave_paths[0], 'r')
            
            f = self._h5_handle
            metadata = {}
            
            if self.data_key in f:
                waveform = f[self.data_key][physical_idx]
                if self.label_key and self.label_key in f:     metadata['label'] = f[self.label_key][physical_idx]
                if self.clarity_key and self.clarity_key in f:   metadata['clarity'] = f[self.clarity_key][physical_idx]
                if self.pick_key and self.pick_key in f:      metadata['p_pick'] = f[self.pick_key][physical_idx]
                
                # Add other metadata fields
                for k in self.metadata_keys:
                    if k in f:
                        metadata[k] = f[k][physical_idx]
            elif 'data' in f:
                waveform = np.zeros((1, 100), dtype=np.float32)
            else:
                waveform = np.zeros((1, 100), dtype=np.float32)
        
        # Ensure (C, N) shape for PyTorch compatibility
        if waveform.ndim == 1:
            waveform = waveform.reshape(1, -1)
        elif waveform.ndim == 2 and waveform.shape[0] > 3:  # (N, C) -> (C, N)
            waveform = waveform.T
        
        return waveform.astype(np.float32), metadata

    def get_sample(self, idx):
        """Compatibility wrapper for get_sample."""
        return self._get_single_item(idx)

    def _read_waveform(self, path, idx, meta):
        # Handle persistent file handle for streaming performance
        # (This is a simplified version of scsn logic)
        f = None
        close_file = True
        
        if self._h5_handle and self._h5_handle.filename == str(path):
            f = self._h5_handle
            close_file = False
        else:
            # Optimistic caching of handle
            if self._h5_handle is None:
                self._h5_handle = h5py.File(path, 'r')
                f = self._h5_handle
                close_file = False
        
        try:
            if self.data_key in f:
                # Flat style
                # Use metadata index or direct idx
                # If metadata was generated from data_key, idx matches
                wav = f[self.data_key][idx]
            elif 'data' in f:
                tname = meta.get('trace_name')
                bucket = tname.split('$')[0] if '$' in tname else tname
                wav = f['data'][bucket][:] # Simplified
            else:
                wav = np.zeros((1, 100))
        except Exception as e:
            logger.error(f"Error reading waveform {idx}: {e}")
            wav = np.zeros((1, 100))
        
        # Ensure shape (C, N)
        if wav.ndim == 1:
            wav = wav.reshape(1, -1)
        
        return wav.astype(np.float32)

    def get_dataloader(self, batch_size=32, num_workers=0, shuffle=False, window_p0=100, window_len=400):
        """
        Create a DataLoader for Streaming/Iterative access.
        
        This handles parallel loading and on-the-fly preprocessing.
        """
        if DataLoader is None:
            raise ImportError("PyTorch not installed.")
        
        # Parallelism strategy
        # RAM mode -> Main process (0 workers) avoids overhead
        # Disk mode -> Multi-process (N workers) hides IO latency
        workers = 0 if self.preload else max(1, num_workers)
        
        logger.info(f"DataLoader: batch={batch_size}, workers={workers}, shuffle={shuffle}, preload={self.preload}")

        # Optional transforms
        try:
            from seispolarity.generate import FixedWindow, Normalize
            window = FixedWindow(p0=window_p0, windowlen=window_len)
            normalizer = Normalize(amp_norm_axis=-1, amp_norm_type="peak")
            use_transforms = True
        except ImportError:
            use_transforms = False
            logger.warning("Transform modules not available, skipping preprocessing")

        def collate_fn(batch):
            processed_w = []
            labels = []
            metadata_list = []
            
            for waveform, meta in batch:
                if use_transforms:
                    # Stateful transform
                    state = {"X": (waveform, meta)}
                    window(state)
                    normalizer(state)
                    w, m = state["X"]
                    processed_w.append(w.squeeze())
                else:
                    processed_w.append(waveform.squeeze())
                    m = meta
                
                labels.append(m.get("label", -1) if m else -1)
                metadata_list.append(m)
            
            if not processed_w: 
                return np.array([]), np.array([]), []
            
            return np.stack(processed_w), np.array(labels), metadata_list

        return DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=workers, 
            collate_fn=collate_fn,
            prefetch_factor=2 if workers > 0 else None
        )

    def _collate_fn(self, batch):
        waves = [b[0] for b in batch]
        metas = [b[1] for b in batch]
        
        X = np.stack(waves)
        
        # Extract labels if present
        Y = []
        for m in metas:
            if 'label' in m: Y.append(m['label'])
            elif self.label_key in m: Y.append(m[self.label_key])
            else: Y.append(-1)
            
        return torch.tensor(X), torch.tensor(Y)

    # Stubs for compatibility
    def _read_data_format(self): return {}
    def _unify_sampling_rate(self): pass
    def _unify_component_order(self): pass
    def _build_trace_name_to_idx_dict(self): pass
    def filter(self, mask):
        self._metadata = self._metadata[mask]
        self._metadata.reset_index(drop=True, inplace=True)
        return self

    def _build_index(self):
        """Build logical to physical index mapping with optional label filtering."""
        if self._metadata.empty:
            # For flat HDF5 without metadata
            chunks, _, wave_paths = self._chunks_with_paths()
            if not wave_paths:
                self._indices = np.array([])
                self._length = 0
                return
            
            # Open first file to get total samples
            with h5py.File(wave_paths[0], 'r') as f:
                total_samples = len(f[self.data_key]) if self.data_key in f else 0
            
            if self.allowed_labels is None:
                self._indices = np.arange(total_samples)
            else:
                # Need to read labels for filtering
                with h5py.File(wave_paths[0], 'r') as f:
                    if self.label_key in f:
                        Y = f[self.label_key][:]
                        mask = np.isin(Y, self.allowed_labels)
                        self._indices = np.where(mask)[0]
                    else:
                        self._indices = np.arange(total_samples)
        else:
            # With metadata
            total_samples = len(self._metadata)
            
            if self.allowed_labels is None:
                self._indices = np.arange(total_samples)
            else:
                # Filter by label column if exists
                if 'label' in self._metadata.columns:
                    mask = self._metadata['label'].isin(self.allowed_labels)
                    self._indices = np.where(mask.values)[0]
                elif self.label_key in self._metadata.columns:
                    mask = self._metadata[self.label_key].isin(self.allowed_labels)
                    self._indices = np.where(mask.values)[0]
                else:
                    self._indices = np.arange(total_samples)
        
        self._length = len(self._indices)
        logger.info(f"Built index with {self._length} samples (filtered from {total_samples})")

    def _opt_load_ram(self):
        """Load filtered dataset into RAM for high-performance access."""
        try:
            import psutil
            psutil_available = True
        except ImportError:
            psutil_available = False
            logger.warning("psutil not installed, skipping memory check")
        
        # Memory Safety Check
        if psutil_available:
            # Estimate required memory (assuming float32)
            sample_size = 1
            chunks, _, wave_paths = self._chunks_with_paths()
            if wave_paths and wave_paths[0]:
                with h5py.File(wave_paths[0], 'r') as f:
                    if self.data_key in f:
                        sample_shape = f[self.data_key].shape[1:]
                        sample_size = np.prod(sample_shape) * 4  # float32 = 4 bytes
            
            required_gb = (self._length * sample_size) / (1024**3) * 1.5  # 1.5x buffer
            avail_gb = psutil.virtual_memory().available / (1024**3)
            
            if avail_gb < required_gb + 1.0:
                logger.warning(f"Low RAM (Need ~{required_gb:.1f}GB, Available {avail_gb:.1f}GB). Falling back to Disk Mode.")
                self.preload = False
                return

        logger.info(f"Loading {self._length} samples into RAM...")
        try:
            chunks, _, wave_paths = self._chunks_with_paths()
            if not wave_paths:
                logger.error("No waveform files found")
                self.preload = False
                return
            
            # For simplicity, assume single file for now
            h5_path = wave_paths[0]
            
            with h5py.File(h5_path, 'r') as f:
                # 1. Initialize Cache
                if self.data_key in f:
                    full_shape = f[self.data_key].shape
                    self.data_cache = {}
                    self.data_cache[self.data_key] = np.empty((self._length,) + full_shape[1:], dtype=f[self.data_key].dtype)
                    
                    # 2. Load Metadata (Fast)
                    logger.info("Loading Metadata...")
                    # Load label, clarity, pick and other metadata
                    meta_keys_to_load = []
                    if self.label_key and self.label_key in f:
                        meta_keys_to_load.append(self.label_key)
                    if self.clarity_key and self.clarity_key in f:
                        meta_keys_to_load.append(self.clarity_key)
                    if self.pick_key and self.pick_key in f:
                        meta_keys_to_load.append(self.pick_key)
                    
                    # Add other metadata keys
                    for k in self.metadata_keys:
                        if k in f:
                            meta_keys_to_load.append(k)
                    
                    for key in meta_keys_to_load:
                        # 创建映射字典，只包含非None的键
                        key_mapping = {}
                        if self.label_key:
                            key_mapping[self.label_key] = 'label'
                        if self.clarity_key:
                            key_mapping[self.clarity_key] = 'clarity'
                        if self.pick_key:
                            key_mapping[self.pick_key] = 'p_pick'
                        
                        target_key = key_mapping.get(key, key)
                        # Read full then slice is fast for small metadata arrays
                        self.data_cache[target_key] = f[key][:][self._indices]

                    # 3. Load Waveforms with Progress Bar
                    total_file_samples = full_shape[0]
                    chunk_size = 10000
                    
                    with tqdm(total=self._length, desc="Loading RAM", unit="samples") as pbar:
                        current_idx_pos = 0
                        
                        for start_file_idx in range(0, total_file_samples, chunk_size):
                            end_file_idx = min(start_file_idx + chunk_size, total_file_samples)
                            
                            # Find range in self._indices that belongs to this file chunk
                            next_idx_pos = np.searchsorted(self._indices, end_file_idx)
                            
                            # If there is overlap
                            if next_idx_pos > current_idx_pos:
                                count = next_idx_pos - current_idx_pos
                                
                                # Read raw chunk from disk
                                chunk_data = f[self.data_key][start_file_idx:end_file_idx]
                                
                                # Map desired indices to chunk-relative coordinates
                                target_indices = self._indices[current_idx_pos:next_idx_pos] - start_file_idx
                                
                                # Fill Cache
                                self.data_cache[self.data_key][current_idx_pos:next_idx_pos] = chunk_data[target_indices]
                                
                                pbar.update(count)
                            
                            current_idx_pos = next_idx_pos
                            if current_idx_pos >= self._length:
                                break
                
                elif 'data' in f:
                    logger.warning("RAM loading for this data format not fully implemented, falling back to disk mode")
                    self.preload = False
                    return

            logger.info("RAM Load Complete.")
        except Exception as e:
            logger.error(f"RAM Load failed: {e}. Reverting to Disk Mode.")
            self.preload = False
            self.data_cache = None

    def close(self):
        """Close file handles and clean up resources."""
        if self._h5_handle:
            self._h5_handle.close()
            self._h5_handle = None
        
        # Clear cache to free memory
        if self.data_cache:
            self.data_cache.clear()
            self.data_cache = None
    
    def __del__(self):
        self.close()
    
    def __getstate__(self):
        """Safety for multiprocessing: don't pickle file handles."""
        state = self.__dict__.copy()
        state['_h5_handle'] = None
        return state
    
    def load_all_data(self, batch_size=1000, **kwargs):
        """Compatibility wrapper to load processed data into arrays."""
        loader = self.get_dataloader(batch_size=batch_size, shuffle=False, **kwargs)
        all_w, all_l = [], []
        for w, l in tqdm(loader, desc="Loading Data", unit="chunk"):
            all_w.append(w)
            all_l.append(l)
        return (np.concatenate(all_w), np.concatenate(all_l)) if all_w else (np.array([]), np.array([]))
    
    @property
    def metadata(self):
        """Get metadata DataFrame."""
        return self._metadata
    
    @property
    def indices(self):
        """Get logical to physical index mapping."""
        return self._indices

class MultiWaveformDataset(Dataset):
    """
    Container for multiple WaveformDatasets to be used as a single dataset.
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self._cumulative_sizes = self.cumsum(datasets)
        self._metadata_cache = None

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __len__(self):
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = np.searchsorted(self._cumulative_sizes, idx, side='right')
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self._cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def metadata(self):
        if self._metadata_cache is None:
            self._metadata_cache = pd.concat([d.metadata for d in self.datasets], ignore_index=True)
        return self._metadata_cache
    
    def train(self):
        return MultiWaveformDataset([d.train() for d in self.datasets])

    def dev(self):
        return MultiWaveformDataset([d.dev() for d in self.datasets])
    
    def test(self):
        return MultiWaveformDataset([d.test() for d in self.datasets])


class WaveformBenchmarkDataset(WaveformDataset):
    """
    Base class for Benchmark datasets. 
    Usually implies auto-download capabilities and specific citation info.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MultiWaveformDataset(Dataset):
    """
    Combines multiple WaveformDatasets.
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self._lens = [len(d) for d in datasets]
        self._cum_lens = np.cumsum(self._lens)
        self._total_len = self._cum_lens[-1] if self._cum_lens.size > 0 else 0

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise IndexError(f"The index ({idx}) is out of bounds.")
            idx += len(self)

        dataset_idx = np.searchsorted(self._cum_lens, idx, side="right")
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self._cum_lens[dataset_idx - 1]

        return self.datasets[dataset_idx][sample_idx]
    
    def get_sample(self, idx):
        # Forward to appropriate dataset
        if idx < 0:
            if -idx > len(self):
                raise IndexError(f"The index ({idx}) is out of bounds.")
            idx += len(self)

        dataset_idx = np.searchsorted(self._cum_lens, idx, side="right")
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self._cum_lens[dataset_idx - 1]

        return self.datasets[dataset_idx].get_sample(sample_idx)

# Compatibility Alias
WaveformBenchmarkDataset = WaveformDataset
