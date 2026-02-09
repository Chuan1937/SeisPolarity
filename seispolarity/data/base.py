from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    DataLoader = None
    Dataset = object


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
        citation: str = None,
        license: str = None,
        repository_lookup: bool = False,
        augmentations: Optional[List[Callable]] = None,
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
        
        # Citation and license information (from WaveformBenchmarkDataset)
        self._citation = citation
        self._license = license
        self._repository_lookup = repository_lookup

        # --- Cropping parameters ---
        self.window_p0 = kwargs.get('window_p0', None)  # ，None（）
        self.window_len = kwargs.get('window_len', None)  # ，None（）
        
        # p_pick
        self.crop_left = kwargs.get('crop_left', None)  # p_pick
        self.crop_right = kwargs.get('crop_right', None)  # p_pick
        self.p_pick_position = kwargs.get('p_pick_position', None)  # P（SCSN）
        
        # --- Data augmentations ---
        self.augmentations = augmentations or []
        
        # --- New parameters for unified loading ---
        self.preload = preload
        self.allowed_labels = allowed_labels
        self.data_cache = None  # Storage for RAM mode
        self._indices = None    # Logical to physical index mapping
        
        # --- Key mapping configuration ---
        # SCSN：X，Y
        self.data_key = kwargs.get('data_key', 'X')
        self.label_key = kwargs.get('label_key', 'Y')
        self.clarity_key = kwargs.get('clarity_key', 'Z')
        self.pick_key = kwargs.get('pick_key', 'p_pick')
        
        # 
        self._label_map = kwargs.get('label_map', {'U': 0, 'D': 1, 'X': 2})
        self._reverse_label_map = {v: k for k, v in self._label_map.items()}
        
        # Clarity（I:0, E:1, K:2）
        self._clarity_map = kwargs.get('clarity_map', {'I': 0, 'E': 1, 'K': 2})
        self._reverse_clarity_map = {v: k for k, v in self._clarity_map.items()}
        
        # 
        self.metadata_keys = kwargs.get('metadata_keys', [])
        
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
        if not self._path:
            return pd.DataFrame()

        metadatas = []
        chunks, metadata_paths, _ = self._chunks_with_paths()

        if not metadata_paths:
            return pd.DataFrame()

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
        if not fpath.is_file():
            return

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
            if not self._path:
                return [], [], []
            
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
        if path.is_file():
            return [""]
        # Simplified directory scan
        if (path / "waveforms.hdf5").exists():
            return [""]
        return []

    def __str__(self):
        base_str = f"{self._name} - {len(self)} samples"
        if self._citation or self._license:
            info_lines = [base_str]
            if self._citation:
                # Take first line of citation for summary
                first_line = self._citation.strip().split('\n')[0]
                if len(first_line) > 100:
                    first_line = first_line[:97] + "..."
                info_lines.append(f"Citation: {first_line}")
            if self._license:
                info_lines.append(f"License: {self._license}")
            return "\n".join(info_lines)
        return base_str

    @property
    def citation(self):
        """
        The suggested citation for this dataset
        """
        return self._citation
    
    @property
    def license(self):
        """
        The license attached to this dataset
        """
        return self._license
    
    def citation_info(self):
        """
        Print detailed citation information for this dataset.
        """
        info = []
        info.append("=" * 80)
        info.append(f"Dataset: {self._name}")
        info.append("=" * 80)
        
        if self._citation:
            info.append("\nCITATION:")
            info.append("-" * 40)
            info.append(self._citation)
        
        if self._license:
            info.append("\nLICENSE:")
            info.append("-" * 40)
            info.append(self._license)
        
        info.append("=" * 80)
        return "\n".join(info)

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
                if self.label_key and self.label_key in f:
                    metadata['label'] = f[self.label_key][physical_idx]
                if self.clarity_key and self.clarity_key in f:
                    metadata['clarity'] = f[self.clarity_key][physical_idx]
                if self.pick_key and self.pick_key in f:
                    metadata['p_pick'] = f[self.pick_key][physical_idx]
                
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
        
        # Apply data augmentations if any
        if self.augmentations:
            state_dict = {"X": (waveform, metadata)}
            for aug in self.augmentations:
                aug(state_dict)
            # state_dict["X"]，
            x_value = state_dict["X"]
            if isinstance(x_value, tuple) and len(x_value) == 2:
                waveform, metadata = x_value
            else:
                # ，
                # 
                waveform = x_value if isinstance(x_value, np.ndarray) else waveform
                # metadata
        
        # Apply cropping if parameters are set
        waveform_len = waveform.shape[1]
        
        # 
        if self.crop_left is not None and self.crop_right is not None:
            total_len = self.crop_left + self.crop_right
            
            # p_pick：1. metadatap_pick，2. p_pick_position，3. p_pick
            p_pick_value = None
            if 'p_pick' in metadata and metadata['p_pick'] is not None:
                p_pick_value = int(metadata['p_pick'])
            elif self.p_pick_position is not None:
                p_pick_value = self.p_pick_position
            
            if p_pick_value is not None:
                # p_pick
                start_idx = p_pick_value - self.crop_left
                end_idx = p_pick_value + self.crop_right
            else:
                # p_pick，
                start_idx = 0
                end_idx = total_len
            
            # 
            if start_idx < 0 or end_idx > waveform_len:
                # 
                padded_waveform = np.zeros((waveform.shape[0], total_len), dtype=np.float32)
                
                # 
                actual_start = max(0, start_idx)
                actual_end = min(waveform_len, end_idx)
                
                if actual_end > actual_start:
                    # 
                    pad_start = max(0, -start_idx)  # start_idx，
                    pad_len = actual_end - actual_start
                    
                    # 
                    if pad_start + pad_len <= total_len:
                        padded_waveform[:, pad_start:pad_start+pad_len] = waveform[:, actual_start:actual_end]
                
                waveform = padded_waveform
            else:
                # 
                waveform = waveform[:, start_idx:end_idx]
                
        # ：
        elif self.window_len is not None:
            # Determine start index: use p_pick from metadata if available, otherwise use window_p0
            if 'p_pick' in metadata and metadata['p_pick'] is not None:
                # Use p_pick as reference point, with window_p0 as offset
                p_pick_value = int(metadata['p_pick'])
                start_idx = p_pick_value + (self.window_p0 if self.window_p0 is not None else 0)
            elif self.window_p0 is not None:
                # Fallback to fixed window_p0
                start_idx = self.window_p0
            else:
                # No cropping start specified
                return waveform.astype(np.float32), metadata
            
            # Handle negative start index (when cropping before p_pick)
            if start_idx < 0:
                # Pad with zeros at the beginning
                padded_waveform = np.zeros((waveform.shape[0], self.window_len), dtype=np.float32)
                actual_start = max(0, start_idx)
                actual_len = min(waveform_len - actual_start, self.window_len + min(0, start_idx))
                if actual_len > 0:
                    padded_waveform[:, -start_idx:-start_idx+actual_len] = waveform[:, actual_start:actual_start+actual_len]
                waveform = padded_waveform
            else:
                # Calculate cropping range
                end_idx = min(start_idx + self.window_len, waveform_len)
                
                # Ensure we have enough samples
                if end_idx <= waveform_len:
                    waveform = waveform[:, start_idx:end_idx]
                else:
                    # If not enough samples, pad with zeros at the end
                    padded_waveform = np.zeros((waveform.shape[0], self.window_len), dtype=np.float32)
                    actual_len = min(waveform_len - start_idx, self.window_len)
                    padded_waveform[:, :actual_len] = waveform[:, start_idx:start_idx+actual_len]
                    waveform = padded_waveform
        
        return waveform.astype(np.float32), metadata

    def get_sample(self, idx):
        """Compatibility wrapper for get_sample."""
        return self._get_single_item(idx)

    @staticmethod
    def from_seisbench_picks(stream, seisbench_outputs, crop_left=200, crop_right=200):
        """
        Integrate ObsPy waveforms and SeisBench P-wave picks into SeisPolarity format.

        Args:
            stream: ObsPy Stream object containing waveform data.
            seisbench_outputs: SeisBench output object containing P-wave picks.
                               Expected to have a 'picks' attribute.
            crop_left: Number of samples to include before the P-wave pick.
            crop_right: Number of samples to include after the P-wave pick.

        Returns:
            List[np.ndarray]: List of waveform segments around each P-wave pick.
            List[dict]: List of metadata dictionaries for each waveform segment.

        Example:
            >>> from obspy import UTCDateTime
            >>> from seisbench.models import PhaseNet
            >>> from seispolarity.data import WaveformDataset
            >>>
            >>> # Load waveform data
            >>> stream = obspy.read("data.mseed")
            >>>
            >>> # Run SeisBench P-wave picker
            >>> pn_model = PhaseNet.from_pretrained("stead")
            >>> outputs = pn_model.classify(stream)
            >>>
            >>> # Integrate for SeisPolarity
            >>> waveforms, metadata_list = WaveformDataset.from_seisbench_picks(
            ...     stream, outputs, crop_left=200, crop_right=200
            ... )
        """
        waveforms = []
        metadata_list = []

        for i, pick in enumerate(seisbench_outputs.picks):
            pick_time = pick.peak_time

            # Find the trace that contains this pick
            trace = None
            for tr in stream:
                if tr.stats.starttime <= pick_time <= tr.stats.endtime:
                    trace = tr
                    break

            if trace is None:
                logger.warning(f"No trace found for pick at {pick_time}")
                continue

            # Calculate sample indices
            sample_rate = trace.stats.sampling_rate
            pick_sample = int((pick_time - trace.stats.starttime) * sample_rate)
            start_sample = pick_sample - crop_left
            end_sample = pick_sample + crop_right

            # Extract waveform segment
            data = trace.data
            waveform_len = len(data)
            total_len = crop_left + crop_right

            # Handle padding/truncation
            if start_sample < 0 or end_sample > waveform_len:
                padded_waveform = np.zeros(total_len, dtype=np.float32)

                actual_start = max(0, start_sample)
                actual_end = min(waveform_len, end_sample)

                if actual_end > actual_start:
                    pad_start = max(0, -start_sample)
                    pad_len = actual_end - actual_start

                    if pad_start + pad_len <= total_len:
                        padded_waveform[pad_start:pad_start + pad_len] = data[actual_start:actual_end]

                waveform = padded_waveform
            else:
                waveform = data[start_sample:end_sample].astype(np.float32)

            # Ensure correct length
            if len(waveform) != total_len:
                if len(waveform) < total_len:
                    padded = np.zeros(total_len, dtype=np.float32)
                    padded[:len(waveform)] = waveform
                    waveform = padded
                else:
                    waveform = waveform[:total_len]

            # Reshape to (1, N) for single component
            waveform = waveform.reshape(1, -1)

            # Create metadata
            metadata = {
                'p_pick': crop_left,  # P-wave pick position in the cropped waveform
                'sample_rate': sample_rate,
                'pick_time': pick_time,  # Keep as UTCDateTime object for visualization
                'pick_confidence': pick.peak_value if hasattr(pick, 'peak_value') else 1.0,
                'network': trace.stats.network,
                'station': trace.stats.station,
                'channel': trace.stats.channel,
                'location': trace.stats.location,
                'start_time': trace.stats.starttime,  # Keep as UTCDateTime object
                'original_sample': pick_sample,
                'trace': trace  # Include original trace for visualization
            }

            waveforms.append(waveform)
            metadata_list.append(metadata)

            logger.debug(f"Extracted waveform {i}: pick at {pick_time}, shape {waveform.shape}")

        return waveforms, metadata_list

    def _read_waveform(self, path, idx, meta):
        # Handle persistent file handle for streaming performance
        # (This is a simplified version of scsn logic)
        f = None
        
        if self._h5_handle and self._h5_handle.filename == str(path):
            f = self._h5_handle
        else:
            # Optimistic caching of handle
            if self._h5_handle is None:
                self._h5_handle = h5py.File(path, 'r')
                f = self._h5_handle
        
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

    def get_dataloader(self, batch_size=32, num_workers=0, shuffle=False, apply_normalization=True):
        """
        Create a DataLoader for Streaming/Iterative access.

        This handles parallel loading and on-the-fly preprocessing.
        Note: Cropping is now applied at dataset level (in __init__ parameters).
        """
        if DataLoader is None:
            raise ImportError("PyTorch not installed.")
        
        # Parallelism strategy
        # RAM mode -> Main process (0 workers) avoids overhead
        # Disk mode -> Multi-process (N workers) hides IO latency
        workers = 0 if self.preload else max(1, num_workers)
        
        logger.info(f"DataLoader: batch={batch_size}, workers={workers}, shuffle={shuffle}, preload={self.preload}")

        # Optional normalization (cropping is already applied at dataset level)
        try:
            from seispolarity.generate import Normalize
            normalizer = Normalize(amp_norm_axis=-1, amp_norm_type="peak")
            use_normalization = apply_normalization
        except ImportError:
            use_normalization = False
            logger.warning("Normalize module not available, skipping normalization")

        def collate_fn(batch):
            processed_w = []
            labels = []
            metadata_list = []
            
            for waveform, meta in batch:
                if use_normalization:
                    # Apply normalization only
                    state = {"X": (waveform, meta)}
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
            if 'label' in m:
                Y.append(m['label'])
            elif self.label_key in m:
                Y.append(m[self.label_key])
            else:
                Y.append(-1)

        return torch.tensor(X), torch.tensor(Y)

    # Stubs for compatibility
    def _read_data_format(self):
        return {}

    def _unify_sampling_rate(self):
        pass

    def _unify_component_order(self):
        pass

    def _build_trace_name_to_idx_dict(self):
        pass
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
                        # ，
                        numeric_labels = self.convert_labels(Y)
                        mask = np.isin(numeric_labels, self.allowed_labels)
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
                    # ，
                    numeric_labels = self.convert_labels(self._metadata['label'].values)
                    mask = np.isin(numeric_labels, self.allowed_labels)
                    self._indices = np.where(mask)[0]
                elif self.label_key in self._metadata.columns:
                    # ，
                    numeric_labels = self.convert_labels(self._metadata[self.label_key].values)
                    mask = np.isin(numeric_labels, self.allowed_labels)
                    self._indices = np.where(mask)[0]
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
                        # ，None
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
        except KeyboardInterrupt:
            logger.warning("RAM loading interrupted by user (Ctrl+C). Reverting to Disk Mode.")
            self.preload = False
            self.data_cache = None
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
        for w, label in tqdm(loader, desc="Loading Data", unit="chunk"):
            all_w.append(w)
            all_l.append(label)
        return (np.concatenate(all_w), np.concatenate(all_l)) if all_w else (np.array([]), np.array([]))
    
    def convert_labels(self, labels):
        """
        。

        ，。
        （、、）。

        Args:
            labels: ，

        Returns:
            np.ndarray: （int64）
        """
        if labels is None or len(labels) == 0:
            return labels
            
        # numpy
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        
        # ，
        if labels.dtype.kind in ('i', 'u', 'f'):  # 、、
            return labels.astype(np.int64)
        
        # ：
        numeric_labels = np.zeros(len(labels), dtype=np.int64)
        
        for i, label in enumerate(labels):
            # 
            if isinstance(label, (int, float, np.integer, np.floating)):
                numeric_labels[i] = int(label)
            # 
            elif isinstance(label, (str, bytes)):
                # 
                if isinstance(label, bytes):
                    label_str = label.decode()
                else:
                    label_str = str(label)
                
                # 
                numeric_labels[i] = self._label_map.get(label_str.upper(), 2)  # X(2)
            else:
                # ，
                try:
                    numeric_labels[i] = int(label)
                except (ValueError, TypeError):
                    # ，X(2)
                    numeric_labels[i] = 2
                    logger.debug(f": {label} (: {type(label)}), 2")
        
        return numeric_labels
    
    def convert_labels_to_chars(self, labels):
        """
        。

        Args:
            labels:

        Returns:
            list:
        """
        if labels is None or len(labels) == 0:
            return labels
            
        char_labels = []
        for label in labels:
            char_labels.append(self._reverse_label_map.get(int(label), 'X'))
        
        return char_labels
    
    def convert_clarities(self, clarities):
        """
        。

        Args:
            clarities:  ('I', 'E', 'K')

        Returns:
            np.ndarray:  (0, 1, 2)
        """
        if clarities is None or len(clarities) == 0:
            return clarities
            
        # ，
        if isinstance(clarities, (list, np.ndarray)):
            sample = clarities[0] if len(clarities) > 0 else None
        else:
            sample = clarities
            
        if isinstance(sample, (str, bytes)):
            # 
            if isinstance(sample, bytes):
                clarities = np.array([c.decode() if isinstance(c, bytes) else c for c in clarities])
            
            # 
            numeric_clarities = np.zeros(len(clarities), dtype=np.int64)
            for i, clarity in enumerate(clarities):
                numeric_clarities[i] = self._clarity_map.get(clarity, 2)  # K(2)
            
            return numeric_clarities
        elif isinstance(sample, (int, float, np.integer, np.floating)):
            # ，
            return np.array(clarities, dtype=np.int64)
        else:
            try:
                return np.array(clarities, dtype=np.int64)
            except (ValueError, TypeError):
                logger.warning(f": {type(sample)}")
                return clarities
    
    def convert_clarities_to_chars(self, clarities):
        """
        。

        Args:
            clarities:

        Returns:
            list:
        """
        if clarities is None or len(clarities) == 0:
            return clarities
            
        char_clarities = []
        for clarity in clarities:
            char_clarities.append(self._reverse_clarity_map.get(int(clarity), 'K'))
        
        return char_clarities
    
    def add_labels(self, label_type='clarity', label_value='K', selector=None, overwrite=False):
        """
        （claritypolarity）。

        ：
        -
        -
        -
        -
        -

        Args:
            label_type: ，'clarity'  'polarity'，'clarity'
            label_value:
                - clarity: 'I' (Impulsive), 'E' (Emergent), 'K' (Uncertain)
                - polarity: 'U' (Up), 'D' (Down), 'X' (Unknown)
            selector: ，：
                - None:
                - list[int]: ， [0, 1, 2]
                - np.ndarray (bool): ，
                - callable: ，
                - str: ，Pandas
                - dict: ， {'label': 'U'}  {'clarity': 'I'}
            overwrite: ，False

        Returns:
            self: ，

        Raises:
            ValueError:
            TypeError: selector
        """
        if self._metadata.empty:
            logger.warning("，")
            return self
        
        # 
        if label_type not in ['clarity', 'polarity']:
            raise ValueError(f": {label_type}， 'clarity'  'polarity'")
        
        # 
        if label_type == 'clarity':
            if label_value not in ['I', 'E', 'K']:
                raise ValueError(f"clarity: {label_value}， 'I', 'E',  'K'")
            column_name = 'clarity'
            config_key = 'clarity_key'
        else:  # polarity
            if label_value not in ['U', 'D', 'X']:
                raise ValueError(f"polarity: {label_value}， 'U', 'D',  'X'")
            column_name = 'label'
            config_key = 'label_key'
        
        # 
        num_samples = len(self._metadata)
        
        # selector
        indices = self._parse_selector(selector)
        
        # 
        if column_name in self._metadata.columns:
            if not overwrite:
                # selector
                default_value = 'K' if label_type == 'clarity' else 'X'
                current_values = self._metadata[column_name]
                
                # ：selector
                update_indices = []
                for idx in indices:
                    if idx < len(current_values):
                        current_val = current_values.iloc[idx]
                        # 
                        if isinstance(current_val, (int, float, np.integer, np.floating)):
                            # ，
                            if label_type == 'clarity':
                                # clarity'K'2
                                if current_val == 2:
                                    update_indices.append(idx)
                            else:
                                # polarity'X'2
                                if current_val == 2:
                                    update_indices.append(idx)
                        else:
                            # ，
                            try:
                                if label_type == 'clarity':
                                    clarity_map = {'I': 0, 'E': 1, 'K': 2}
                                    char_val = str(current_val)
                                    if isinstance(current_val, bytes):
                                        char_val = current_val.decode()
                                    if clarity_map.get(char_val, 2) == 2:
                                        update_indices.append(idx)
                                else:
                                    polarity_map = {'U': 0, 'D': 1, 'X': 2}
                                    char_val = str(current_val)
                                    if isinstance(current_val, bytes):
                                        char_val = current_val.decode()
                                    if polarity_map.get(char_val, 2) == 2:
                                        update_indices.append(idx)
                            except (ValueError, TypeError, AttributeError):
                                pass
                
                if not update_indices:
                    logger.info(f"{column_name}，")
                    return self
                
                indices = update_indices
                logger.info(f"{column_name}，{len(indices)}")
            else:
                logger.info(f"{column_name}，{len(indices)}")
        else:
            # 
            if label_type == 'clarity':
                default_value = 'K'
            else:  # polarity
                default_value = 'X'
            
            # ，
            if label_type == 'clarity':
                # clarity'K'2
                numeric_default = 2
            else:  # polarity
                # polarity'X'2
                numeric_default = 2
            
            self._metadata[column_name] = np.full(num_samples, numeric_default, dtype=np.int64)
            logger.info(f"{column_name}，{numeric_default}（'{default_value}'）")
        
        # （）
        # 
        if label_type == 'clarity':
            # clarity: 'I'->0, 'E'->1, 'K'->2
            clarity_map = {'I': 0, 'E': 1, 'K': 2}
            numeric_value = clarity_map.get(label_value, 2)
        else:  # polarity
            # polarity: 'U'->0, 'D'->1, 'X'->2
            polarity_map = {'U': 0, 'D': 1, 'X': 2}
            numeric_value = polarity_map.get(label_value, 2)
        
        for idx in indices:
            self._metadata.at[idx, column_name] = numeric_value
        
        # 
        if not hasattr(self, config_key) or getattr(self, config_key) is None:
            setattr(self, config_key, column_name)
        
        # RAM，
        if self.preload and self.data_cache:
            # ，
            if column_name == 'clarity':
                numeric_labels = self.convert_clarities(self._metadata[column_name].values)
            else:  # label
                numeric_labels = self.convert_labels(self._metadata[column_name].values)
            
            self.data_cache[column_name] = numeric_labels
        
        logger.info(f"{len(indices)}{label_type}，'{label_value}'")
        
        return self
    
    def _parse_selector(self, selector):
        """
        ，。

        Args:
            selector: ，

        Returns:
            list[int]:
        """
        num_samples = len(self._metadata)
        
        if selector is None:
            # 
            return list(range(num_samples))
        
        elif isinstance(selector, (list, tuple, np.ndarray)):
            # 
            selector = np.array(selector)
            
            if selector.dtype == bool:
                # 
                if len(selector) != num_samples:
                    raise ValueError(f"({len(selector)})({num_samples})")
                return np.where(selector)[0].tolist()
            else:
                # 
                indices = selector.astype(np.int64)
                if np.any(indices < 0) or np.any(indices >= num_samples):
                    raise ValueError(f":  [0, {num_samples-1}]")
                return indices.tolist()
        
        elif callable(selector):
            # 
            mask = []
            for idx, row in self._metadata.iterrows():
                try:
                    mask.append(bool(selector(row)))
                except Exception as e:
                    raise ValueError(f" ( {idx}): {e}")
            
            mask = np.array(mask, dtype=bool)
            if len(mask) != num_samples:
                raise ValueError(f"({len(mask)})({num_samples})")
            
            return np.where(mask)[0].tolist()
        
        elif isinstance(selector, str):
            # 
            try:
                # Pandas
                mask = self._metadata.eval(selector)
                if not isinstance(mask, (pd.Series, np.ndarray)):
                    raise ValueError("")
                
                if len(mask) != num_samples:
                    raise ValueError(f"({len(mask)})({num_samples})")
                
                return np.where(mask)[0].tolist()
            except Exception as e:
                raise ValueError(f": {e}")
        
        elif isinstance(selector, dict):
            # 
            mask = np.ones(num_samples, dtype=bool)
            
            for column, value in selector.items():
                if column not in self._metadata.columns:
                    raise ValueError(f" '{column}' ")
                
                # 
                self._metadata[column].dtype
                col_values = self._metadata[column]
                
                # 
                if isinstance(value, (int, float, np.integer, np.floating)):
                    # ，
                    column_mask = (col_values == value)
                elif isinstance(value, str):
                    # ，
                    # polarityclarity
                    if column == 'label' or column == 'polarity':
                        # polarity
                        polarity_map = {'U': 0, 'D': 1, 'X': 2}
                        numeric_value = polarity_map.get(value.upper(), 2)
                        column_mask = (col_values == numeric_value)
                    elif column == 'clarity':
                        # clarity
                        clarity_map = {'I': 0, 'E': 1, 'K': 2}
                        numeric_value = clarity_map.get(value.upper(), 2)
                        column_mask = (col_values == numeric_value)
                    else:
                        # ，
                        column_mask = (col_values == value)
                else:
                    # ，
                    column_mask = (col_values == value)
                
                mask = mask & column_mask
            
            return np.where(mask)[0].tolist()
        
        else:
            raise TypeError(f": {type(selector)}，: list, np.ndarray, callable, str, dict")
    
    def add_clarity_labels(self, clarity_value='K', selector=None, overwrite=False):
        """
        clarity（）。

        Args:
            clarity_value: clarity，'K'（）
                - 'I': Impulsive（）
                - 'E': Emergent（）
                - 'K': Uncertain（）
            selector: ，
            overwrite: ，False

        Returns:
            self: ，
        """
        return self.add_labels(
            label_type='clarity', 
            label_value=clarity_value, 
            selector=selector, 
            overwrite=overwrite
        )
    
    def add_polarity_labels(self, polarity_value='X', selector=None, overwrite=False):
        """
        polarity。

        Args:
            polarity_value: polarity，'X'（）
                - 'U': Up（）
                - 'D': Down（）
                - 'X': Unknown（）
            selector: ，
            overwrite: ，False

        Returns:
            self: ，
        """
        return self.add_labels(
            label_type='polarity', 
            label_value=polarity_value, 
            selector=selector, 
            overwrite=overwrite
        )


    def __add__(self, other):
        """
        ，MultiWaveformDataset。

        Args:
            other: WaveformDatasetMultiWaveformDataset

        Returns:
            MultiWaveformDataset:

        Raises:
            TypeError: otherWaveformDatasetMultiWaveformDataset
        """
        if isinstance(other, WaveformDataset):
            return MultiWaveformDataset([self, other])
        elif isinstance(other, MultiWaveformDataset):
            return MultiWaveformDataset([self] + other.datasets)
        else:
            raise TypeError(
                "WaveformDatasetMultiWaveformDatasetWaveformDataset。"
            )
    
    @property
    def metadata(self):
        """Get metadata DataFrame."""
        return self._metadata
    
    @property
    def indices(self):
        """Get logical to physical index mapping."""
        return self._indices
    
    @property
    def label_map(self):
        """"""
        return self._label_map
    
    @property
    def reverse_label_map(self):
        """"""
        return self._reverse_label_map
    
    @property
    def clarity_map(self):
        """"""
        return self._clarity_map
    
    @property
    def reverse_clarity_map(self):
        """"""
        return self._reverse_clarity_map
    
    # --- Data augmentation methods ---
    
    def add_augmentation(self, augmentation: Callable):
        """
        。

        Args:
            augmentation: ，state_dict
        """
        self.augmentations.append(augmentation)
    
    def add_augmentations(self, augmentations: List[Callable]):
        """
        。

        Args:
            augmentations:
        """
        self.augmentations.extend(augmentations)
    
    def clear_augmentations(self):
        """"""
        self.augmentations.clear()
    
    def get_augmentations(self) -> List[Callable]:
        """"""
        return self.augmentations.copy()



# Note: WaveformBenchmarkDataset functionality has been merged into WaveformDataset
# All datasets now support citation and license information directly

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

    def get_dataloader(self, batch_size=32, num_workers=0, shuffle=False, apply_normalization=True):
        """
        MultiWaveformDatasetDataLoader。
        ：MultiWaveformDataset，
        get_dataloader。
        """
        if not self.datasets:
            raise ValueError("MultiWaveformDataset")
        
        # get_dataloader
        # ：
        return self.datasets[0].get_dataloader(
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            apply_normalization=apply_normalization
        )

