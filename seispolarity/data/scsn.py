import h5py
import numpy as np
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
from typing import Tuple, Optional, List

try:
    import psutil
except ImportError:
    psutil = None

from seispolarity.data.download import fetch_hf_file
from seispolarity.config import settings
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class SCSNDataset(Dataset):
    """
    SCSN Dataset handler tailored for Seismic Polarity tasks.
    
    Supports two primary modes of operation:
    1. **Memory Mode (preload=True)**: Loads the entire dataset into RAM. Fastest for training if RAM permits.
    2. **Disk Streaming Mode (preload=False)**: Reads data on-the-fly from HDF5. Low RAM usage, parallelizable via DataLoader.
    """

    # =========================================================================
    # 1. Initialization & Global Setup
    # =========================================================================

    def __init__(self, h5_path: Optional[str] = None, 
                 preload: bool = False, split: str = "train", allowed_labels: Optional[List[int]] = None):
        """
        Initialize the dataset config, resolve file paths, and build index.

        Args:
            h5_path (str, optional): Path to HDF5 file. Auto-downloads if None.
            preload (bool): Whether to load everything into RAM.
            split (str): 'train' or 'test' (used for auto-download).
            allowed_labels (list): Filter samples by label (e.g. [0, 1]).
        """
        # --- A. Resolve File Path ---
        if h5_path is None:
            repo_id = "chuanjun1978/Seismic-AI-Data"
            if split == "train":
                filename = "scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5"
            elif split == "test":
                filename = "scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5"
            else:
                raise ValueError(f"Unknown split '{split}'. Use 'train' or 'test'.")
            
            logger.info(f"Fetching {filename} from HF...")
            h5_path = fetch_hf_file(repo_id=repo_id, repo_path=f"SCEDC/{filename}")

        self.h5_path = str(h5_path)
        logger.info(f"Dataset Path: {self.h5_path}")

        # --- B. Global Parameters ---
        self.preload = preload
        self.allowed_labels = allowed_labels
        
        # Internal state
        self.data_cache = None       # Storage for RAM mode
        self._h5_handle = None       # Storage for Disk mode (process-local)
        self.indices = None          # Map: Logical Index -> Physical HDF5 Index
        self.length = 0

        # --- C. Indexing & Validation ---
        self._build_index()
        
        # --- D. Conditional Preloading ---
        if self.preload:
            self._opt_load_ram()
        else:
            logger.info("Operating in Disk Streaming Mode.")

    def _build_index(self):
        """Scan HDF5 file to build the logical-to-physical index map."""
        with h5py.File(self.h5_path, 'r') as f:
            total_samples = len(f['X'])
            
            # Simple case: Use all data
            if self.allowed_labels is None:
                self.indices = np.arange(total_samples)
            # Filtered case: Find indices matching labels
            else:
                logger.info(f"Filtering for labels: {self.allowed_labels}...")
                Y = f['Y'][:] # Read all labels (fast for ~1M ints)
                mask = np.isin(Y, self.allowed_labels)
                self.indices = np.where(mask)[0]
                logger.info(f"Filtered {total_samples} -> {len(self.indices)} samples.")
        
        self.length = len(self.indices)
        
    def __len__(self):
        return self.length

    # =========================================================================
    # 2. Mode: RAM Loading (preload=True)
    # =========================================================================

    def _opt_load_ram(self):
        """Load filtered dataset into RAM for high-performance access."""
        # Memory Safety Check
        if psutil:
            required_gb = (self.length * 600 * 4) / (1024**3) * 1.5 
            avail_gb = psutil.virtual_memory().available / (1024**3)
            if avail_gb < required_gb + 1.0:
                logger.warning(f"Low RAM (Need ~{required_gb:.1f}GB). Falling back to Disk Mode.")
                self.preload = False
                return

        logger.info(f"Loading {self.length} samples into RAM...")
        try:
            with h5py.File(self.h5_path, 'r') as f:
                # 1. Initialize Cache
                full_shape = f['X'].shape
                total_file_samples = full_shape[0]
                self.data_cache = {}
                self.data_cache['X'] = np.empty((self.length,) + full_shape[1:], dtype=f['X'].dtype)
                
                # 2. Load Metadata (Fast)
                logger.info("Loading Metadata...")
                meta_keys = [k for k in ['Y', 'snr', 'evids'] if k in f]
                for key in meta_keys:
                    target_key = {'Y': 'label', 'evids': 'evid'}.get(key, key)
                    # Read full then slice is fast for small metadata arrays
                    self.data_cache[target_key] = f[key][:][self.indices]

                # 3. Load Waveforms with Progress Bar (The Heavy Part)
                # Iterate file in chunks to show progress and avoid UI freeze
                chunk_size = 10000
                
                with tqdm(total=self.length, desc="Loading RAM", unit="samples") as pbar:
                    
                    current_idx_pos = 0 # Pointer to position in self.indices (and self.data_cache)
                    
                    for start_file_idx in range(0, total_file_samples, chunk_size):
                        end_file_idx = min(start_file_idx + chunk_size, total_file_samples)
                        
                        # Find range in self.indices that belongs to this file chunk
                        # Since self.indices is sorted, we can search effectively
                        next_idx_pos = np.searchsorted(self.indices, end_file_idx)
                        
                        # If there is overlap
                        if next_idx_pos > current_idx_pos:
                            count = next_idx_pos - current_idx_pos
                            
                            # Read raw chunk from disk
                            chunk_data = f['X'][start_file_idx:end_file_idx]
                            
                            # Map desired indices to chunk-relative coordinates
                            target_indices = self.indices[current_idx_pos:next_idx_pos] - start_file_idx
                            
                            # Fill Cache
                            self.data_cache['X'][current_idx_pos:next_idx_pos] = chunk_data[target_indices]
                            
                            pbar.update(count)
                        
                        current_idx_pos = next_idx_pos
                        if current_idx_pos >= self.length:
                            break

            logger.info("RAM Load Complete.")
        except Exception as e:
            logger.error(f"RAM Load failed: {e}. Reverting to Disk Mode.")
            self.preload = False
            self.data_cache = None

    # =========================================================================
    # 3. Mode: Disk Streaming (preload=False) & Access
    # =========================================================================

    def get_dataloader(self, batch_size=1000, num_workers=4, shuffle=False, window_p0=100, window_len=400):
        """
        Create a DataLoader for Streaming/Iterative access.
        
        This is the primary way to use Disk Mode. It handles parallel loading 
        and on-the-fly preprocessing (Windowing + Normalization).
        """
        from seispolarity.generate import FixedWindow, Normalize
        
        # Parallelism strategy
        # RAM mode -> Main process (0 workers) avoids overhead
        # Disk mode -> Multi-process (N workers) hides IO latency
        workers = 0 if self.preload else max(1, num_workers)
        prefetch = 2 if workers > 0 else None
        
        logger.info(f"DataLoader: batch={batch_size}, workers={workers}, shuffle={shuffle}")

        # Transforms
        window = FixedWindow(p0=window_p0, windowlen=window_len)
        normalizer = Normalize(amp_norm_axis=-1, amp_norm_type="peak")

        def collate_fn(batch):
            processed_w = []
            labels = []
            
            for waveform, meta in batch:
                # Stateful transform
                state = {"X": (waveform, meta)}
                window(state)
                normalizer(state)
                
                w, m = state["X"]
                processed_w.append(w.squeeze()) 
                labels.append(m.get("label", -1) if m else -1)
            
            if not processed_w: return np.array([]), np.array([])
            return np.stack(processed_w), np.array(labels)

        return DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=workers, 
            collate_fn=collate_fn,
            prefetch_factor=prefetch
        )

    def __getitem__(self, idx):
        """
        Unified Access Point:
        - If RAM Mode: Returns from self.data_cache
        - If Disk Mode: Reads from HDF5 via persistent handle
        """
        if self.preload and self.data_cache:
            # --- RAM Path ---
            waveform = self.data_cache['X'][idx]
            # Fast dict comprehension for metadata
            metadata = {k: v[idx] for k, v in self.data_cache.items() if k != 'X'}
            
        else:
            # --- Disk Path ---
            # Lazy Open: Only open file handle on first access in this process
            if self._h5_handle is None:
                self._h5_handle = h5py.File(self.h5_path, 'r')
            
            physical_idx = self.indices[idx]
            f = self._h5_handle
            
            waveform = f['X'][physical_idx]
            metadata = {}
            if 'Y' in f:     metadata['label'] = f['Y'][physical_idx]
            if 'snr' in f:   metadata['snr'] = f['snr'][physical_idx]
            if 'evids' in f: metadata['evid'] = f['evids'][physical_idx]

        # Ensure (1, N) shape for PyTorch / SeisBench compatibility
        waveform = waveform.reshape(1, -1).astype(np.float32)
        return waveform, metadata

    # --- Utilities ---
    
    def load_all_data(self, **kwargs):
        """Compatibility wrapper to load processed data into arrays."""
        loader = self.get_dataloader(shuffle=False, **kwargs)
        all_w, all_l = [], []
        for w, l in tqdm(loader, desc="Loading Data", unit="chunk"):
            all_w.append(w)
            all_l.append(l)
        return (np.concatenate(all_w), np.concatenate(all_l)) if all_w else (np.array([]), np.array([]))

    def close(self):
        if self._h5_handle:
            self._h5_handle.close()
            self._h5_handle = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        # Safety for multiprocessing: don't pickle file handles
        state = self.__dict__.copy()
        state['_h5_handle'] = None
        return state
