import h5py
import numpy as np
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
from typing import Tuple

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
    Dataset for loading SCSN HDF5 data.
    Supports optional RAM preloading for performance.
    Automatically downloads data from Hugging Face if h5_path is not provided.
    Supports filtering by label (e.g. only 0 and 1).
    """
    def load_all_data(self, window_p0: int = 100, window_len: int = 400, num_workers: int = 4, batch_size: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess all waveforms efficiently into memory.
        Uses multiprocessing or direct memory access depending on preload state.
        
        Args:
            window_p0 (int): P-pick center index for windowing.
            window_len (int): Output window length.
            num_workers (int): Number of workers for DataLoader (if not preloaded).
            batch_size (int): Batch size for loading.
            
        Returns:
            (np.ndarray, np.ndarray): Waveforms (N, L), Labels (N,)
        """
        from seispolarity.generate import FixedWindow, Normalize

        # If data fits in RAM (preload=True), prefer single-process loading (workers=0)
        # to avoid multiprocessing serialization overhead.
        effective_workers = 0 if self.preload else max(1, num_workers)
        prefetch_factor = None if effective_workers == 0 else 2

        logger.info(
            "Loading strategy: preload=%s workers=%s batch_size=%s",
            self.preload,
            effective_workers,
            batch_size,
        )
        
        window = FixedWindow(p0=window_p0, windowlen=window_len)
        normalizer = Normalize(amp_norm_axis=-1, amp_norm_type="peak")
        
        def collate_fn(batch):
            processed_waveforms = []
            labels = []
            
            for waveform, metadata in batch:
                state = {"X": (waveform, metadata)}
                window(state)
                normalizer(state)
                
                w, m = state["X"]
                processed_waveforms.append(w.squeeze()) # (1, L) -> (L,)
                labels.append(m.get("label", -1) if m else -1)
                
            return np.stack(processed_waveforms), np.array(labels)
            
        loader_kwargs = {
            "dataset": self,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": effective_workers,
            "collate_fn": collate_fn,
        }
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

        loader = DataLoader(**loader_kwargs)

        all_waveforms = []
        all_labels = []
        
        for waveforms, labels in tqdm(loader, desc="Loading Data", unit="chunk"):
            all_waveforms.append(waveforms)
            all_labels.append(labels)
            
        if not all_waveforms:
            return np.array([]), np.array([])
            
        return np.concatenate(all_waveforms), np.concatenate(all_labels)

    def __init__(self, h5_path=None, limit=None, preload=False, split="train", allowed_labels=None):
        """
        :param h5_path: Path to HDF5 file. If None, downloads from HF based on split.
        :param limit: Limit number of samples.
        :param preload: Whether to preload data into RAM.
        :param split: 'train' or 'test' (only used if h5_path is None).
        :param allowed_labels: List of labels to keep (e.g. [0, 1]). If None, keep all.
        """
        if h5_path is None:
            # Default HF paths
            repo_id = "chuanjun1978/Seismic-AI-Data"
            if split == "train":
                filename = "scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5"
            elif split == "test":
                filename = "scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5"
            else:
                raise ValueError(f"Unknown split '{split}'. Use 'train' or 'test'.")
            
            repo_path = f"SCEDC/{filename}"
            logger.info(f"No h5_path provided. Fetching {repo_path} from HF...")
            h5_path = fetch_hf_file(repo_id=repo_id, repo_path=repo_path)
            logger.info(f"Using dataset at: {h5_path}")

        self.h5_path = str(h5_path)
        self.limit = limit
        self.preload = preload
        self.data_cache = None
        """
        with h5py.File(self. Path to HDF5 file. If None, downloads from HF based on split.
        :param limit: Limit number of samples.
        :param preload: Whether to preload data into RAM.
        :param split: 'train' or 'test' (only used if h5_path is None).
        """
        if h5_path is None:
            # Default HF paths
            repo_id = "chuanjun1978/Seismic-AI-Data"
            if split == "train":
                filename = "scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5"
            elif split == "test":
                filename = "scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5"
            else:
                raise ValueError(f"Unknown split '{split}'. Use 'train' or 'test'.")
            
            repo_path = f"SCEDC/{filename}"
            logger.info(f"No h5_path provided. Fetching {repo_path} from HF...")
            h5_path = fetch_hf_file(repo_id=repo_id, repo_path=repo_path)
            logger.info(f"Using dataset at: {h5_path}")

        self.h5_path = str(h5_path)
        self.limit = limit
        self.preload = preload
        self.data_cache = None
        self.allowed_labels = allowed_labels
        self.indices = None # logical -> physical index map

        with h5py.File(self.h5_path, 'r') as f:
            total_samples = len(f['X'])
            if self.allowed_labels is not None:
                logger.info(f"Filtering dataset for labels: {self.allowed_labels}...")
                # Read all labels (should be fast for ~18MB)
                Y = f['Y'][:]
                mask = np.isin(Y, self.allowed_labels)
                self.indices = np.where(mask)[0]
                self.length = len(self.indices)
                logger.info(f"Filtered {total_samples} -> {self.length} samples.")
            else:
                self.indices = np.arange(total_samples)
                self.length = total_samples
            
            if limit:
                if self.length > limit:
                    self.length = limit
                    self.indices = self.indices[:limit]
        
        if self.preload:
            self._preload_data()
                
    def _preload_data(self):
        """Load data into RAM if memory permits."""
        # Check memory
        if psutil:
            vm = psutil.virtual_memory()
            avail_gb = vm.available / (1024**3)
            # Estimate size: length * 600 * 4 bytes (float32)
            # Plus metadata overhead. Rough estimate.
            required_gb = (self.length * 600 * 4) / (1024**3) * 1.5 
            
            if avail_gb < required_gb + 1.0: # Leave 1GB buffer
                logger.warning(f"Not enough RAM to preload {self.h5_path}. "
                               f"Available: {avail_gb:.2f}GB, Required: ~{required_gb:.2f}GB. "
                               "Falling back to disk-based loading.")
                self.preload = False
                return

        logger.info(f"Preloading {self.h5_path} into RAM...")
        try:
            with h5py.File(self.h5_path, 'r') as f:
                self.data_cache = {}
                
                # Pre-allocate X
                # Assuming X is (N, 600)
                x_shape = (self.length,) + f['X'].shape[1:]
                self.data_cache['X'] = np.empty(x_shape, dtype=f['X'].dtype)
                
                # Identify other keys to load
                keys_map = {} # file_key -> cache_key
                if 'Y' in f: keys_map['Y'] = 'Y'
                if 'snr' in f: keys_map['snr'] = 'snr'
                if 'evids' in f: keys_map['evids'] = 'evid' # Rename evids -> evid
                
                # Pre-allocate others
                for f_key, c_key in keys_map.items():
                    shape = (self.length,) + f[f_key].shape[1:]
                    self.data_cache[c_key] = np.empty(shape, dtype=f[f_key].dtype)
                
                # Chunked loading with tqdm
                chunk_size = 100000
                total_chunks = (self.length + chunk_size - 1) // chunk_size
                
                desc = f"Loading {self.h5_path.split('/')[-1]}"
                
                # If we have indices (filtering enabled), we can't use simple slicing for contiguous reads if indices are sparse.
                # However, usually we preload everything relevant.
                # If indices are not contiguous, h5py slicing with list is slow/not supported efficiently.
                # Only use smart slicing if no filtering.
                if self.allowed_labels is None and self.limit is None:
                    # Optimized contiguous load
                    for i in tqdm(range(0, self.length, chunk_size), desc=desc, total=total_chunks, unit="chunk"):
                        end = min(i + chunk_size, self.length)
                        self.data_cache['X'][i:end] = f['X'][i:end]
                        for f_key, c_key in keys_map.items():
                            self.data_cache[c_key][i:end] = f[f_key][i:end]
                else:
                    # Filtered load Optimization
                    # Using f['X'][indices] is very slow in h5py.
                    # It is much faster to load the full dataset (sequentially) into RAM and then filter it,
                    # provided we have enough RAM (which we checked in _preload_data).
                    
                    logger.info("Loading filtered data: Reading FULL dataset sequentially for speed, then filtering in RAM...")
                    
                    # 1. Load FULL X temporarily
                    full_len = f['X'].shape[0]
                    # Check if we have enough RAM for full copy + filtered copy? 
                    # If dataset is 6GB, full copy is 6GB. Filtered is < 6GB. Total 12GB peak.
                    # Assuming available RAM is sufficient (checked earlier with margin).
                    
                    # Read Full X in one go or large chunks
                    # Actually, we can just read chunk by chunk of the FULL dataset, filter, and assign.
                    # This avoids holding 2x data in memory.
                    
                    # Re-calculate chunks based on FULL length
                    full_total_chunks = (full_len + chunk_size - 1) // chunk_size
                    desc = f"Scanning {self.h5_path.split('/')[-1]} for filtering"
                    
                    # Indices mask for the whole file
                    # We need a boolean mask for the full file to know which rows in a chunk to keep
                    if self.allowed_labels is not None:
                        # We already have self.indices which are the indices to keep.
                        # Create a full boolean mask
                        full_mask = np.zeros(full_len, dtype=bool)
                        full_mask[self.indices] = True
                    else:
                        full_mask = np.ones(full_len, dtype=bool)
                        if self.limit:
                             full_mask[self.limit:] = False

                    # Current pointer in the destination (self.data_cache)
                    dest_ptr = 0
                    
                    for i in tqdm(range(0, full_len, chunk_size), desc=desc, total=full_total_chunks, unit="chunk"):
                        file_end = min(i + chunk_size, full_len)
                        
                        # Get mask for this chunk
                        chunk_mask = full_mask[i:file_end]
                        if not np.any(chunk_mask):
                            continue
                            
                        # Read chunk from file (FAST sequential read)
                        chunk_X = f['X'][i:file_end]
                        
                        # Filter
                        filtered_X = chunk_X[chunk_mask]
                        n_keep = len(filtered_X)
                        
                        # Assign to cache
                        self.data_cache['X'][dest_ptr : dest_ptr + n_keep] = filtered_X
                        
                        # Handle other keys
                        for f_key, c_key in keys_map.items():
                            chunk_data = f[f_key][i:file_end]
                            self.data_cache[c_key][dest_ptr : dest_ptr + n_keep] = chunk_data[chunk_mask]
                            
                        dest_ptr += n_keep
                    
                    # Verify
                    if dest_ptr != self.length:
                        logger.warning(f"Filter mismatch: expected {self.length} samples, loaded {dest_ptr}.")

            logger.info("Preloading complete.")
        except Exception as e:
            logger.error(f"Preloading failed: {e}. Falling back to disk.")
            self.preload = False
            self.data_cache = None

    def __len__(self):
        return self.length
    
    def get_sample(self, idx):
        if self.preload and self.data_cache is not None:
             # preload data cache is already filtered and dense (0...length-1)
             # So we use direct idx
            waveform = self.data_cache['X'][idx]
            metadata = {}
            if 'Y' in self.data_cache:
                metadata['label'] = self.data_cache['Y'][idx]
            if 'snr' in self.data_cache:
                metadata['snr'] = self.data_cache['snr'][idx]
            if 'evid' in self.data_cache:
                metadata['evid'] = self.data_cache['evid'][idx]
        else:
             # Map logical index to physical index
            physical_idx = self.indices[idx]
            with h5py.File(self.h5_path, 'r') as f:
                waveform = f['X'][physical_idx]
                # Metadata
                metadata = {}
                if 'Y' in f:
                    metadata['label'] = f['Y'][physical_idx]
                if 'snr' in f:
                    metadata['snr'] = f['snr'][physical_idx]
                if 'evids' in f:
                    metadata['evid'] = f['evids'][physical_idx]
            
        # Waveform shape is (600,) -> (1, 600) for consistency with SeisBench (C, N)
        waveform = waveform.reshape(1, -1).astype(np.float32)
        
        return waveform, metadata

    def __getitem__(self, idx):
        # This is for direct usage without Generator
        return self.get_sample(idx)
