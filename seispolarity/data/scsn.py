import h5py
import numpy as np
from torch.utils.data import Dataset
import logging
from tqdm import tqdm

try:
    import psutil
except ImportError:
    psutil = None

from seispolarity.data.download import fetch_hf_file
from seispolarity.config import settings

logger = logging.getLogger(__name__)

class SCSNDataset(Dataset):
    """
    Dataset for loading SCSN HDF5 data.
    Supports optional RAM preloading for performance.
    Automatically downloads data from Hugging Face if h5_path is not provided.
    """
    def __init__(self, h5_path=None, limit=None, preload=False, split="train"):
        """
        :param h5_path: Path to HDF5 file. If None, downloads from HF based on split.
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
        
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['X'])
            if limit:
                self.length = min(self.length, limit)
        
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
                for i in tqdm(range(0, self.length, chunk_size), desc=desc, total=total_chunks, unit="chunk"):
                    end = min(i + chunk_size, self.length)
                    
                    self.data_cache['X'][i:end] = f['X'][i:end]
                    for f_key, c_key in keys_map.items():
                        self.data_cache[c_key][i:end] = f[f_key][i:end]

            logger.info("Preloading complete.")
        except Exception as e:
            logger.error(f"Preloading failed: {e}. Falling back to disk.")
            self.preload = False
            self.data_cache = None

    def __len__(self):
        return self.length
    
    def get_sample(self, idx):
        if self.preload and self.data_cache is not None:
            waveform = self.data_cache['X'][idx]
            metadata = {}
            if 'Y' in self.data_cache:
                metadata['label'] = self.data_cache['Y'][idx]
            if 'snr' in self.data_cache:
                metadata['snr'] = self.data_cache['snr'][idx]
            if 'evid' in self.data_cache:
                metadata['evid'] = self.data_cache['evid'][idx]
        else:
            with h5py.File(self.h5_path, 'r') as f:
                waveform = f['X'][idx]
                # Metadata
                metadata = {}
                if 'Y' in f:
                    metadata['label'] = f['Y'][idx]
                if 'snr' in f:
                    metadata['snr'] = f['snr'][idx]
                if 'evids' in f:
                    metadata['evid'] = f['evids'][idx]
            
        # Waveform shape is (600,) -> (1, 600) for consistency with SeisBench (C, N)
        waveform = waveform.reshape(1, -1).astype(np.float32)
        
        return waveform, metadata

    def __getitem__(self, idx):
        # This is for direct usage without Generator
        return self.get_sample(idx)
