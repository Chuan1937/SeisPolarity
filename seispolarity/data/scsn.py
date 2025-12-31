import h5py
import numpy as np
from torch.utils.data import Dataset

class SCSNDataset(Dataset):
    """
    Dataset for loading SCSN HDF5 data.
    """
    def __init__(self, h5_path, limit=None):
        self.h5_path = h5_path
        self.limit = limit
        
        with h5py.File(h5_path, 'r') as f:
            self.length = len(f['X'])
            if limit:
                self.length = min(self.length, limit)
                
    def __len__(self):
        return self.length
    
    def get_sample(self, idx):
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
