"""
Instance Dataset Processor for SeisPolarity

This module processes raw Instance seismic data and converts it to format
required by SeisPolarity for training and inference.

Data Format:
- CSV: Contains metadata (trace_name_original, trace_polarity, trace_P_arrival_sample)
- HDF5: Contains waveform data in bucket structure (shape: n_samples, 3, 12000)

Label sampling strategy (matching notebook processing):
- Select all positive samples
- Select all negative samples
- Select undecidable = (positive + negative) * 2

Usage:
    processor = Instance(
        csv_path='/path/to/datasets/Instance/Instance_polarity.csv',
        hdf5_path='/path/to/datasets/Instance/Instance_polarity.hdf5',
        output_polarity='/path/to/datasets/Instance/'
    )
    processor.process()

Author: SeisPolarity
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from .download import fetch_dataset_folder

logger = logging.getLogger(__name__)


class Instance:
    """
    Instance seismic polarity data processor.
    
    Converts raw Instance data to SeisPolarity format:
    - X: Waveform data (n_samples, waveform_length)
    - Y: Polarity labels (0=positive, 1=negative, 2=undecidable)
    - p_pick: P-wave arrival sample indices
    
    Note: Label mapping (SeisPolarity convention):
    - 0 = positive
    - 1 = negative
    - 2 = undecidable
    
    Label sampling strategy (matching notebook processing):
    - Select all positive samples
    - Select all negative samples
    - Apply polarity inversion to both positive and negative to achieve 1:1 balance
    - Select undecidable = positive + negative (after inversion, each category has same count)
    - Final distribution: positive:negative:undecidable = 1:1:1
    
    Example:
        processor = Instance(
            csv_path='/path/to/datasets/Instance/Instance_polarity.csv',
            hdf5_path='/path/to/datasets/Instance/Instance_polarity.hdf5',
            output_polarity='/path/to/datasets/Instance/'
        )
        processor.process()
    """
    
    def __init__(
        self,
        csv_path: str,
        hdf5_path: str,
        output_polarity: str,
        component_order: str = "ENZ",
        component: str = "Z",
        sampling_rate: int = 100,
        auto_download: bool = False,
        use_hf: bool = False,
        force_download: bool = False
    ):
        """
        Initialize Instance processor.
        
        Args:
            csv_path: Path to Instance CSV file
            hdf5_path: Path to Instance HDF5 file
            output_polarity: Directory to save processed files (Instance_polarity.csv and Instance_polarity.hdf5)
            component_order: Order of components in raw HDF5 (default: "ENZ")
            component: Which component to extract (default: "Z" for vertical)
            sampling_rate: Target sampling rate in Hz (default: 100)
            auto_download: If True, automatically download data if csv_path/hdf5_path not found (default: False)
            use_hf: If True, use Hugging Face instead of ModelScope for download (default: False)
            force_download: If True, force re-download even if files exist (default: False)
        """
        self.csv_path = Path(csv_path)
        self.hdf5_path = Path(hdf5_path)
        self.output_dir = Path(output_polarity)
        self.output_csv = self.output_dir / 'Instance_polarity.csv'
        self.output_hdf5 = self.output_dir / 'Instance_polarity.hdf5'
        self.waveform_length = 12000  # Instance fixed waveform length
        self.component_order = component_order
        self.component = component.upper()
        self.sampling_rate = sampling_rate
        self.auto_download = auto_download
        self.use_hf = use_hf
        self.force_download = force_download
        
        # Auto-download if enabled and files don't exist
        if auto_download:
            self._auto_download_data()
        
        # Component index mapping
        self._component_index = self._get_component_index(component_order, component)
        
        # Polarity label mapping (convert from paper convention to SeisPolarity)
        # Paper: positive=1, negative=0, undecidable=2
        # SeisPolarity: positive=0, negative=1, undecidable=2
        self._label_map = {
            'positive': 0,  # Convert from 1 to 0
            'negative': 1,  # Convert from 0 to 1
            'undecidable': 2  # Keep as 2
        }
        
        logger.info("Instance Processor initialized:")
        logger.info(f"  CSV: {self.csv_path}")
        logger.info(f"  HDF5: {self.hdf5_path}")
        logger.info(f"  Output CSV: {self.output_csv}")
        logger.info(f"  Output HDF5: {self.output_hdf5}")
        logger.info(f"  Waveform length: {self.waveform_length} (fixed)")
        logger.info(f"  Component: {self.component}")
        logger.info(f"  Sampling rate: {sampling_rate} Hz")
        logger.info(f"  Auto download: {auto_download}")
        logger.info(f"  Use HF: {use_hf}")
    
    def _auto_download_data(self):
        """
        Automatically download Instance dataset if csv_path or hdf5_path not found.
        
        This method uses fetch_dataset_folder to download the Instance dataset
        from ModelScope or Hugging Face to the specified output directory.
        """
        csv_exists = self.csv_path.exists()
        hdf5_exists = self.hdf5_path.exists()
        
        if not self.force_download and csv_exists and hdf5_exists:
            logger.info("Both CSV and HDF5 files exist, skipping download")
            return
        
        if self.force_download:
            logger.info("Force download enabled, proceeding with download")
        elif not csv_exists:
            logger.warning(f"CSV file not found: {self.csv_path}")
        elif not hdf5_exists:
            logger.warning(f"HDF5 file not found: {self.hdf5_path}")
        else:
            logger.warning("Files missing from data directory")
        
        try:
            logger.info(f"Downloading Instance dataset to: {self.output_dir}")
            downloaded_path = fetch_dataset_folder(
                dataset_name="Instance",
                folder_path=str(self.output_dir),
                cache_dir=str(self.output_dir),
                use_hf=self.use_hf,
                force_download=self.force_download
            )
            logger.info(f"Instance dataset downloaded successfully to: {downloaded_path}")
            
            # Update paths based on downloaded location
            downloaded_path = Path(downloaded_path)
            if downloaded_path.exists():
                # Check for expected files in downloaded directory
                potential_csv = downloaded_path / "Instance_polarity.csv"
                potential_hdf5 = downloaded_path / "Instance_polarity.hdf5"
                
                if potential_csv.exists() and potential_hdf5.exists():
                    self.csv_path = potential_csv
                    self.hdf5_path = potential_hdf5
                    logger.info(f"Updated CSV path: {self.csv_path}")
                    logger.info(f"Updated HDF5 path: {self.hdf5_path}")
                else:
                    logger.warning(f"Expected files not found in downloaded directory: {downloaded_path}")
            else:
                logger.warning(f"Downloaded directory not found: {downloaded_path}")
                
        except Exception as e:
            logger.error(f"Failed to download Instance dataset: {e}")
            raise
    
    def _get_component_index(self, order: str, component: str) -> int:
        """Get component index based on order string."""
        try:
            return order.index(component)
        except ValueError:
            logger.warning(f"Component {component} not found in order {order}, defaulting to Z")
            return order.index('Z')
    
    @staticmethod
    def parse_trace_id(trace_id: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
        """
        Parse Instance trace_name_original format: "bucket0$0,:3,:12000"
        
        Args:
            trace_id: Trace identifier string
            
        Returns:
            Tuple of (group_name, first_idx, second_idx, third_idx)
            Returns (None, None, None, None) on error
        """
        try:
            parts = trace_id.split('$')
            if len(parts) != 2:
                raise ValueError(f"Invalid trace_id format: {trace_id}")
            
            group_name = parts[0]
            trace_part = parts[1]
            indices = trace_part.split(',')
            
            if len(indices) != 3:
                raise ValueError(f"Invalid index format: {trace_part}")
            
            # Parse first index (sample index)
            first_str = indices[0]
            first_index = 0 if first_str == ':' else int(first_str.replace(':', ''))
            
            # Parse second index (components)
            second_str = indices[1]
            second_index = 3 if second_str == ':' else int(second_str.replace(':', ''))
            
            # Parse third index (waveform length)
            third_str = indices[2]
            third_index = 12000 if third_str == ':' else int(third_str.replace(':', ''))
            
            return group_name, first_index, second_index, third_index
            
        except Exception as e:
            logger.warning(f"Failed to parse trace_id {trace_id}: {e}")
            return None, None, None, None
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load raw Instance CSV metadata.
        
        Returns:
            DataFrame with valid samples
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        logger.info(f"Loading Instance CSV: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        logger.info(f"  Total rows: {len(df)}")
        logger.info(f"  Columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['trace_name_original', 'trace_polarity', 'trace_P_arrival_sample']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Log polarity distribution
        if 'trace_polarity' in df.columns:
            polarity_counts = df['trace_polarity'].value_counts()
            logger.info("  Polarity distribution:")
            for pol, count in polarity_counts.items():
                logger.info(f"    {pol}: {count} ({count/len(df)*100:.2f}%)")
        
        # Filter valid samples
        df_valid = df[df['trace_P_arrival_sample'].notna()].copy()
        df_valid = df_valid[df_valid['trace_name_original'].notna()].copy()
        logger.info(f"  Valid samples: {len(df_valid)}")
        
        return df_valid
    
    def sample_by_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sampling strategy to balance labels with polarity inversion.
        After inversion, positive and negative each have (positive + negative) samples.
        Undecidable is sampled to match this count for 1:1:1 balance.
        
        Args:
            df: Input DataFrame with polarity labels
            
        Returns:
            Sampled DataFrame
        """
        df = df.copy()
        
        # Count samples by polarity
        pos_count = len(df[df['trace_polarity'] == 'positive'])
        neg_count = len(df[df['trace_polarity'] == 'negative'])
        undec_count = len(df[df['trace_polarity'] == 'undecidable'])
        
        # After polarity inversion, each of positive and negative will have this count
        total_polarized = pos_count + neg_count
        
        logger.info("Balance strategy (with polarity inversion):")
        logger.info(f"  Positive: {pos_count}")
        logger.info(f"  Negative: {neg_count}")
        logger.info(f"  After inversion - positive: {total_polarized}, negative: {total_polarized}")
        logger.info(f"  Undecidable target: {total_polarized}")
        logger.info(f"  Undecidable available: {undec_count}")
        
        # Sample undecidable to match total_polarized count
        if undec_count >= total_polarized:
            undec_sample = df[df['trace_polarity'] == 'undecidable'].sample(
                total_polarized, random_state=42
            )
        else:
            undec_sample = df[df['trace_polarity'] == 'undecidable']
            logger.warning(f"Undecidable samples ({undec_count}) less than target ({total_polarized})")
        
        result = pd.concat([
            df[df['trace_polarity'] == 'positive'],
            df[df['trace_polarity'] == 'negative'],
            undec_sample
        ]).reset_index(drop=True)
        
        logger.info(f"  Total samples (before inversion): {len(result)}")
        logger.info(f"  Expected after inversion: positive={total_polarized}, negative={total_polarized}, undecidable={len(undec_sample)}")
        return result
    
    def extract_waveform(self, hdf5_file: h5py.File, trace_name: str) -> Optional[np.ndarray]:
        """
        Extract waveform from HDF5 based on trace_name.
        
        Args:
            hdf5_file: Open HDF5 file handle
            trace_name: Trace identifier string
            
        Returns:
            Waveform array or None on failure
        """
        group_name, first_idx, second_idx, third_idx = self.parse_trace_id(trace_name)
        if group_name is None:
            return None
        
        try:
            # Instance HDF5 structure: buckets are under 'data' group
            # Data shape: (n_samples, 3, 12000) where 3 = components (ENZ)
            if 'data' in hdf5_file:
                data_group = hdf5_file['data']
                if group_name in data_group:
                    # Extract vertical component directly (index 0)
                    # waveform shape will be (12000,)
                    waveform = data_group[group_name][first_idx, self._component_index, :third_idx]
                    return waveform
                else:
                    logger.warning(f"Group {group_name} not found in HDF5 data group")
                    return None
            else:
                logger.warning("'data' group not found in HDF5 file")
                return None
        except Exception as e:
            logger.warning(f"Failed to extract waveform for {trace_name}: {e}")
            return None
    
    def process(self):
        """
        Process raw Instance data and save to SeisPolarity format.
        """
        # Load metadata
        df = self.load_metadata()
        
        # Apply sampling strategy
        df = self.sample_by_strategy(df)
        
        # Load HDF5
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")
        
        logger.info(f"Opening HDF5 file: {self.hdf5_path}")
        hdf5_file = h5py.File(self.hdf5_path, 'r')
        
        # Extract waveforms and labels with polarity inversion
        waveforms = []
        labels = []
        p_picks = []
        
        logger.info("Extracting waveforms and labels with polarity inversion...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            trace_name = row['trace_name_original']
            polarity = row['trace_polarity']
            p_arrival = int(row['trace_P_arrival_sample'])
            
            # Extract waveform
            waveform = self.extract_waveform(hdf5_file, trace_name)
            if waveform is None:
                continue
            
            # Trim to desired length
            if len(waveform) > self.waveform_length:
                waveform = waveform[:self.waveform_length]
            elif len(waveform) < self.waveform_length:
                # Pad with zeros if too short
                pad_len = self.waveform_length - len(waveform)
                waveform = np.pad(waveform, (0, pad_len), mode='constant')
            
            # Adjust p_pick if waveform was trimmed
            if p_arrival >= self.waveform_length:
                p_arrival = self.waveform_length - 1
            
            # Apply polarity inversion to achieve 1:1 balance
            if polarity == 'positive':
                # Positive: keep as is (label 0) and inverted (label 1)
                waveforms.append(waveform)
                labels.append(self._label_map['positive'])  # 0
                p_picks.append(p_arrival)
                
                waveforms.append(-waveform)  # Inverted
                labels.append(self._label_map['negative'])  # 1
                p_picks.append(p_arrival)
                
            elif polarity == 'negative':
                # Negative: keep as is (label 1) and inverted (label 0)
                waveforms.append(waveform)
                labels.append(self._label_map['negative'])  # 1
                p_picks.append(p_arrival)
                
                waveforms.append(-waveform)  # Inverted
                labels.append(self._label_map['positive'])  # 0
                p_picks.append(p_arrival)
                
            else:  # undecidable
                # Undecidable: keep as is (label 2)
                waveforms.append(waveform)
                labels.append(self._label_map['undecidable'])  # 2
                p_picks.append(p_arrival)
        
        hdf5_file.close()
        
        logger.info(f"Successfully extracted {len(waveforms)} samples (with polarity inversion)")
        
        # Convert to arrays
        X = np.array(waveforms, dtype=np.float32)
        Y = np.array(labels, dtype=np.int32)
        p_pick = np.array(p_picks, dtype=np.int32)
        
        logger.info(f"Waveforms shape: {X.shape}")
        logger.info(f"Labels shape: {Y.shape}")
        logger.info(f"P-picks shape: {p_pick.shape}")
        
        # Log final label distribution
        unique, counts = np.unique(Y, return_counts=True)
        label_names = ['positive', 'negative', 'undecidable']
        logger.info("Final label distribution:")
        for u, c in zip(unique, counts):
            logger.info(f"  {label_names[u] if u < len(label_names) else str(u)}: {c}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV output (selected samples metadata)
        logger.info(f"Saving CSV output to: {self.output_csv}")
        output_df = df.copy()
        output_df.to_csv(self.output_csv, index=False)
        
        # Save HDF5 output
        logger.info(f"Saving HDF5 output to: {self.output_hdf5}")
        
        with h5py.File(self.output_hdf5, 'w') as f:
            f.create_dataset('X', data=X, compression='gzip')
            f.create_dataset('Y', data=Y, compression='gzip')
            f.create_dataset('p_pick', data=p_pick, compression='gzip')
            
            # Save metadata
            f.attrs['waveform_length'] = self.waveform_length
            f.attrs['sampling_rate'] = self.sampling_rate
            f.attrs['component'] = self.component
            f.attrs['num_samples'] = len(X)
            f.attrs['sampling_strategy'] = 'polarity_inversion_1to1to1'
            f.attrs['label_map'] = str(self._label_map)
        
        logger.info("Processed data saved successfully!")
        logger.info(f"  CSV: {self.output_csv}")
        logger.info(f"  HDF5: {self.output_hdf5}")
        logger.info("You can now use this file with WaveformDataset:")
        logger.info(f"  dataset = WaveformDataset(path='{self.output_hdf5}')")
