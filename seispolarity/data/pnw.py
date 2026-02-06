"""
PNW Dataset Processor for SeisPolarity

This module processes raw PNW seismic data and converts it to format
required by SeisPolarity for training and inference.

Data Format:
- CSV: Contains metadata
- HDF5: Contains waveform data (typically with 'data' and 'data_format' keys)

Label sampling strategy:
1. If unknown_count > 2 * (up_count + down_count):
   - Select all up and down samples
   - Select unknown = 2 * (up_count + down_count)
2. If unknown_count < 2 * (up_count + down_count):
   - Find minimum count m among up, down, unknown
   - Select m samples from each category

Usage:
    processor = PNW(
        csv_path='/path/to/pnw_polarity.csv',
        hdf5_path='/path/to/pnw_polarity.hdf5',
        output_polarity='/path/to/PNW/'
    )
    processor.process()

Author: SeisPolarity
"""

import logging
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Literal
from tqdm import tqdm

from .download import fetch_dataset_folder

logger = logging.getLogger(__name__)


class PNW:
    """
    PNW seismic polarity data processor.
    
    Converts raw PNW data to SeisPolarity format:
    - X: Waveform data (n_samples, waveform_length)
    - Y: Polarity labels (0=U/up, 1=D/down, 2=unknown)
    - p_pick: P-wave arrival sample indices
    
    Label sampling strategy:
    1. If unknown_count > 2 * (up_count + down_count):
       - Select all up and down samples
       - Select unknown = 2 * (up_count + down_count)
    2. If unknown_count < 2 * (up_count + down_count):
       - Find minimum count m among up, down, unknown
       - Select m samples from each category
    
    Example:
        processor = PNW(
            csv_path='/path/to/pnw_polarity.csv',
            hdf5_path='/path/to/pnw_polarity.hdf5',
            output_polarity='/path/to/PNW/'
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
        Initialize PNW processor.
        
        Args:
            csv_path: Path to PNW CSV file
            hdf5_path: Path to PNW HDF5 file
            output_polarity: Directory to save processed files (PNW_polarity.csv and PNW_polarity.hdf5)
            component_order: Order of components in raw HDF5 (default: "ENZ")
            component: Which component to extract (default: "Z" for vertical)
            sampling_rate: Target sampling rate in Hz (default: 100)
            auto_download: If True, automatically download data if csv_path or hdf5_path not found (default: False)
            use_hf: If True, use Hugging Face instead of ModelScope for download (default: False)
            force_download: If True, force re-download even if files exist (default: False)
        """
        self.csv_path = Path(csv_path)
        self.hdf5_path = Path(hdf5_path)
        self.output_dir = Path(output_polarity)
        self.output_csv = self.output_dir / 'PNW_polarity.csv'
        self.output_hdf5 = self.output_dir / 'PNW_polarity.hdf5'
        self.waveform_length = 15001  # PNW fixed waveform length
        self.component_order = component_order
        self.component = component.upper()
        self.sampling_rate = sampling_rate
        self.auto_download = auto_download
        self.use_hf = use_hf
        self.force_download = force_download
        
        # Auto-download if enabled and paths don't exist
        if auto_download:
            self._auto_download_data()
        
        # Component index mapping
        self._component_index = self._get_component_index(component_order, component)
        
        # Polarity label mapping (PNW uses positive/negative/undecidable convention)
        # positive = 0 (U), negative = 1 (D), undecidable = 2 (X)
        self._label_map = {
            'positive': 0,  # U / positive
            'negative': 1,  # D / negative
            'undecidable': 2,  # X / unknown
            'U': 0,  # Legacy support
            'D': 1,  # Legacy support
            'Unknown': 2,  # Legacy support
            'unknown': 2,  # Legacy support
            'X': 2  # Legacy support
        }
        
        logger.info(f"PNW Processor initialized:")
        logger.info(f"  CSV: {self.csv_path}")
        logger.info(f"  HDF5: {self.hdf5_path}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Output CSV: {self.output_csv}")
        logger.info(f"  Output HDF5: {self.output_hdf5}")
        logger.info(f"  Waveform length: {self.waveform_length} (fixed)")
        logger.info(f"  Component: {self.component}")
        logger.info(f"  Sampling rate: {sampling_rate} Hz")
        logger.info(f"  Auto download: {auto_download}")
        logger.info(f"  Use HF: {use_hf}")
    
    def _auto_download_data(self):
        """
        Automatically download PNW dataset if csv_path or hdf5_path not found.
        
        This method uses fetch_dataset_folder to download the PNW dataset
        from ModelScope or Hugging Face to the specified output directory.
        """
        csv_exists = self.csv_path.exists()
        hdf5_exists = self.hdf5_path.exists()
        
        if not self.force_download and csv_exists and hdf5_exists:
            logger.info(f"Both CSV and HDF5 files exist, skipping download")
            return
        
        if self.force_download:
            logger.info(f"Force download enabled, proceeding with download")
        elif not csv_exists:
            logger.warning(f"CSV file not found: {self.csv_path}")
        elif not hdf5_exists:
            logger.warning(f"HDF5 file not found: {self.hdf5_path}")
        
        try:
            logger.info(f"Downloading PNW dataset to: {self.output_dir}")
            downloaded_path = fetch_dataset_folder(
                dataset_name="PNW",
                folder_path="PNW",
                cache_dir=self.output_dir.parent,
                use_hf=self.use_hf,
                force_download=self.force_download
            )
            logger.info(f"PNW dataset downloaded successfully to: {downloaded_path}")
            
            # Update paths based on downloaded location (raw data: PNW.csv and PNW.hdf5)
            downloaded_path = Path(downloaded_path)
            expected_csv = downloaded_path / "PNW.csv"
            expected_hdf5 = downloaded_path / "PNW.hdf5"
            
            if expected_csv.exists() and expected_hdf5.exists():
                self.csv_path = expected_csv
                self.hdf5_path = expected_hdf5
                logger.info(f"Updated paths (raw data):")
                logger.info(f"  CSV: {self.csv_path}")
                logger.info(f"  HDF5: {self.hdf5_path}")
            else:
                logger.warning(f"Expected raw files not found in downloaded directory")
                logger.warning(f"Expected CSV: {expected_csv}")
                logger.warning(f"Expected HDF5: {expected_hdf5}")
                
        except Exception as e:
            logger.error(f"Failed to download PNW dataset: {e}")
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
        Parse PNW trace_name format: "bucket4$0,:3,:15001"
        
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
            third_index = 15001 if third_str == ':' else int(third_str.replace(':', ''))
            
            return group_name, first_index, second_index, third_index
            
        except Exception as e:
            logger.warning(f"Failed to parse trace_id {trace_id}: {e}")
            return None, None, None, None
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load raw PNW CSV metadata.
        
        Returns:
            DataFrame with valid samples
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        logger.info(f"Loading PNW CSV: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        logger.info(f"  Total rows: {len(df)}")
        logger.info(f"  Columns: {list(df.columns)}")
        
        # Determine column names (may vary)
        polarity_col = None
        trace_name_col = None
        p_pick_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'polar' in col_lower:
                polarity_col = col
            elif 'trace_name' in col_lower or 'name' in col_lower:
                trace_name_col = col
            elif 'p_arrival' in col_lower or 'p_pick' in col_lower or 'pick' in col_lower:
                p_pick_col = col
        
        logger.info(f"  Polarity column: {polarity_col}")
        logger.info(f"  Trace name column: {trace_name_col}")
        logger.info(f"  P-pick column: {p_pick_col}")
        
        self.polarity_col = polarity_col
        self.trace_name_col = trace_name_col
        self.p_pick_col = p_pick_col
        
        # Log polarity distribution
        if polarity_col and polarity_col in df.columns:
            polarity_counts = df[polarity_col].value_counts()
            logger.info("  Polarity distribution:")
            for pol, count in polarity_counts.items():
                logger.info(f"    {pol}: {count} ({count/len(df)*100:.2f}%)")
        
        # Filter valid samples
        df_valid = df.copy()
        if p_pick_col:
            df_valid = df_valid[df_valid[p_pick_col].notna()]
        if trace_name_col:
            df_valid = df_valid[df_valid[trace_name_col].notna()]
        
        logger.info(f"  Valid samples: {len(df_valid)}")
        
        return df_valid
    
    def sample_by_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sampling strategy to balance labels.
        
        Args:
            df: Input DataFrame with polarity labels
            
        Returns:
            Sampled DataFrame
        """
        if self.polarity_col is None:
            logger.warning("No polarity column found, returning all data")
            return df
        
        df = df.copy()
        
        # Map labels to positive/negative/undecidable
        df['_label_category'] = df[self.polarity_col].apply(
            lambda x: 'positive' if str(x) == 'positive' 
                     else 'negative' if str(x) == 'negative'
                     else 'undecidable'
        )
        
        counts = df['_label_category'].value_counts()
        positive_count = counts.get('positive', 0)
        negative_count = counts.get('negative', 0)
        undecidable_count = counts.get('undecidable', 0)
        
        logger.info(f"Label counts - positive: {positive_count}, negative: {negative_count}, undecidable: {undecidable_count}")
        
        total_polarized = positive_count + negative_count
        
        # Strategy 1: If undecidable_count > 2 * (positive + negative)
        if undecidable_count > 2 * total_polarized:
            logger.info("Using Strategy 1: undecidable > 2*(positive+negative)")
            target_undecidable = 2 * total_polarized
            
            # Select all positive and negative
            sampled_polarized = df[df['_label_category'].isin(['positive', 'negative'])]
            
            # Select target number of undecidable samples
            undecidable_pool = df[df['_label_category'] == 'undecidable']
            sampled_undecidable = undecidable_pool.sample(min(target_undecidable, len(undecidable_pool)), random_state=42)
            
            result = pd.concat([sampled_polarized, sampled_undecidable])
        
        # Strategy 2: If undecidable_count < 2 * (positive + negative)
        else:
            logger.info("Using Strategy 2: undecidable < 2*(positive+negative)")
            m = min(positive_count, negative_count, undecidable_count)
            
            logger.info(f"Minimum count m = {m}")
            
            # Sample m from each category
            result_list = []
            for label in ['positive', 'negative', 'undecidable']:
                pool = df[df['_label_category'] == label]
                sampled = pool.sample(m, random_state=42)
                result_list.append(sampled)
            
            result = pd.concat(result_list)
        
        logger.info(f"Sampled {len(result)} samples")
        return result.reset_index(drop=True)
    
    def extract_waveform(self, hdf5_file: h5py.File, trace_name: str) -> Optional[np.ndarray]:
        """
        Extract waveform from HDF5 based on trace_name.
        
        Args:
            hdf5_file: Open HDF5 file handle
            trace_name: Trace identifier string
            
        Returns:
            Waveform array or None on failure
        """
        try:
            # PNW HDF5 structure: buckets are under 'data' group
            # Data shape: (n_samples, 3, 15001) where 3 = components (ENZ)
            # We need Z component (index 2 if order is ENZ)
            
            if '$' in trace_name:
                group_name, first_idx, second_idx, third_idx = self.parse_trace_id(trace_name)
                if group_name and 'data' in hdf5_file:
                    data_group = hdf5_file['data']
                    if group_name in data_group:
                        # Extract Z component (vertical) directly
                        # waveform shape will be (15001,)
                        waveform = data_group[group_name][first_idx, self._component_index, :third_idx]
                        return waveform
                    else:
                        logger.warning(f"Group {group_name} not found in HDF5 data group")
                        return None
                else:
                    logger.warning(f"Invalid trace_name or 'data' group not found")
                    return None
            else:
                logger.warning(f"Invalid trace_name format: {trace_name}")
                return None
            
        except Exception as e:
            logger.warning(f"Failed to extract waveform for {trace_name}: {e}")
            return None
    
    def process(self):
        """
        Process raw PNW data and save to SeisPolarity format.
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
        
        # Extract waveforms and labels
        waveforms = []
        labels = []
        p_picks = []
        
        logger.info("Extracting waveforms and labels...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            # Get trace name
            trace_name = str(row.get(self.trace_name_col, '')) if self.trace_name_col else str(idx)
            
            # Get polarity label
            if self.polarity_col:
                polarity = row[self.polarity_col]
                if pd.notna(polarity):
                    polarity_str = str(polarity)
                    label = self._label_map.get(polarity_str, self._label_map.get('unknown', 2))
                else:
                    label = 2  # unknown
            else:
                label = 2  # unknown
            
            # Get p_pick
            p_arrival = None
            if self.p_pick_col and pd.notna(row.get(self.p_pick_col)):
                p_arrival = int(row[self.p_pick_col])
            
            # Extract waveform (already extracted the correct component)
            waveform = self.extract_waveform(hdf5_file, trace_name)
            if waveform is None:
                continue
            
            # Trim or pad to desired length
            if len(waveform) > self.waveform_length:
                waveform = waveform[:self.waveform_length]
            elif len(waveform) < self.waveform_length:
                # Pad with zeros if too short
                pad_len = self.waveform_length - len(waveform)
                waveform = np.pad(waveform, (0, pad_len), mode='constant')
            
            # Set p_pick if not available (center of waveform)
            if p_arrival is None or p_arrival >= self.waveform_length:
                p_arrival = self.waveform_length // 2
            
            waveforms.append(waveform)
            labels.append(label)
            p_picks.append(p_arrival)
        
        hdf5_file.close()
        
        logger.info(f"Successfully extracted {len(waveforms)} samples")
        
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
        
        # Save processed data
        logger.info(f"Saving processed data to: {self.output_hdf5}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(self.output_hdf5, 'w') as f:
            f.create_dataset('X', data=X, compression='gzip')
            f.create_dataset('Y', data=Y, compression='gzip')
            f.create_dataset('p_pick', data=p_pick, compression='gzip')
            
            # Save metadata
            f.attrs['waveform_length'] = self.waveform_length
            f.attrs['sampling_rate'] = self.sampling_rate
            f.attrs['component'] = self.component
            f.attrs['num_samples'] = len(X)
            f.attrs['sampling_strategy'] = 'smart'
            f.attrs['label_map'] = str(self._label_map)
        
        logger.info(f"Processed data saved successfully!")
        logger.info(f"  CSV: {self.output_csv}")
        logger.info(f"  HDF5: {self.output_hdf5}")
        logger.info("You can now use this file with WaveformDataset:")
        logger.info(f"  dataset = WaveformDataset(path='{self.output_hdf5}')")
