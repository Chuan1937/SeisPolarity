"""
TXED Dataset Processor for SeisPolarity

This module processes raw TXED seismic data and converts it to format
required by SeisPolarity for training and inference.

Data Format:
- CSV: Contains metadata (trace_name, trace_polarity, trace_p_arrival_sample)
- HDF5: Contains waveform data in bucket structure (shape: n_samples, 3, 6000)

Label sampling strategy (matching notebook processing):
- Select all up samples
- Select all down samples
- Select unknown = 2 * (up_count + down_count)

Usage:
    processor = TXED(
        csv_path='/path/to/datasets/TXED/TXED.csv',
        hdf5_path='/path/to/datasets/TXED/TXED.hdf5',
        output_polarity='/path/to/datasets/TXED/'
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

logger = logging.getLogger(__name__)


class TXED:
    """
    TXED seismic polarity data processor.
    
    Converts raw TXED data to SeisPolarity format:
    - X: Waveform data (n_samples, waveform_length)
    - Y: Polarity labels (0=U/up, 1=D/down, 2=K/X/unknown)
    - p_pick: P-wave arrival sample indices
    
    Label sampling strategy (matching notebook processing):
    - Select all up samples
    - Select all down samples
    - Select unknown = 2 * (up_count + down_count)
    
    Example:
        processor = TXED(
            csv_path='/path/to/datasets/TXED/TXED.csv',
            hdf5_path='/path/to/datasets/TXED/TXED.hdf5',
            output_polarity='/path/to/datasets/TXED/'
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
        sampling_rate: int = 100
    ):
        """
        Initialize TXED processor.
        
        Args:
            csv_path: Path to TXED CSV file
            hdf5_path: Path to TXED HDF5 file
            output_polarity: Directory to save processed files (TXED_polarity.csv and TXED_polarity.hdf5)
            component_order: Order of components in raw HDF5 (default: "ENZ")
            component: Which component to extract (default: "Z" for vertical)
            sampling_rate: Target sampling rate in Hz (default: 100)
        """
        self.csv_path = Path(csv_path)
        self.hdf5_path = Path(hdf5_path)
        self.output_dir = Path(output_polarity)
        self.output_csv = self.output_dir / 'TXED_polarity.csv'
        self.output_hdf5 = self.output_dir / 'TXED_polarity.hdf5'
        self.waveform_length = 6000  # TXED fixed waveform length
        self.component_order = component_order
        self.component = component.upper()
        self.sampling_rate = sampling_rate
        
        # Component index mapping
        self._component_index = self._get_component_index(component_order, component)
        
        # Polarity label mapping
        self._label_map = {
            'U': 0,  # Up / positive
            'D': 1,  # Down / negative
            'unknown': 2  # Unknown
        }
        
        logger.info(f"TXED Processor initialized:")
        logger.info(f"  CSV: {self.csv_path}")
        logger.info(f"  HDF5: {self.hdf5_path}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Output CSV: {self.output_csv}")
        logger.info(f"  Output HDF5: {self.output_hdf5}")
        logger.info(f"  Waveform length: {self.waveform_length} (fixed)")
        logger.info(f"  Component: {self.component}")
        logger.info(f"  Sampling rate: {sampling_rate} Hz")
    
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
        Parse TXED trace_name format: "bucket209$106,:3,:6000"
        
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
            third_index = 6000 if third_str == ':' else int(third_str.replace(':', ''))
            
            return group_name, first_index, second_index, third_index
            
        except Exception as e:
            logger.warning(f"Failed to parse trace_id {trace_id}: {e}")
            return None, None, None, None
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load raw TXED CSV metadata.
        
        Returns:
            DataFrame with valid samples
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        logger.info(f"Loading TXED CSV: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        logger.info(f"  Total rows: {len(df)}")
        logger.info(f"  Columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['trace_name', 'trace_polarity', 'trace_p_arrival_sample']
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
        df_valid = df[df['trace_p_arrival_sample'].notna()].copy()
        df_valid = df_valid[df_valid['trace_name'].notna()].copy()
        logger.info(f"  Valid samples: {len(df_valid)}")
        
        return df_valid
    
    def sample_by_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply notebook-based sampling strategy: unknown = 2 * (U + D).
        
        Args:
            df: Input DataFrame with polarity labels
            
        Returns:
            Sampled DataFrame
        """
        df = df.copy()
        
        # Map labels to U/D/unknown
        df['_label_category'] = df['trace_polarity'].apply(
            lambda x: 'U' if str(x) == 'U' 
                     else 'D' if str(x) == 'D'
                     else 'unknown'
        )
        
        counts = df['_label_category'].value_counts()
        u_count = counts.get('U', 0)
        d_count = counts.get('D', 0)
        unknown_count = counts.get('unknown', 0)
        
        logger.info(f"Label counts - U: {u_count}, D: {d_count}, unknown: {unknown_count}")
        
        total_ud = u_count + d_count
        target_unknown = 2 * total_ud
        
        logger.info(f"Notebook strategy: select all U and D, unknown = 2 * (U + D)")
        logger.info(f"  Target unknown count: {target_unknown}")
        
        # Select all U and D samples
        sampled_ud = df[df['_label_category'].isin(['U', 'D'])]
        
        # Select target number of unknown samples
        unknown_pool = df[df['_label_category'] == 'unknown']
        if len(unknown_pool) >= target_unknown:
            sampled_unknown = unknown_pool.sample(target_unknown, random_state=42)
            logger.info(f"  Sampled {target_unknown} unknown from {len(unknown_pool)} available")
        else:
            sampled_unknown = unknown_pool
            logger.warning(f"  Unknown samples ({len(unknown_pool)}) less than target ({target_unknown}), using all available")
        
        result = pd.concat([sampled_ud, sampled_unknown])
        
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
        group_name, first_idx, second_idx, third_idx = self.parse_trace_id(trace_name)
        if group_name is None:
            return None
        
        try:
            # Access bucket groups under 'data' group
            data_group = hdf5_file['data']
            if group_name in data_group:
                group = data_group[group_name]
                waveform = group[first_idx, :second_idx, :third_idx]
                
                # Extract specified component
                comp_idx = self._component_index
                if comp_idx < waveform.shape[0]:
                    return waveform[comp_idx, :]
                else:
                    logger.warning(f"Component index {comp_idx} out of range for {trace_name}")
                    return None
            else:
                logger.warning(f"Group {group_name} not found in HDF5 data group")
                return None
        except Exception as e:
            logger.warning(f"Failed to extract waveform for {trace_name}: {e}")
            return None
    
    def process(self):
        """
        Process raw TXED data and save to SeisPolarity format.
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
            trace_name = row['trace_name']
            polarity = row['trace_polarity']
            p_arrival = int(row['trace_p_arrival_sample'])
            
            # Map polarity label
            if polarity in self._label_map:
                label = self._label_map[polarity]
            else:
                logger.warning(f"Unknown polarity: {polarity}")
                continue
            
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
        label_names = ['U', 'D', 'unknown']
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
        
        self.output_hdf5.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(self.output_hdf5, 'w') as f:
            f.create_dataset('X', data=X, compression='gzip')
            f.create_dataset('Y', data=Y, compression='gzip')
            f.create_dataset('p_pick', data=p_pick, compression='gzip')
            
            # Save metadata
            f.attrs['waveform_length'] = self.waveform_length
            f.attrs['sampling_rate'] = self.sampling_rate
            f.attrs['component'] = self.component
            f.attrs['num_samples'] = len(X)
            f.attrs['sampling_strategy'] = 'notebook_unknown_2x_UD'
            f.attrs['label_map'] = str(self._label_map)
        
        logger.info(f"Processed data saved successfully!")
        logger.info(f"  CSV: {self.output_csv}")
        logger.info(f"  HDF5: {self.output_hdf5}")
        logger.info("You can now use this file with WaveformDataset:")
        logger.info(f"  dataset = WaveformDataset(path='{self.output_hdf5}')")
