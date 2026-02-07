"""
DiTing Dataset Processor for SeisPolarity

This module processes raw DiTing seismic data and converts it to the format
required by SeisPolarity for training and inference.

DiTing Data Format:
- CSV: DiTing_all_processed_merged.csv (single file with all metadata)
  - Contains columns: p_motion_processed (polarity), p_clarity_processed (clarity)
- HDF5: DiTing330km_part_0.hdf5, DiTing330km_part_1.hdf5, ... (multiple files)
  - Each HDF5 file contains 'earthquake' group
  - Waveforms are stored in earthquake/{key_formatted} format

Key Features:
- Supports multiple HDF5 part files
- Supports resampling (e.g., 50Hz -> 100Hz)
- Parallel processing for efficiency
- Supports both polarity and clarity labels

Author: SeisPolarity
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

from .download import fetch_dataset_folder

logger = logging.getLogger(__name__)


class DiTing:
    """
    DiTing seismic polarity data processor.
    
    Converts raw DiTing data to SeisPolarity format:
    - X: Waveform data (n_samples, waveform_length)
    - Y: Polarity labels (0=U/up, 1=D/down, 2=unknown/X)
    - Z: Clarity labels (I/E/K as string)
    - p_pick: P-wave arrival sample indices (relative position)
    
    DiTing dataset structure:
    - data_dir/DiTing_all_processed_merged.csv (single file with all metadata)
    - data_dir/DiTing330km_part_0.hdf5, DiTing330km_part_1.hdf5, ... (multiple files)
    
    DiTing has two label types:
    - p_motion: Polarity (U=up, D=down, K/unknown)
    - p_clarity: Clarity (I=impulsive, E=emergent, K=uncertain/-)
    
    Label sampling strategy:
    1. If unknown_count > 2 * (up_count + down_count):
       - Select all up and down samples
       - Select unknown = 2 * (up_count + down_count)
    2. If unknown_count < 2 * (up_count + down_count):
       - Find minimum count m among up, down, unknown
       - Select m samples from each category
    
    Example:
        processor = DiTing(
            data_dir='/path/to/DiTing/',
            output_polarity='/path/to/output/'
        )
        processor.process()
    """
    
    def __init__(
        self,
        data_dir: str,
        output_polarity: str,
        component: str = "Z",
        sampling_rate: int = 100,
        original_sampling_rate: int = 50,
        auto_download: bool = False,
        use_hf: bool = False,
        force_download: bool = False
    ):
        """
        Initialize DiTing processor.
        
        Args:
            data_dir: Directory containing DiTing data (with DiTing_all_processed_merged.csv and DiTing330km_part_X.hdf5 files)
            output_polarity: Directory to save processed files (DiTing_polarity.csv and DiTing_polarity.hdf5)
            component: Which component to extract (default: "Z" for vertical)
            sampling_rate: Target sampling rate in Hz (default: 100)
            original_sampling_rate: Original sampling rate in Hz (default: 50)
            auto_download: If True, automatically download data if data_dir not found (default: False)
            use_hf: If True, use Hugging Face instead of ModelScope for download (default: False)
            force_download: If True, force re-download even if files exist (default: False)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_polarity)
        self.output_csv = self.output_dir / 'DiTing_polarity.csv'
        self.output_hdf5 = self.output_dir / 'DiTing_polarity.hdf5'
        self.waveform_length = 3000  # DiTing fixed waveform length
        self.component = component.upper()
        self.sampling_rate = sampling_rate
        self.original_sampling_rate = original_sampling_rate
        self.auto_download = auto_download
        self.use_hf = use_hf
        self.force_download = force_download
        
        # Auto-download if enabled and data_dir doesn't exist
        if auto_download:
            self._auto_download_data()
        
        # Calculate resampling ratio
        self.sr_ratio = self.sampling_rate / self.original_sampling_rate
        
        # Polarity label mapping
        self._label_map = {
            'U': 0,  # Up
            'D': 1,  # Down
            'X': 2,  # Unknown
            'K': 2,  # Unknown (alternate)
            'unknown': 2,
            'Unknown': 2
        }
        
        # Clarity label mapping (keep as strings)
        self._clarity_map = {
            'I': 'I',  # Impulsive
            'E': 'E',  # Emergent
            'K': 'K',  # Uncertain
            '-': 'K',  # Uncertain (alternate)
            'unknown': 'K'
        }
        
        # Find CSV file
        self.csv_path = self._find_csv_file()
        
        # Find HDF5 base path
        self.hdf5_base = self._find_hdf5_base()
        
        logger.info("DiTing Processor initialized:")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  CSV file: {self.csv_path}")
        logger.info(f"  HDF5 base: {self.hdf5_base}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Output CSV: {self.output_csv}")
        logger.info(f"  Output HDF5: {self.output_hdf5}")
        logger.info(f"  Waveform length: {self.waveform_length} (fixed)")
        logger.info(f"  Component: {self.component}")
        logger.info(f"  Sampling rate: {original_sampling_rate} -> {sampling_rate} Hz")
        logger.info(f"  Auto download: {auto_download}")
        logger.info(f"  Use HF: {use_hf}")
    
    def _auto_download_data(self):
        """
        Automatically download DiTing dataset if data_dir not found.
        
        This method uses fetch_dataset_folder to download the DiTing dataset
        from ModelScope or Hugging Face to the specified output directory.
        """
        data_dir_exists = self.data_dir.exists()
        
        if not self.force_download and data_dir_exists:
            csv_found = any(self.data_dir.glob('*.csv'))
            hdf5_found = any(self.data_dir.glob('*.hdf5'))
            
            if csv_found and hdf5_found:
                logger.info("Data directory exists with CSV and HDF5 files, skipping download")
                return
        
        if self.force_download:
            logger.info("Force download enabled, proceeding with download")
        elif not data_dir_exists:
            logger.warning(f"Data directory not found: {self.data_dir}")
        else:
            logger.warning("Data directory exists but missing required files")
        
        try:
            logger.info(f"Downloading DiTing dataset to: {self.data_dir}")
            downloaded_path = fetch_dataset_folder(
                dataset_name="DiTing",
                folder_path=str(self.data_dir),
                cache_dir=str(self.data_dir),
                use_hf=self.use_hf,
                force_download=self.force_download
            )
            logger.info(f"DiTing dataset downloaded successfully to: {downloaded_path}")
            
            # Update data_dir based on downloaded location
            downloaded_path = Path(downloaded_path)
            if downloaded_path.exists():
                self.data_dir = downloaded_path
                logger.info(f"Updated data directory: {self.data_dir}")
            else:
                logger.warning(f"Downloaded directory not found: {downloaded_path}")
                
        except Exception as e:
            logger.error(f"Failed to download DiTing dataset: {e}")
            raise
    
    def _find_csv_file(self) -> Path:
        """Find CSV file(s) in data directory.
        
        DiTing has two formats:
        1. Single merged CSV: DiTing_all_processed_merged.csv
        2. Multiple part CSVs: DiTing330km_part_0.csv, DiTing330km_part_1.csv, ...
        """
        # Try to find single merged CSV first
        single_csv = self.data_dir / 'DiTing_all_processed_merged.csv'
        if single_csv.exists():
            return single_csv
        
        # Try common patterns for single CSV
        patterns = ['DiTing*.csv']
        for pattern in patterns:
            matches = list(self.data_dir.glob(pattern))
            # Filter out part CSVs if looking for single CSV
            non_part_csvs = [m for m in matches if '_part_' not in m.name]
            if non_part_csvs:
                csv_file = non_part_csvs[0]
                if len(non_part_csvs) > 1:
                    logger.warning(f"Multiple CSV files found, using: {csv_file}")
                return csv_file
        
        # If no single CSV found, check for part CSVs
        part_csvs = list(self.data_dir.glob('*_part_*.csv'))
        if part_csvs:
            logger.info(f"Found {len(part_csvs)} part CSV files")
            # Return the first part CSV to indicate multi-part format
            # The processor will handle merging all parts
            return part_csvs[0]
        
        raise FileNotFoundError(f"No CSV file found in {self.data_dir}")
    
    def _find_hdf5_base(self) -> Path:
        """Find HDF5 base path by looking for pattern DiTing330km_part_X.hdf5."""
        # Try to find HDF5 files with pattern
        hdf5_files = list(self.data_dir.glob('DiTing*_part_*.hdf5'))
        
        if not hdf5_files:
            # Try alternative pattern
            hdf5_files = list(self.data_dir.glob('*_part_*.hdf5'))
        
        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 part files found in {self.data_dir}")
        
        # Extract base name from first file (e.g., DiTing330km from DiTing330km_part_0.hdf5)
        first_file = hdf5_files[0].name
        base_name = first_file.split('_part_')[0]
        base_path = self.data_dir / base_name
        
        logger.info(f"Found {len(hdf5_files)} HDF5 part files")
        return base_path
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load and validate DiTing CSV metadata.
        
        Supports two formats:
        1. Single merged CSV file
        2. Multiple part CSV files (DiTing330km_part_0.csv, part_1.csv, ...)
        
        Returns:
            DataFrame with valid samples
        """
        # Check if this is multi-part format
        if '_part_' in self.csv_path.name:
            # Load and merge all part CSVs
            part_csvs = sorted(self.data_dir.glob('*_part_*.csv'))
            logger.info(f"Loading {len(part_csvs)} part CSV files...")
            
            df_list = []
            for csv_file in part_csvs:
                logger.info(f"  Loading: {csv_file.name}")
                df_part = pd.read_csv(csv_file)
                # Add part column if not present
                if 'part' not in df_part.columns:
                    part_num = int(csv_file.name.split('_part_')[1].replace('.csv', ''))
                    df_part['part'] = part_num
                df_list.append(df_part)
            
            df = pd.concat(df_list, ignore_index=True)
            logger.info(f"  Total rows after merge: {len(df)}")
        else:
            # Single CSV file
            if not self.csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
            logger.info(f"Loading DiTing CSV: {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            logger.info(f"  Total rows: {len(df)}")
        
        logger.info(f"  Columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['part', 'p_pick']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Check for key column (key or key_formatted)
        if 'key' in df.columns:
            self.key_col = 'key'
        elif 'key_formatted' in df.columns:
            self.key_col = 'key_formatted'
        else:
            raise ValueError("Missing required column: 'key' or 'key_formatted'")
        
        # Check for polarity column
        polarity_col = None
        for col in df.columns:
            if 'p_motion' in col.lower() or 'polar' in col.lower():
                polarity_col = col
                break
        
        if polarity_col is None:
            raise ValueError("Could not find polarity column (should contain 'p_motion' or 'polar')")
        
        # Check for clarity column
        clarity_col = None
        for col in df.columns:
            if 'p_clarity' in col.lower():
                clarity_col = col
                break
        
        if clarity_col is None:
            raise ValueError("Could not find clarity column (should contain 'p_clarity')")
        
        self.polarity_col = polarity_col
        self.clarity_col = clarity_col
        logger.info(f"  Polarity column: {polarity_col}")
        logger.info(f"  Clarity column: {clarity_col}")
        
        # Log polarity distribution
        polarity_counts = df[polarity_col].value_counts()
        logger.info("  Polarity distribution:")
        for pol, count in polarity_counts.items():
            logger.info(f"    {pol}: {count} ({count/len(df)*100:.2f}%)")
        
        # Log clarity distribution
        clarity_counts = df[clarity_col].value_counts()
        logger.info("  Clarity distribution:")
        for clr, count in clarity_counts.items():
            logger.info(f"    {clr}: {count} ({count/len(df)*100:.2f}%)")
        
        # Log part distribution
        part_counts = df['part'].value_counts().sort_index()
        logger.info(f"  Number of parts: {len(part_counts)}")
        
        return df
    
    def sample_by_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply smart sampling strategy to balance labels.
        
        Args:
            df: Input DataFrame with polarity labels
            
        Returns:
            Sampled DataFrame
        """
        df = df.copy()
        
        # Map labels to U/D/X
        df['_label_category'] = df[self.polarity_col].apply(
            lambda x: 'U' if str(x).upper() == 'U'
                     else 'D' if str(x).upper() == 'D'
                     else 'X'
        )
        
        counts = df['_label_category'].value_counts()
        u_count = counts.get('U', 0)
        d_count = counts.get('D', 0)
        x_count = counts.get('X', 0)
        
        logger.info(f"Label counts - U: {u_count}, D: {d_count}, X: {x_count}")
        
        total_ud = u_count + d_count
        
        # Strategy 1: If X > 2 * (U + D)
        if x_count > 2 * total_ud:
            logger.info("Using Strategy 1: X > 2*(U+D)")
            target_x = 2 * total_ud
            
            # Select all U and D
            sampled_ud = df[df['_label_category'].isin(['U', 'D'])]
            
            # Select target number of X samples
            x_pool = df[df['_label_category'] == 'X']
            sampled_x = x_pool.sample(min(target_x, len(x_pool)), random_state=42)
            
            result = pd.concat([sampled_ud, sampled_x])
            
        # Strategy 2: If X < 2 * (U + D)
        else:
            logger.info("Using Strategy 2: X < 2*(U+D)")
            m = min(u_count, d_count, x_count)
            
            logger.info(f"Minimum count m = {m}")
            
            # Sample m from each category
            result_list = []
            for label in ['U', 'D', 'X']:
                pool = df[df['_label_category'] == label]
                sampled = pool.sample(min(m, len(pool)), random_state=42)
                result_list.append(sampled)
            
            result = pd.concat(result_list)
        
        logger.info(f"Sampled {len(result)} samples")
        return result.reset_index(drop=True)
    
    def _process_partition(self, part_id: int, df_part: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Process a single partition (HDF5 file).
        
        Args:
            part_id: Partition ID (used to construct HDF5 filename)
            df_part: DataFrame subset for this partition
            
        Returns:
            Tuple of (waveforms, labels, clarities, p_picks, stats)
        """
        # Construct HDF5 file path
        hdf5_path = self.hdf5_base.parent / f"{self.hdf5_base.name}_part_{int(part_id)}.hdf5"
        
        if not hdf5_path.exists():
            logger.warning(f"HDF5 file not found: {hdf5_path}")
            return None, None, None, None, {}
        
        waveforms = []
        labels = []
        clarities = []
        p_picks = []
        stats = {'U': 0, 'D': 0, 'X': 0}
        
        half_window = self.waveform_length // 2
        
        try:
            with h5py.File(hdf5_path, 'r') as f:
                if 'earthquake' not in f:
                    logger.warning(f"No 'earthquake' group in {hdf5_path}")
                    return None, None, None, None, {}
                
                eq_group = f['earthquake']
                
                for _, row in df_part.iterrows():
                    # Get key formatted
                    key_raw = row[self.key_col]
                    
                    # Format HDF5 key: "000001.0001" format
                    try:
                        k_float = float(key_raw)
                        hdf5_key = f"{int(k_float):06d}.{str(k_float).split('.')[1].ljust(4,'0')[:4]}"
                    except (ValueError, TypeError, AttributeError):
                        hdf5_key = str(key_raw)
                    
                    if hdf5_key not in eq_group:
                        continue
                    
                    # Extract waveform
                    data = eq_group[hdf5_key][:]
                    
                    # Handle component selection
                    if len(data.shape) == 2:
                        # Shape is (n_samples, n_components)
                        comp_map = {'Z': 0, 'N': 1, 'E': 2}
                        comp_idx = comp_map.get(self.component, 0)
                        if comp_idx < data.shape[1]:
                            waveform = data[:, comp_idx]
                        else:
                            waveform = data[:, 0]
                    else:
                        waveform = data
                    
                    # Resample if needed
                    if self.sr_ratio != 1.0:
                        target_len = int(len(waveform) * self.sr_ratio)
                        waveform = signal.resample(waveform, target_len)
                    
                    # Get p_pick and calculate window
                    p_pick_float = float(row['p_pick']) * self.sr_ratio
                    center_idx = int(p_pick_float)
                    start_idx = center_idx - half_window
                    end_idx = center_idx + half_window
                    
                    # Calculate relative p_pick
                    relative_p_pick = p_pick_float - start_idx
                    
                    # Extract window with padding
                    if start_idx < 0:
                        pad_left = abs(start_idx)
                        window = np.pad(waveform[0:end_idx], (pad_left, 0), 'constant')
                    elif end_idx > len(waveform):
                        pad_right = end_idx - len(waveform)
                        window = np.pad(waveform[start_idx:], (0, pad_right), 'constant')
                    else:
                        window = waveform[start_idx:end_idx]
                    
                    # Ensure exact length
                    if len(window) < self.waveform_length:
                        window = np.pad(window, (0, self.waveform_length - len(window)), 'constant')
                    elif len(window) > self.waveform_length:
                        window = window[:self.waveform_length]
                    
                    # Get polarity label
                    polarity = str(row[self.polarity_col]).upper()
                    if polarity == 'U':
                        label = 0
                        stats['U'] += 1
                    elif polarity == 'D':
                        label = 1
                        stats['D'] += 1
                    else:
                        label = 2
                        stats['X'] += 1
                    
                    # Get clarity label
                    clarity_raw = row.get(self.clarity_col, 'K')
                    clarity_str = str(clarity_raw).strip()
                    if clarity_str == '' or clarity_str == '-':
                        clarity_str = 'K'
                    clarity_str = clarity_str.upper()
                    if clarity_str == 'K':
                        clarity_str = 'K'
                    elif clarity_str == 'I':
                        clarity_str = 'I'
                    elif clarity_str == 'E':
                        clarity_str = 'E'
                    else:
                        clarity_str = 'K'  # Default to unknown
                    
                    # Add sample
                    waveforms.append(window)
                    labels.append(label)
                    clarities.append(clarity_str.encode('ascii'))
                    p_picks.append(relative_p_pick)
        
        except Exception as e:
            logger.error(f"Error processing partition {part_id}: {e}")
            return None, None, None, None, {}
        
        if len(waveforms) == 0:
            return None, None, None, None, {}
        
        return (
            np.array(waveforms, dtype=np.float32),
            np.array(labels, dtype=np.int32),
            np.array(clarities, dtype='S1'),
            np.array(p_picks, dtype=np.float32),
            stats
        )
    
    def process(self, use_multiprocessing: bool = True, max_workers: int = None):
        """
        Process raw DiTing data and save to SeisPolarity format.
        
        Args:
            use_multiprocessing: Whether to use parallel processing (default: True)
            max_workers: Maximum number of workers (default: CPU count)
        """
        # Load metadata
        df = self.load_metadata()
        
        # Apply sampling strategy
        df = self.sample_by_strategy(df)
        
        # Group by part
        grouped = df.groupby('part')
        tasks = list(grouped)
        logger.info(f"Processing {len(tasks)} HDF5 partitions...")
        
        all_waveforms = []
        all_labels = []
        all_clarities = []
        all_p_picks = []
        global_stats = {'U': 0, 'D': 0, 'X': 0}
        
        if use_multiprocessing:
            # Parallel processing
            if max_workers is None:
                max_workers = min(os.cpu_count(), 16)
            
            logger.info(f"Using {max_workers} parallel workers...")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._process_partition, part_id, df_part): part_id
                          for part_id, df_part in tasks}
                
                for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
                    part_id = futures[future]
                    try:
                        waveforms, labels, clarities, p_picks, stats = future.result()
                        if waveforms is not None and len(waveforms) > 0:
                            all_waveforms.append(waveforms)
                            all_labels.append(labels)
                            all_clarities.append(clarities)
                            all_p_picks.append(p_picks)
                            for k in global_stats:
                                global_stats[k] += stats.get(k, 0)
                    except Exception as e:
                        logger.error(f"Error in partition {part_id}: {e}")
        else:
            # Sequential processing
            for part_id, df_part in tqdm(tasks, desc="Processing"):
                waveforms, labels, clarities, p_picks, stats = self._process_partition(part_id, df_part)
                if waveforms is not None and len(waveforms) > 0:
                    all_waveforms.append(waveforms)
                    all_labels.append(labels)
                    all_clarities.append(clarities)
                    all_p_picks.append(p_picks)
                    for k in global_stats:
                        global_stats[k] += stats.get(k, 0)
        
        # Concatenate all results
        if len(all_waveforms) == 0:
            raise ValueError("No waveforms were processed successfully")
        
        X = np.concatenate(all_waveforms, axis=0)
        Y = np.concatenate(all_labels, axis=0)
        Z = np.concatenate(all_clarities, axis=0)
        p_pick = np.concatenate(all_p_picks, axis=0)
        
        logger.info(f"Successfully processed {len(X)} samples")
        logger.info(f"Global stats - U: {global_stats['U']}, D: {global_stats['D']}, X: {global_stats['X']}")
        
        # Log final label distribution
        unique, counts = np.unique(Y, return_counts=True)
        label_names = ['U', 'D', 'X']
        logger.info("Final label distribution:")
        for u, c in zip(unique, counts):
            logger.info(f"  {label_names[u] if u < len(label_names) else str(u)}: {c}")
        
        # Log clarity distribution
        unique_clarity, counts_clarity = np.unique(Z, return_counts=True)
        logger.info("Final clarity distribution:")
        for u, c in zip(unique_clarity, counts_clarity):
            clarity_name = chr(Z[u]) if u < len(Z) else str(Z[u])
            logger.info(f"  {clarity_name}: {c}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV output (selected samples metadata)
        logger.info(f"Saving CSV output to: {self.output_csv}")
        df.to_csv(self.output_csv, index=False)
        
        # Save HDF5 output
        logger.info(f"Saving HDF5 output to: {self.output_hdf5}")
        
        with h5py.File(self.output_hdf5, 'w') as f:
            f.create_dataset('X', data=X, compression='gzip')
            f.create_dataset('Y', data=Y, compression='gzip')
            f.create_dataset('Z', data=Z, compression='gzip')  # Clarity labels
            f.create_dataset('p_pick', data=p_pick, compression='gzip')
            
            # Save metadata
            f.attrs['waveform_length'] = self.waveform_length
            f.attrs['sampling_rate'] = self.sampling_rate
            f.attrs['original_sampling_rate'] = self.original_sampling_rate
            f.attrs['component'] = self.component
            f.attrs['num_samples'] = len(X)
            f.attrs['sampling_strategy'] = 'smart'
            f.attrs['label_map'] = str(self._label_map)
            f.attrs['clarity_map'] = str(self._clarity_map)
        
        logger.info("Processed data saved successfully!")
        logger.info(f"  CSV: {self.output_csv}")
        logger.info(f"  HDF5: {self.output_hdf5}")
        logger.info("You can now use this file with WaveformDataset:")
        logger.info(f"  dataset = WaveformDataset(path='{self.output_hdf5}')")
