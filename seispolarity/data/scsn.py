"""
SCSN Dataset Processor for SeisPolarity

This module provides download functionality for SCSN (Southern California Seismic Network)
polarity dataset. Unlike other datasets, SCSN data is pre-processed and ready to use,
so this module only handles downloading.

Data Format:
- HDF5: Contains waveform data with polarity labels (0=Down, 1=Up, 2=Unknown)

Usage:
    processor = SCSNData(
        output_dir='/path/to/datasets/SCSN/'
    )
    processor.download()

Author: SeisPolarity
"""

import logging
from pathlib import Path
from typing import Optional

from .download import fetch_dataset_from_remote

logger = logging.getLogger(__name__)


class SCSNData:
    """
    SCSN seismic polarity data downloader.
    
    SCSN data is pre-processed and ready to use, so this class only handles
    downloading the dataset from ModelScope or Hugging Face.
    
    The downloaded HDF5 file contains:
    - Waveform data (n_samples, waveform_length)
    - Polarity labels (0=Down, 1=Up, 2=Unknown)
    - P-wave arrival sample indices
    
    Example:
        processor = SCSNData(
            output_dir='/path/to/datasets/SCSN/'
        )
        processor.download()
    """
    
    def __init__(
        self,
        output_dir: str,
        dataset_name: str = "SCSN",
        repo_path: Optional[str] = None,
        use_hf: bool = False,
        force_download: bool = False
    ):
        """
        Initialize SCSN downloader.
        
        Args:
            output_dir: Directory to save downloaded dataset
            dataset_name: Dataset name for logging (default: "SCSN")
            repo_path: Repository path for download. If None, uses default SCSN path
            use_hf: If True, use Hugging Face instead of ModelScope (default: False)
            force_download: If True, force re-download even if files exist (default: False)
        """
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.repo_path = repo_path
        self.use_hf = use_hf
        self.force_download = force_download
        
        # Default repo path if not specified
        if self.repo_path is None:
            self.repo_path = "SCSN/SCSN_P_2000_2017_6SEC_0.5R_FM_TRAIN.hdf5"
        
        # Label mapping for SCSN
        self._label_map = {
            0: 'Down',
            1: 'Up',
            2: 'Unknown'
        }
        
        logger.info("SCSN Downloader initialized:")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Dataset name: {self.dataset_name}")
        logger.info(f"  Repo path: {self.repo_path}")
        logger.info(f"  Use HF: {use_hf}")
        logger.info(f"  Force download: {force_download}")
    
    def download(self) -> Path:
        """
        Download SCSN dataset from remote repository.
        
        This method downloads the pre-processed SCSN dataset from ModelScope
        (default) or Hugging Face. The dataset is ready to use directly with
        WaveformDataset for training or inference.
        
        Returns:
            Path to downloaded HDF5 file
            
        Raises:
            Exception: If download fails
        """
        logger.info(f"Downloading {self.dataset_name} dataset...")
        
        try:
            downloaded_path = fetch_dataset_from_remote(
                dataset_name=self.dataset_name,
                repo_path=self.repo_path,
                cache_dir=str(self.output_dir),
                use_hf=self.use_hf,
                force_download=self.force_download
            )
            
            downloaded_path = Path(downloaded_path)
            
            if not downloaded_path.exists():
                raise FileNotFoundError(f"Downloaded file not found: {downloaded_path}")
            
            logger.info(f"{self.dataset_name} dataset downloaded successfully!")
            logger.info(f"  File path: {downloaded_path}")
            logger.info(f"  File size: {downloaded_path.stat().st_size / (1024*1024):.2f} MB")
            
            logger.info("You can now use this dataset with WaveformDataset:")
            logger.info(f"  dataset = WaveformDataset(path='{downloaded_path}')")
            logger.info(f"  dataset = WaveformDataset(path='{downloaded_path}', label_map={self._label_map})")
            
            return downloaded_path
            
        except Exception as e:
            logger.error(f"Failed to download {self.dataset_name} dataset: {e}")
            raise


def download_scsn(
    output_dir: str,
    use_hf: bool = False,
    force_download: bool = False
) -> Path:
    """
    Convenience function to download SCSN dataset.
    
    Args:
        output_dir: Directory to save downloaded dataset
        use_hf: If True, use Hugging Face instead of ModelScope (default: False)
        force_download: If True, force re-download even if files exist (default: False)
        
    Returns:
        Path to downloaded HDF5 file
        
    Example:
        from seispolarity.data.scsn import download_scsn
        
        dataset_path = download_scsn(output_dir='/path/to/datasets/SCSN/')
    """
    processor = SCSNData(
        output_dir=output_dir,
        use_hf=use_hf,
        force_download=force_download
    )
    return processor.download()


if __name__ == "__main__":
    # Example usage
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Default output directory
    default_output = "/home/yuan/code/SeisPolarity/datasets/SCSN"
    
    # Parse command line arguments
    use_hf = "--use-hf" in sys.argv
    force = "--force" in sys.argv
    output_dir = sys.argv[1] if len(sys.argv) > 1 else default_output
    
    logger.info("Starting SCSN download...")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Use Hugging Face: {use_hf}")
    logger.info(f"  Force download: {force}")
    
    try:
        path = download_scsn(
            output_dir=output_dir,
            use_hf=use_hf,
            force_download=force
        )
        logger.info(f"Download completed successfully: {path}")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)
