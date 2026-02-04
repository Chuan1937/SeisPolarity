"""
Unified Inference Interface for SeisPolarity models.
"""

import os
import sys
import torch
import numpy as np
import warnings
import httpx
import requests
from typing import Union, List, Optional
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add project root to sys.path

from seispolarity.models import SCSN, EQPolarityCCT, DitingMotion, PPNet, PolarCAP, CFM, RPNet

# Constants
HF_REPO = "HeXingChen/SeisPolarity-Model"
MODELSCOPE_REPO = "chuanjun/HeXingChen"

MODELS_CONFIG = {
    "ROSS_SCSN": {
        "filename": "ROSS_SCSN.pth",
        "filename_hf": "ROSS/ROSS_SCSN.pth",
        "filename_ms": "ROSS/ROSS_SCSN.pth",
        "model_class": SCSN,
        "input_len": 400,
        "num_classes": 3,
        "class_map": {0: "Up", 1: "Down", 2: "Unknown"},
        "output_index": None,
        "description": "ROSS model: Ross, Z. E., Meier, M. & Hauksson, E. P Wave Arrival Picking and First-Motion Polarity Determination With Deep Learning. JGR Solid Earth 123, 5120-5129 (2018) [13]. Trained on SCSN dataset. 3-class classification (Up, Down, Unknown). Input length: 400 samples. Model weights converted and maintained by He XingChen (Chinese, Han ethnicity). GitHub: https://github.com/Chuan1937"
    },
    "ROSS_GLOBAL": {
        "filename": "ROSS_GLOBAL.pth",
        "filename_hf": "ROSS/ROSS_GLOBAL.pth",
        "filename_ms": "ROSS/ROSS_GLOBAL.pth",
        "model_class": SCSN,
        "input_len": 400,
        "num_classes": 3,
        "class_map": {0: "Up", 1: "Down", 2: "Unknown"},
        "output_index": None,
        "description": "ROSS model: Ross, Z. E., Meier, M. & Hauksson, E. P Wave Arrival Picking and First-Motion Polarity Determination With Deep Learning. JGR Solid Earth 123, 5120-5129 (2018) [13]. Trained on global datasets: SCSN [13], INSTANCE: Michelini, A. et al. INSTANCE-the Italian seismic dataset for machine learning. Earth System Science Data 13, 5509-5544 (2021) [12], PNW: Ni, Y. et al. Curated Pacific Northwest AI-ready seismic dataset. https://eartharxiv.org/repository/view/5049/ (2023) [6], TXED: Chen, Y. et al. TXED: The Texas Earthquake Dataset for AI. Seismological Research Letters 95, 2013-2022 (2024) [21], DiTing. 3-class classification (Up, Down, Unknown). Input length: 400 samples. Model weights converted and maintained by He XingChen (Chinese, Han ethnicity). GitHub: https://github.com/Chuan1937"
    },
    "EQPOLARITY_SCSN": {
        "filename": "EQPOLARITY_SCSN.pth",
        "filename_hf": "EQPOLARITY/EQPOLARITY_SCSN.pth",
        "filename_ms": "EQPOLARITY/EQPOLARITY_SCSN.pth",
        "model_class": EQPolarityCCT,
        "input_len": 600,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": None,
        "description": "EQPolarity CCT model: Chen, Y. et al. Deep learning for P-wave first-motion polarity determination and its application to focal mechanism inversion. IEEE Transactions on Geoscience and Remote Sensing (2024) [7]. Trained on SCSN dataset: Ross, Z. E., Meier, M. & Hauksson, E. P Wave Arrival Picking and First-Motion Polarity Determination With Deep Learning. JGR Solid Earth 123, 5120-5129 (2018) [13]. 2-class classification (Up, Down). Input length: 600 samples. Model weights converted and maintained by He XingChen (Chinese, Han ethnicity). GitHub: https://github.com/Chuan1937"
    },
    "EQPOLARITY_TXED": {
        "filename": "EQPOLARITY_TXED.pth",
        "filename_hf": "EQPOLARITY/EQPOLARITY_TXED.pth",
        "filename_ms": "EQPOLARITY/EQPOLARITY_TXED.pth",
        "model_class": EQPolarityCCT,
        "input_len": 600,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": None,
        "description": "EQPolarity CCT model: Chen, Y. et al. Deep learning for P-wave first-motion polarity determination and its application to focal mechanism inversion. IEEE Transactions on Geoscience and Remote Sensing (2024) [7]. Trained on TXED dataset: Chen, Y. et al. TXED: The Texas Earthquake Dataset for AI. Seismological Research Letters 95, 2013-2022 (2024) [21]. 2-class classification (Up, Down). Input length: 600 samples. Model weights converted and maintained by He XingChen (Chinese, Han ethnicity). GitHub: https://github.com/Chuan1937"
    },
    "EQPOLARITY_GLOBAL": {
        "filename": "EQPOLARITY_GLOBAL.pth",
        "filename_hf": "EQPOLARITY/EQPOLARITY_GLOBAL.pth",
        "filename_ms": "EQPOLARITY/EQPOLARITY_GLOBAL.pth",
        "model_class": EQPolarityCCT,
        "input_len": 600,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": None,
        "description": "EQPolarity CCT model: Chen, Y. et al. Deep learning for P-wave first-motion polarity determination and its application to focal mechanism inversion. IEEE Transactions on Geoscience and Remote Sensing (2024) [7]. Trained on global datasets: SCSN [13], INSTANCE [12], PNW [6], TXED [21], DiTing. 2-class classification (Up, Down). Input length: 600 samples. Model weights converted and maintained by He XingChen (Chinese, Han ethnicity). GitHub: https://github.com/Chuan1937"
    },
    "DITINGMOTION_DITINGSCSN": {
        "filename": "DITINGMOTION_DITINGSCSN.pth",
        "filename_hf": "DITINGMOTION/DITINGMOTION_DITINGSCSN.pth",
        "filename_ms": "DITINGMOTION/DITINGMOTION_DITINGSCSN.pth",
        "model_class": DitingMotion,
        "input_len": 128,
        "num_classes": 3,
        "class_map": {0: "Up", 1: "Down", 2: "Unknown"},
        "output_index": 3,
        "description": "DiTingMotion model: Zhao, M. et al. DiTingMotion: A deep-learning first-motion-polarity classifier and its application to focal mechanism inversion. Frontiers in Earth Science 11, 1103914 (2023) [9]. Trained on DiTing dataset. Multi-output model for polarity and clarity prediction. 3-class classification (Up, Down, Unknown). Input length: 128 samples. Model weights converted and maintained by He XingChen (Chinese, Han ethnicity). GitHub: https://github.com/Chuan1937"
    },
    "DITING_GLOBAL": {
        "filename": "DITING_GLOBAL.pth",
        "filename_hf": "DITINGMOTION/DITING_GLOBAL.pth",
        "filename_ms": "DITINGMOTION/DITING_GLOBAL.pth",
        "model_class": DitingMotion,
        "input_len": 128,
        "num_classes": 3,
        "class_map": {0: "Up", 1: "Down", 2: "Unknown"},
        "output_index": 3,
        "description": "DiTingMotion model: Zhao, M. et al. DiTingMotion: A deep-learning first-motion-polarity classifier and its application to focal mechanism inversion. Frontiers in Earth Science 11, 1103914 (2023) [9]. Trained on global datasets: SCSN [13], INSTANCE [12], PNW [6], TXED [21], DiTing. Multi-output model for polarity and clarity prediction. 3-class classification (Up, Down, Unknown). Input length: 128 samples. Model weights converted and maintained by He XingChen (Chinese, Han ethnicity). GitHub: https://github.com/Chuan1937"
    },
    "APP_SCSN": {
        "filename": "APP_SCSN.pth",
        "filename_hf": "APP/APP_SCSN.pth",
        "filename_ms": "APP/APP_SCSN.pth",
        "model_class": PPNet,
        "input_len": 400,
        "num_classes": 3,
        "class_map": {0: "Up", 1: "Down", 2: "Unknown"},
        "output_index": 1,
        "description": "APP (PPNet) model: Song, J., Zhu, W., Zi, J., Yang, H. & Chu, R. An Enhanced Focal Mechanism Catalog of Induced Earthquakes in Weiyuan, Sichuan, from Dense Array Data and a Multitask Deep Learning Model. The Seismic Record 5, 175-184 (2025) [2]; Zhu, W., Tai, K. S., Mousavi, S. M., Bailis, P. & Beroza, G. C. An End-to-End Earthquake Detection Method for Joint Phase Picking and Association using Deep Learning. JGR Solid Earth 127, e2021JB023283 (2022) [1]. Trained on SCSN dataset: Ross, Z. E., Meier, M. & Hauksson, E. P Wave Arrival Picking and First-Motion Polarity Determination With Deep Learning. JGR Solid Earth 123, 5120-5129 (2018) [13]. U-Net + LSTM + attention architecture. Multi-output model. 3-class classification (Up, Down, Unknown). Input length: 400 samples. Model weights converted and maintained by He XingChen (Chinese, Han ethnicity). GitHub: https://github.com/Chuan1937"
    },
    "APP_GLOBAL": {
        "filename": "APP_GLOBAL.pth",
        "filename_hf": "APP/APP_GLOBAL.pth",
        "filename_ms": "APP/APP_GLOBAL.pth",
        "model_class": PPNet,
        "input_len": 400,
        "num_classes": 3,
        "class_map": {0: "Up", 1: "Down", 2: "Unknown"},
        "output_index": 1,
        "description": "APP (PPNet) model: Song, J., Zhu, W., Zi, J., Yang, H. & Chu, R. An Enhanced Focal Mechanism Catalog of Induced Earthquakes in Weiyuan, Sichuan, from Dense Array Data and a Multitask Deep Learning Model. The Seismic Record 5, 175-184 (2025) [2]; Zhu, W., Tai, K. S., Mousavi, S. M., Bailis, P. & Beroza, G. C. An End-to-End Earthquake Detection Method for Joint Phase Picking and Association using Deep Learning. JGR Solid Earth 127, e2021JB023283 (2022) [1]. Trained on global datasets: SCSN [13], INSTANCE [12], PNW [6], TXED [21], DiTing. U-Net + LSTM + attention architecture. Multi-output model. 3-class classification (Up, Down, Unknown). Input length: 400 samples. Model weights converted and maintained by He XingChen (Chinese, Han ethnicity). GitHub: https://github.com/Chuan1937"
    },
    "CFM_SCSN": {
        "filename": "CFM_SCSN.pth",
        "filename_hf": "CFM/CFM_SCSN.pth",
        "filename_ms": "CFM/CFM_SCSN.pth",
        "model_class": CFM,
        "input_len": 160,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": None,
        "description": "CFM model: Messuti, G. et al. CFM: a convolutional neural network for first-motion polarity classification of seismic records in volcanic and tectonic areas. Frontiers in Earth Science 11, 1223686 (2023) [5]. Trained on SCSN dataset: Ross, Z. E., Meier, M. & Hauksson, E. P Wave Arrival Picking and First-Motion Polarity Determination With Deep Learning. JGR Solid Earth 123, 5120-5129 (2018) [13]. Multi-layer Conv1D + dense heads architecture. 2-class classification (Up, Down). Input length: 160 samples. Model weights converted and maintained by He XingChen (Chinese, Han ethnicity). GitHub: https://github.com/Chuan1937"
    },
    "CFM_GLOBAL": {
        "filename": "CFM_GLOBAL.pth",
        "filename_hf": "CFM/CFM_GLOBAL.pth",
        "filename_ms": "CFM/CFM_GLOBAL.pth",
        "model_class": CFM,
        "input_len": 160,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": None,
        "description": "CFM model: Messuti, G. et al. CFM: a convolutional neural network for first-motion polarity classification of seismic records in volcanic and tectonic areas. Frontiers in Earth Science 11, 1223686 (2023) [5]. Trained on global datasets: SCSN [13], INSTANCE [12], PNW [6], TXED [21], DiTing. Multi-layer Conv1D + dense heads architecture. 2-class classification (Up, Down). Input length: 160 samples. Model weights converted and maintained by He XingChen (Chinese, Han ethnicity). GitHub: https://github.com/Chuan1937"
    },
    "POLARCAP_SCSN": {
        "filename": "POLARCAP_SCSN.pth",
        "filename_hf": "POLARCAP/POLARCAP_SCSN.pth",
        "filename_ms": "POLARCAP/POLARCAP_SCSN.pth",
        "model_class": PolarCAP,
        "input_len": 64,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": 1,
        "description": "PolarCAP model: Chakraborty, M. et al. PolarCAP-A deep learning approach for first motion polarity classification of earthquake waveforms. Artificial Intelligence in Geosciences 3, 46-52 (2022) [16]. Trained on SCSN dataset: Ross, Z. E., Meier, M. & Hauksson, E. P Wave Arrival Picking and First-Motion Polarity Determination With Deep Learning. JGR Solid Earth 123, 5120-5129 (2018) [13]. Encoder-decoder + classification head architecture. Multi-output model. 2-class classification (Up, Down). Input length: 64 samples. Model weights converted and maintained by He XingChen (Chinese, Han ethnicity). GitHub: https://github.com/Chuan1937"
    },
    "POLARCAP_GLOBAL": {
        "filename": "POLARCAP_GLOBAL.pth",
        "filename_hf": "POLARCAP/POLARCAP_GLOBAL.pth",
        "filename_ms": "POLARCAP/POLARCAP_GLOBAL.pth",
        "model_class": PolarCAP,
        "input_len": 64,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": 1,
        "description": "PolarCAP model: Chakraborty, M. et al. PolarCAP-A deep learning approach for first motion polarity classification of earthquake waveforms. Artificial Intelligence in Geosciences 3, 46-52 (2022) [16]. Trained on global datasets: SCSN [13], INSTANCE [12], PNW [6], TXED [21], DiTing. Encoder-decoder + classification head architecture. Multi-output model. 2-class classification (Up, Down). Input length: 64 samples. Model weights converted and maintained by He XingChen (Chinese, Han ethnicity). GitHub: https://github.com/Chuan1937"
    },
    "RPNET_SCSN": {
        "filename": "RPNET_SCSN.pth",
        "filename_hf": "RPNET/RPNET_SCSN.pth",
        "filename_ms": "RPNET/RPNET_SCSN.pth",
        "model_class": RPNet,
        "input_len": 400,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": None,
        "description": "RPNet model: Han, J., Kim, S. & Sheen, D.-H. RPNet: Robust P-Wave First-motion polarity determination using deep learning. Seismological Research Letters (2025) [17]. Trained on SCSN dataset: Ross, Z. E., Meier, M. & Hauksson, E. P Wave Arrival Picking and First-Motion Polarity Determination With Deep Learning. JGR Solid Earth 123, 5120-5129 (2018) [13]. ResNet-based architecture for polarity prediction. 2-class classification (Up, Down). Input length: 400 samples. Model weights converted and maintained by He XingChen (Chinese, Han ethnicity). GitHub: https://github.com/Chuan1937"
    },
    "RPNET_GLOBAL": {
        "filename": "RPNET_GLOBAL.pth",
        "filename_hf": "RPNET/RPNET_GLOBAL.pth",
        "filename_ms": "RPNET/RPNET_GLOBAL.pth",
        "model_class": RPNet,
        "input_len": 400,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": None,
        "description": "RPNet model: Han, J., Kim, S. & Sheen, D.-H. RPNet: Robust P-Wave First-motion polarity determination using deep learning. Seismological Research Letters (2025) [17]. Trained on global datasets: SCSN [13], INSTANCE [12], PNW [6], TXED [21], DiTing. ResNet-based architecture for polarity prediction. 2-class classification (Up, Down). Input length: 400 samples. Model weights converted and maintained by He XingChen (Chinese, Han ethnicity). GitHub: https://github.com/Chuan1937"
    }
}

class Predictor:
    """
    High-level interface for polarity prediction.
    
    Usage:
        >>> from seispolarity.inference import Predictor
        >>> Predictor.list_pretrained(details=True)
        >>> model = Predictor("ROSS_GLOBAL")
        >>> preds = model.predict(waveforms)
    
    Available Models:
        Use Predictor.list_pretrained(details=True) to see all available models and their descriptions.
    
    Note: Use the full model name (e.g., "ROSS_SCSN", "ROSS_GLOBAL") to initialize the predictor.
    """
    
    @staticmethod
    def list_pretrained(details: bool = True) -> dict:
        """
        List all available pretrained models.
        
        Args:
            details (bool): If True, return model descriptions. If False, return only model names.
        
        Returns:
            dict: Dictionary mapping model names to model configurations.
        
        Example:
            >>> from seispolarity.inference import Predictor
            >>> models = Predictor.list_pretrained(details=True)
            >>> for name, config in models.items():
            ...     print(f"{name}: {config['filename']}")
        """
        if details:
            print("Available pretrained models:\n")
            for name, config in MODELS_CONFIG.items():
                print(f"{name}:")
                print(f"  Filename: {config['filename']}")
                print(f"  Input Length: {config['input_len']}")
                print(f"  Classes: {config['num_classes']} ({', '.join([config['class_map'][i] for i in range(config['num_classes'])])})")
                print(f"  Description: {config['description']}")
                print("-" * 55)
                print()
        return MODELS_CONFIG
    
    def __init__(self, model_name: str = "ROSS_SCSN", device: Optional[str] = None, cache_dir: str = "./checkpoints_download", model_path: Optional[str] = None, force_ud: bool = False):
        """
        Initialize the predictor.

        Args:
            model_name (str): Name of the model to use (default: "ROSS_SCSN").
                             Must be one of the available model names (e.g., "ROSS_SCSN", "ROSS_GLOBAL", "APP_GLOBAL").
                             Must use full model names (e.g., "ROSS_SCSN", "ROSS_GLOBAL", "APP_GLOBAL").
            device (str, optional): "cuda" or "cpu". If None, auto-detect.
            cache_dir (str): Directory to store downloaded models (default: "./checkpoints_download").
            model_path (str, optional): Manually specified path to the model file. If provided, skips download.
            force_ud (bool): Force output to U/D only (no X). For DiTingMotion model, if True, when model predicts X,
                             select the one with higher probability between U and D as final prediction.

        Raises:
            ValueError: If model_name is not found in available models.
        """

        # model_name can be config key (e.g., "ross") or full filename (e.g., "ROSS_SCSN.pth")
        # Find matching config in MODELS_CONFIG
        self.model_filename = model_name
        found_config = None
        for name, config in MODELS_CONFIG.items():
            if model_name == name or model_name == config['filename'] or model_name == config['filename_hf'] or model_name == config['filename_ms'].split('/')[-1]:
                found_config = config
                self.config_key = name
                break

        if found_config is None:
            raise ValueError(f"Unknown model '{model_name}'. Available names: {list(MODELS_CONFIG.keys())}")

        print(f"Using model: {self.model_filename} ({self.config_key})")

        self.config = found_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.force_ud = force_ud
        print(f"Using device: {self.device}")
        if force_ud and "DITING" in self.config_key:
            print("Force U/D mode enabled: When prediction is X, select the one with higher probability between U and D")
        
        # 1. Download/Load Checkpoint
        self.checkpoint_path = self._resolve_model_path(cache_dir, self.config["filename"], model_path)
        
        # 2. Initialize Model
        # Use different initialization parameters based on model type
        if self.config_key in ["EQPOLARITY_SCSN", "EQPOLARITY_TXED", "EQPOLARITY_GLOBAL"]:
            # EQPolarityCCT requires input_length parameter
            self.model = self.config["model_class"](input_length=self.config["input_len"])
        elif self.config_key in ["DITINGMOTION_DITINGSCSN", "DITING_GLOBAL"]:
            # DitingMotion requires input_channels parameter
            self.model = self.config["model_class"](input_channels=2)
        elif self.config_key in ["APP_SCSN", "APP_GLOBAL"]:
            # PPNet requires input_len, input_channels, num_classes parameters
            self.model = self.config["model_class"](
                input_len=self.config["input_len"],
                input_channels=1,
                num_classes=self.config["num_classes"]
            )
        elif self.config_key in ["POLARCAP_SCSN", "POLARCAP_GLOBAL"]:
            # PolarCAP requires drop_rate parameter
            self.model = self.config["model_class"](drop_rate=0.3)
        elif self.config_key in ["CFM_SCSN", "CFM_GLOBAL"]:
            # CFM requires sample_rate parameter
            self.model = self.config["model_class"](sample_rate=100.0)
        elif self.config_key in ["RPNET_SCSN", "RPNET_GLOBAL"]:
            # RPNet requires sample_rate parameter
            self.model = self.config["model_class"](sample_rate=100.0)
        else:
            # Other models (ross_scsn, ross_global) use num_fm_classes parameter
            self.model = self.config["model_class"](num_fm_classes=self.config["num_classes"])
        
        # 3. Load weights (handles output layer shape mismatch)
        self._load_weights(self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_model_path(self, cache_dir: str, filename: str, user_path: Optional[str]) -> str:
        """
        Resolve model path:
        1. User specified path (if provided, use it directly)
        2. Hugging Face (priority)
        3. ModelScope (backup for domestic users)
        If all fail, raise error
        """
        # A. Check User Specified Path (highest priority)
        if user_path:
            if os.path.exists(user_path):
                print(f"Using manually specified model: {user_path}")
                return user_path
            else:
                raise FileNotFoundError(f"Model path not found: {user_path}")

        # B. Try Hugging Face and ModelScope (network download)
        return self._ensure_model(cache_dir, filename)

    def _ensure_model(self, cache_dir: str, filename: str) -> str:
        """Download model if not exists (Check Hugging Face network, try ModelScope if not accessible)."""
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        target_path = os.path.join(cache_dir, filename)
        if os.path.exists(target_path):
             print(f"Found existing local file: {target_path}")
             return target_path

        print(f"Checking for model '{filename}' in {cache_dir}...")

        # Get file paths for different sources
        model_info = MODELS_CONFIG[self.config_key]
        filename_hf = model_info.get("filename_hf", filename)
        filename_ms = model_info.get("filename_ms", filename)

        # 1. Check Hugging Face network connectivity
        hf_accessible = False
        try:
            import socket
            # Try to connect to huggingface.co port 443, timeout 1 second
            socket.setdefaulttimeout(1.0)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("huggingface.co", 443))
            hf_accessible = True
            print("Hugging Face network is accessible.")
        except Exception:
            print("Hugging Face network is not accessible, will use ModelScope.")
        finally:
            socket.setdefaulttimeout(None)

        # 2. Try Hugging Face (only try when network is accessible)
        if hf_accessible:
            try:
                print("Attempting download from Hugging Face...")
                local_path = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=filename_hf,
                    local_dir=cache_dir,
                )
                # If downloaded to subfolder, move to cache_dir
                if os.path.dirname(local_path) != cache_dir:
                    import shutil
                    final_path = os.path.join(cache_dir, filename)
                    shutil.move(local_path, final_path)
                    print(f"Model loaded from Hugging Face: {final_path}")
                    return final_path
                print(f"Model loaded from Hugging Face: {local_path}")
                return local_path
            except Exception as e:
                print(f"Hugging Face download failed: {e}")

        # 2. Try ModelScope (backup for domestic users)
        try:
            print("Attempting download from ModelScope...")
            # Try to import modelscope
            try:
                from modelscope.hub.file_download import model_file_download
            except ImportError:
                print("ModelScope not installed. Installing...")
                os.system("pip install modelscope -q")
                from modelscope.hub.file_download import model_file_download

            # Download specific file from ModelScope
            downloaded_file = model_file_download(
                model_id=MODELSCOPE_REPO,
                file_path=filename_ms,  # Path with subfolder
                cache_dir=cache_dir,
            )

            if os.path.exists(downloaded_file):
                print(f"Model loaded from ModelScope: {downloaded_file}")
                return downloaded_file
            else:
                raise FileNotFoundError(f"Could not find downloaded file: {downloaded_file}")

        except Exception as e:
            print(f"ModelScope download failed: {e}")

        # 3. If all fails, raise error
        raise RuntimeError(
            f"Could not download model '{filename}' from Hugging Face or ModelScope.\n"
            f"Please manually download it and place in '{cache_dir}' or specify model_path."
        )

    def _load_weights(self, path: str):
        """Load weights safely handling different saving formats (state_dict vs full checkpoint)."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            # If checkpoint has 'state_dict' (like from our Trainer), extract it
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                # Assume raw state_dict
                state_dict = checkpoint
            
            # Remove module. prefix if saved from DataParallel
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[name] = v
            
            # Try to load weights
            try:
                self.model.load_state_dict(new_state_dict, strict=True)
                print("Weights loaded successfully (strict mode).")
            except RuntimeError as e:
                # If strict mode fails, check if it's output layer shape mismatch
                if "output_layer" in str(e) and "size mismatch" in str(e):
                    print("Output layer size mismatch detected. Adapting model...")
                    
                    # For EQPolarity model, dynamically adjust output layer
                    if hasattr(self.model, 'output_layer'):
                        # Get output layer shape from checkpoint
                        checkpoint_output_weight = new_state_dict.get('output_layer.weight')
                        checkpoint_output_bias = new_state_dict.get('output_layer.bias')
                        
                        if checkpoint_output_weight is not None:
                            # Get number of output classes
                            num_classes = checkpoint_output_weight.shape[0]
                            in_features = checkpoint_output_weight.shape[1]
                            
                            # Dynamically create new output layer
                            import torch.nn as nn
                            self.model.output_layer = nn.Linear(in_features, num_classes)
                            
                            # Retry loading (non-strict mode)
                            self.model.load_state_dict(new_state_dict, strict=False)
                            print(f"Weights loaded successfully (adapted output layer to {num_classes} classes).")
                        else:
                            raise RuntimeError("Cannot adapt output layer: weight not found in checkpoint")
                    else:
                        # Non-strict mode loading (skip mismatched layers)
                        self.model.load_state_dict(new_state_dict, strict=False)
                        print("Weights loaded successfully (non-strict mode).")
                else:
                    # Other errors, re-raise
                    raise e
                
        except Exception as e:
            raise RuntimeError(f"Error loading model weights from {path}: {e}")

    def preprocess(self, waveforms: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
        """
        Preprocess waveforms: List/Array -> Normalized Tensor (N, 1, L).
        """
        # 1. Convert to Numpy
        if isinstance(waveforms, list):
            waveforms = np.array(waveforms)
            
        if isinstance(waveforms, torch.Tensor):
            waveforms = waveforms.cpu().numpy()

        if not isinstance(waveforms, np.ndarray):
            # Try last resort conversion
            try:
                waveforms = np.array(waveforms)
            except:
                raise TypeError(f"Expected list or numpy array, got {type(waveforms)}")
            
        # Ensure float32
        data = waveforms.astype(np.float32)
        
        # 2. Normalization (MaxAbs per sample)
        # Avoid division by zero
        if data.size > 0:
            max_vals = np.abs(data).max(axis=-1, keepdims=True)
            max_vals[max_vals == 0] = 1.0
            data = data / max_vals
        
        # 3. To Tensor
        tensor = torch.from_numpy(data)
        
        # 4. Shape Check
        # Expected: (N, L) -> (N, 1, L)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(1)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            
        return tensor.float()

    def predict(self, waveforms: Union[np.ndarray, List[np.ndarray]], return_probs: bool = False, batch_size: int = 2048, force_ud: Optional[bool] = None):
        """
        Run inference.
        
        Args:
            waveforms: Input data.
            return_probs: If True, return probabilities instead of class indices.
            batch_size: Batch size for inference to avoid OOM.
            force_ud: Force output to U/D only (no X). If None, use initialization setting.
                     For DiTingMotion model, if True, when model predicts X,
                     select the one with higher probability between U and D as final prediction.
            
        Returns:
            np.ndarray: Predicted classes (or probabilities).
            
        Example:
            >>> from seispolarity.inference import Predictor
            >>> predictor = Predictor("ROSS_GLOBAL")
            >>> waveforms = np.random.randn(10, 1, 400).astype(np.float32)
            >>> predictions = predictor.predict(waveforms)
        """
        if isinstance(waveforms, torch.Tensor):
             waveforms = waveforms.cpu().numpy()

        input_tensor = self.preprocess(waveforms)
        
        n_samples = input_tensor.shape[0]
        # Pre-allocate output to save memory (if needed) or just list append
        # For probs: (N, C), for classes: (N,)
        
        results = []
        
        # Determine whether to use force_ud
        use_force_ud = force_ud if force_ud is not None else self.force_ud
        
        with torch.no_grad():
            for i in tqdm(range(0, n_samples, batch_size), desc="Predicting", disable=n_samples <= batch_size):
                batch = input_tensor[i : i + batch_size].to(self.device)
                logits = self.model(batch)
                
                # Handle different output shapes
                output_index = self.config.get("output_index", None)
                
                if isinstance(logits, (tuple, list)):
                    # Multi-output model
                    if len(logits) == 8:
                        # DitingMotion model: use fused output (index 3)
                        fuse_output = logits[3]
                        probs = torch.softmax(fuse_output, dim=1)
                    elif len(logits) == 2:
                        # PPNet or PolarCAP: use output specified by output_index
                        if output_index is not None:
                            selected_output = logits[output_index]
                            probs = torch.softmax(selected_output, dim=1)
                        else:
                            # Default to second output (classification output)
                            probs = torch.softmax(logits[1], dim=1)
                    else:
                        # Other multi-output cases, use last output
                        probs = torch.softmax(logits[-1], dim=1)
                elif logits.shape[1] == 1:
                    # Binary classification sigmoid output (batch, 1)
                    probs = torch.sigmoid(logits)
                    # Convert to two-class probability distribution
                    probs = torch.cat([1 - probs, probs], dim=1)
                else:
                    # Multi-class softmax output (batch, num_classes)
                    probs = torch.softmax(logits, dim=1)
                
                # Handle force_ud (force output U/D)
                if use_force_ud and probs.shape[1] == 3:
                    # For three-class (U, D, X), force output U/D
                    # If prediction is X (index 2), select the one with higher probability between U and D
                    preds = torch.argmax(probs, dim=1)
                    x_mask = preds == 2  # Find samples predicted as X
                    
                    if x_mask.any():
                        # For samples predicted as X, select the one with higher probability between U(0) and D(1)
                        ud_probs = probs[x_mask, :2]  # Only get U and D probabilities
                        ud_preds = torch.argmax(ud_probs, dim=1)
                        preds[x_mask] = ud_preds
                else:
                    preds = torch.argmax(probs, dim=1)
                
                if return_probs:
                    results.append(probs.cpu().numpy())
                else:
                    results.append(preds.cpu().numpy())
            
        return np.concatenate(results, axis=0)

    def predict_from_loader(self, loader: torch.utils.data.DataLoader, return_probs: bool = False, force_ud: Optional[bool] = None):
        """
        Run inference on a DataLoader.
        
        Args:
            loader: PyTorch DataLoader yielding (waveforms, labels) or waveforms.
            return_probs: If True, return probabilities.
            force_ud: Force output to U/D only (no X). If None, use initialization setting.
                     For DiTingMotion model, if True, when model predicts X,
                     select the one with higher probability between U and D as final prediction.
            
        Returns:
            (predictions, labels): 
                - predictions: (N,) indices or (N, C) probabilities.
                - labels: (N,) ground truth labels if available, else None.
                
        Example:
            >>> from seispolarity.inference import Predictor
            >>> predictor = Predictor("ROSS_GLOBAL")
            >>> predictions, labels = predictor.predict_from_loader(dataloader)
        """
        self.model.eval()
        results = []
        labels_list = []
        
        device = self.device
        
        # Determine whether to use force_ud
        use_force_ud = force_ud if force_ud is not None else self.force_ud
        
        # Iterate through loader
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting", unit="batch"):
                # 1. Unpack Batch
                # Support (X, Y) or X
                if isinstance(batch, (list, tuple)):
                    imgs = batch[0]
                    if len(batch) > 1:
                        labels_list.append(batch[1])
                else:
                    imgs = batch
                
                # 2. Prepare Input
                # Determine standard tensor shape: (B, 1, L)
                if isinstance(imgs, np.ndarray):
                    x = torch.from_numpy(imgs)
                else:
                    x = imgs
                
                x = x.float().to(device)
                
                # Fix dimensions: (B, L) -> (B, 1, L)
                if x.ndim == 2:
                    x = x.unsqueeze(1)
                
                # 3. Model Inference
                logits = self.model(x)
                
                # Handle different output shapes
                output_index = self.config.get("output_index", None)
                
                if isinstance(logits, (tuple, list)):
                    # Multi-output model
                    if len(logits) == 8:
                        # DitingMotion model: use fused output (index 3)
                        fuse_output = logits[3]
                        probs = torch.softmax(fuse_output, dim=1)
                    elif len(logits) == 2:
                        # PPNet or PolarCAP: use output specified by output_index
                        if output_index is not None:
                            selected_output = logits[output_index]
                            probs = torch.softmax(selected_output, dim=1)
                        else:
                            # Default to second output (classification output)
                            probs = torch.softmax(logits[1], dim=1)
                    else:
                        # Other multi-output cases, use last output
                        probs = torch.softmax(logits[-1], dim=1)
                elif logits.shape[1] == 1:
                    # Binary classification sigmoid output (batch, 1)
                    probs = torch.sigmoid(logits)
                    # Convert to two-class probability distribution
                    probs = torch.cat([1 - probs, probs], dim=1)
                else:
                    # Multi-class softmax output (batch, num_classes)
                    probs = torch.softmax(logits, dim=1)
                
                # 4. Handle force_ud (force output U/D)
                if use_force_ud and probs.shape[1] == 3:
                    # For three-class (U, D, X), force output U/D
                    # If prediction is X (index 2), select the one with higher probability between U and D
                    preds = torch.argmax(probs, dim=1)
                    x_mask = preds == 2  # Find samples predicted as X
                    
                    if x_mask.any():
                        # For samples predicted as X, select the one with higher probability between U(0) and D(1)
                        ud_probs = probs[x_mask, :2]  # Only get U and D probabilities
                        ud_preds = torch.argmax(ud_probs, dim=1)
                        preds[x_mask] = ud_preds
                else:
                    preds = torch.argmax(probs, dim=1)
                
                # 5. Store Results
                if return_probs:
                    results.append(probs.cpu().numpy())
                else:
                    results.append(preds.cpu().numpy())
        
        # Concatenate
        final_preds = np.concatenate(results, axis=0) if results else np.array([])
        final_labels = np.concatenate(labels_list, axis=0) if labels_list else None
        
        return final_preds, final_labels

