"""
Unified Inference Interface for SeisPolarity models.
seispolarity 模型的统一推断接口。
"""

import os

# Force usage of official Hugging Face endpoint by removing mirror settings
# 强制使用官方 Hugging Face 端点，移除镜像设置
if "HF_ENDPOINT" in os.environ:
    del os.environ["HF_ENDPOINT"]

import torch
import numpy as np
import warnings
from typing import Union, List, Optional
from huggingface_hub import hf_hub_download

# Import models
from seispolarity.models import SCSN

# Constants
HF_REPO = "HeXingChen/SeisPolarity-Model"
MODELS_CONFIG = {
    "ross": {
        "filename": "Ross_SCSN.pth",         # Filename in HF repo
        "model_class": SCSN,                 # Python class
        "input_len": 400,                    # Expected input length
        "center_crop": True,                 # Whether to crop center
        "crop_len": 400,                     # Length after crop
        "p_arrival_offset": 300,             # Assumed P-arrival in raw data (if cropping)
        "num_classes": 3,
        "class_map": {0: "Up", 1: "Down", 2: "Unknown"} # Example map
    }
}

class Predictor:
    """
    High-level interface for polarity prediction.
    极性预测的高级接口。
    
    Usage:
        >>> from seispolarity.inference import Predictor
        >>> model = Predictor("ross")
        >>> preds = model.predict(waveforms)
    """
    
    def __init__(self, model_name: str = "ross", device: Optional[str] = None, cache_dir: str = "./checkpoints_download"):
        """
        Initialize the predictor.
        初始化预测器。
        
        Args:
            model_name (str): Name of the model to use (default: "ross").
            device (str, optional): "cuda" or "cpu". If None, auto-detect.
            cache_dir (str): Directory to store downloaded models.
        """
        if model_name not in MODELS_CONFIG:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS_CONFIG.keys())}")
        
        self.config = MODELS_CONFIG[model_name]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 1. Download/Load Checkpoint
        self.checkpoint_path = self._ensure_model(cache_dir, self.config["filename"])
        
        # 2. Initialize Model
        self.model = self.config["model_class"](num_fm_classes=self.config["num_classes"])
        self._load_weights(self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
    def _ensure_model(self, cache_dir: str, filename: str) -> str:
        """Download model if not exists (Try HF first, then GitHub)."""
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        target_path = os.path.join(cache_dir, filename)
        if os.path.exists(target_path):
             print(f"Found existing local file: {target_path}")
             return target_path

        print(f"Checking for model '{filename}' in {cache_dir}...")
        
        # 1. Try Hugging Face
        try:
            print("Attempting download from Hugging Face...")
            local_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                local_dir=cache_dir,
            )
            print(f"Model loaded from {local_path}")
            return local_path
        except Exception as e:
            print(f"Hugging Face download failed: {e}")
        
        # 2. Try GitHub Fallback
        print("Attempting download from GitHub (Backup)...")
        # Construct GitHub Raw URL (Assuming file structure matches)
        # User specified: pretrained_model/Ross/Ross_SCSN.pth
        # We need to map the short filename 'Ross_SCSN.pth' to the repo path if they differ.
        # But for 'ross', the config has filename="Ross_SCSN.pth".
        # We'll hardcode the mapping logic or assume the user meant a specific location.
        
        github_url = f"https://github.com/Chuan1937/SeisPolarity/raw/main/pretrained_model/Ross/{filename}"
        
        try:
            torch.hub.download_url_to_file(github_url, target_path)
            if os.path.exists(target_path):
                # Check if file is valid (sometimes raw links return 404 html text)
                if os.path.getsize(target_path) < 1000: # suspiciously small
                    os.remove(target_path)
                    raise RuntimeError("Downloaded file is too small (likely 404 page).")
                print(f"Model successfully downloaded from GitHub to {target_path}")
                return target_path
            else:
                 raise RuntimeError("Download completed but file not found.")
        except Exception as e:
            print(f"GitHub download also failed: {e}")
            
            # Final check just in case
            if os.path.exists(target_path):
                return target_path
                
            raise RuntimeError("Could not download model from either Hugging Face or GitHub.")

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
                
            self.model.load_state_dict(new_state_dict)
            print("Weights loaded successfully.")
            
        except Exception as e:
            raise RuntimeError(f"Error loading model weights from {path}: {e}")

    def preprocess(self, waveforms: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
        """
        Preprocess waveforms: Crop, Normalize, Tensorize.
        预处理波形: 裁剪, 归一化, 转Tensor.
        
        Args:
            waveforms: Numpy array (N, L) or List of arrays.
        """
        # Convert to numpy if needed
        if isinstance(waveforms, list):
            waveforms = np.array(waveforms)
            
        if waveforms.ndim == 1:
            waveforms = waveforms[np.newaxis, :] # (1, L)
            
        N, L = waveforms.shape
        target_len = self.config["input_len"]
        
        # 1. Processing Loop (numpy vectorized)
        # Handle cropping
        if self.config["center_crop"] and L > target_len:
            # Default logic: Assume P is at 300 (standard for some data), or just center.
            # Our config says p_arrival_offset=300, crop_len=400. 
            # So start = 300 - 400//2 = 100. End = 500.
            # Check if data length supports this assumption
            start_idx = self.config["p_arrival_offset"] - (target_len // 2)
            if start_idx < 0: start_idx = 0
            end_idx = start_idx + target_len
            
            if end_idx > L:
                # Fallback to true center if length doesn't match standard
                mid = L // 2
                start_idx = mid - (target_len // 2)
                end_idx = start_idx + target_len
                
            processed_data = waveforms[:, start_idx:end_idx]
        elif L < target_len:
            raise ValueError(f"Input length {L} is shorter than model requirement {target_len}.")
        else:
            processed_data = waveforms
            
        # 2. Normalization (MaxAbs per sample)
        # (N, L)
        max_vals = np.abs(processed_data).max(axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1.0 # Avoid div by zero
        processed_data = processed_data / max_vals
        
        # 3. To Tensor
        # Model expects (N, 1, L) for Conv1d
        tensor = torch.from_numpy(processed_data).float()
        tensor = tensor.unsqueeze(1) # Add channel dim -> (N, 1, L)
        
        return tensor.to(self.device)

    def predict(self, waveforms: Union[np.ndarray, List[np.ndarray]], return_probs: bool = False):
        """
        Run inference.
        
        Args:
            waveforms: Input data.
            return_probs: If True, return probabilities instead of class indices.
            
        Returns:
            np.ndarray: Predicted classes (or probabilities).
        """
        x = self.preprocess(waveforms)
        
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            
        if return_probs:
            return probs.cpu().numpy()
        else:
            classes = torch.argmax(probs, dim=1)
            return classes.cpu().numpy()
