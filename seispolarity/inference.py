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
import httpx
from typing import Union, List, Optional
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Import models
from seispolarity.models import SCSN

# Constants
HF_REPO = "HeXingChen/SeisPolarity-Model"
MODELS_CONFIG = {
    "ross": {
        "filename": "Ross_SCSN.pth",         # Filename in HF repo
        "model_class": SCSN,                 # Python class
        "input_len": 400,                    # Expected input length
        "num_classes": 3,
        "class_map": {0: "Up", 1: "Down", 2: "Unknown"} # Example map
    }
}

from torch.utils.data import DataLoader
from seispolarity.data.scsn import SCSNDataset
from seispolarity.generate import FixedWindow, Normalize

def load_scsn_waveforms(h5_path, limit=None, window_p0=100, window_len=400, num_workers=4, batch_size=10000):
    """
    Load and preprocess SCSN waveforms using multiprocessing.
    
    Args:
        h5_path (str): Path to SCSN HDF5 file.
        limit (int, optional): Max samples.
        window_p0 (int): P-pick center index for windowing.
        window_len (int): Output window length.
        num_workers (int): Number of workers for DataLoader.
        batch_size (int): Batch size for loading.
        
    Returns:
        (np.ndarray, np.ndarray): Waveforms (N, L), Labels (N,)
    """
    dataset = SCSNDataset(h5_path=h5_path, limit=limit, preload=False)
    
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
        
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=2
    )

    all_waveforms = []
    all_labels = []
    
    for waveforms, labels in tqdm(loader, desc="Loading Data"):
        all_waveforms.append(waveforms)
        all_labels.append(labels)
        
    if not all_waveforms:
        return np.array([]), np.array([])
        
    return np.concatenate(all_waveforms), np.concatenate(all_labels)

class Predictor:
    """
    High-level interface for polarity prediction.
    极性预测的高级接口。
    
    Usage:
        >>> from seispolarity.inference import Predictor
        >>> model = Predictor("ross")
        >>> preds = model.predict(waveforms)
    """
    
    def __init__(self, model_name: str = "ross", device: Optional[str] = None, cache_dir: str = "./checkpoints_download", model_path: Optional[str] = None):
        """
        Initialize the predictor.
        初始化预测器。
        
        Args:
            model_name (str): Name of the model to use (default: "ross").
            device (str, optional): "cuda" or "cpu". If None, auto-detect.
            cache_dir (str): Directory to store downloaded models (default: "./checkpoints_download").
            model_path (str, optional): Manually specified path to the model file. If provided, skips download.
        """
        if model_name not in MODELS_CONFIG:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS_CONFIG.keys())}")
        
        self.config = MODELS_CONFIG[model_name]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 1. Download/Load Checkpoint
        self.checkpoint_path = self._resolve_model_path(cache_dir, self.config["filename"], model_path)
        
        # 2. Initialize Model
        self.model = self.config["model_class"](num_fm_classes=self.config["num_classes"])
        self._load_weights(self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_model_path(self, cache_dir: str, filename: str, user_path: Optional[str]) -> str:
        """
        Resolve model path in the following order:
        1. Pre-defined local paths (e.g. source repo structure)
        2. User specified path (if provided)
        3. Local cache directory
        4. Auto-download (HF -> GitHub)
        """
        # A. Check pre-defined local paths (relative to this file or project root)
        # Assuming inference.py is in seispolarity/inference.py
        # Project root is likely two levels up
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        candidates = [
            # 1. Source repo structure: pretrained_model/Ross/Ross_SCSN.pth
            os.path.join(project_root, "pretrained_model", "Ross", filename),
            # 2. Source repo structure: pretrained_models/Ross/Ross_SCSN.pth (plural)
            os.path.join(project_root, "pretrained_models", "Ross", filename),
        ]
        
        for cand in candidates:
            if os.path.exists(cand):
                print(f"Found model in repo structure: {cand}")
                return cand

        # B. Check User Specified Path
        if user_path:
            if os.path.exists(user_path):
                print(f"Using manually specified model: {user_path}")
                return user_path
            else:
                print(f"Warning: Manual model path '{user_path}' not found.")

        # C. Check Local Cache / Auto-Download
        return self._ensure_model(cache_dir, filename)

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
        # Construct Raw GitHub URL
        github_url = f"https://raw.githubusercontent.com/Chuan1937/SeisPolarity/main/pretrained_model/Ross/{filename}"
        
        try:
            print(f"Downloading from {github_url}...")
            with httpx.stream("GET", github_url, follow_redirects=True) as response:
                response.raise_for_status()
                with open(target_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
            print(f"Model downloaded from GitHub: {target_path}")
            return target_path
        except Exception as e:
            print(f"GitHub download failed: {e}")
            
            # Clean up partial file
            if os.path.exists(target_path):
                os.remove(target_path)
                
            raise RuntimeError(f"Could not download model from HF or GitHub. Please manually place '{filename}' in '{cache_dir}'.")
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
        # Verify length (Cropping must be done externally)
        if L != target_len:
            raise ValueError(f"Input length {L} does not match model requirement {target_len}.\n"
                             f"Please crop or pad your data externally before passing to Predictor.")
        
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
        
        return tensor

    def predict(self, waveforms: Union[np.ndarray, List[np.ndarray]], return_probs: bool = False, batch_size: int = 2048):
        """
        Run inference.
        
        Args:
            waveforms: Input data.
            return_probs: If True, return probabilities instead of class indices.
            batch_size: Batch size for inference to avoid OOM.
            
        Returns:
            np.ndarray: Predicted classes (or probabilities).
        """
        if isinstance(waveforms, torch.Tensor):
             waveforms = waveforms.cpu().numpy()

        input_tensor = self.preprocess(waveforms)
        
        n_samples = input_tensor.shape[0]
        # Pre-allocate output to save memory (if needed) or just list append
        # For probs: (N, C), for classes: (N,)
        
        results = []
        
        with torch.no_grad():
            for i in tqdm(range(0, n_samples, batch_size), desc="Predicting", disable=n_samples <= batch_size):
                batch = input_tensor[i : i + batch_size].to(self.device)
                logits = self.model(batch)
                
                probs = torch.softmax(logits, dim=1)
                
                if return_probs:
                    results.append(probs.cpu().numpy())
                else:
                    classes = torch.argmax(probs, dim=1)
                    results.append(classes.cpu().numpy())
            
        return np.concatenate(results, axis=0)
