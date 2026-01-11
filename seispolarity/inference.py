"""
Unified Inference Interface for SeisPolarity models.
seispolarity 模型的统一推断接口。
"""

import os
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

    def predict_from_loader(self, loader: torch.utils.data.DataLoader, return_probs: bool = False):
        """
        Run inference on a DataLoader.
        
        Args:
            loader: PyTorch DataLoader yielding (waveforms, labels) or waveforms.
            return_probs: If True, return probabilities.
            
        Returns:
            (predictions, labels): 
                - predictions: (N,) indices or (N, C) probabilities.
                - labels: (N,) ground truth labels if available, else None.
        """
        self.model.eval()
        results = []
        labels_list = []
        
        device = self.device
        
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
                probs = torch.softmax(logits, dim=1)
                
                # 4. Store Results
                if return_probs:
                    results.append(probs.cpu().numpy())
                else:
                    preds = torch.argmax(probs, dim=1)
                    results.append(preds.cpu().numpy())
        
        # Concatenate
        final_preds = np.concatenate(results, axis=0) if results else np.array([])
        final_labels = np.concatenate(labels_list, axis=0) if labels_list else None
        
        return final_preds, final_labels

