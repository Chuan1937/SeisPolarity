"""
Unified Inference Interface for SeisPolarity models.
seispolarity 模型的统一推断接口。
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
        "output_index": None
    },
    "ROSS_GLOBAL": {
        "filename": "ROSS_GLOBAL.pth",
        "filename_hf": "ROSS/ROSS_GLOBAL.pth",
        "filename_ms": "ROSS/ROSS_GLOBAL.pth",
        "model_class": SCSN,
        "input_len": 400,
        "num_classes": 3,
        "class_map": {0: "Up", 1: "Down", 2: "Unknown"},
        "output_index": None
    },
    "EQPOLARITY_SCSN": {
        "filename": "EQPOLARITY_SCSN.pth",
        "filename_hf": "EQPOLARITY/EQPOLARITY_SCSN.pth",
        "filename_ms": "EQPOLARITY/EQPOLARITY_SCSN.pth",
        "model_class": EQPolarityCCT,
        "input_len": 600,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": None
    },
    "EQPOLARITY_TXED": {
        "filename": "EQPOLARITY_TXED.pth",
        "filename_hf": "EQPOLARITY/EQPOLARITY_TXED.pth",
        "filename_ms": "EQPOLARITY/EQPOLARITY_TXED.pth",
        "model_class": EQPolarityCCT,
        "input_len": 600,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": None
    },
    "EQPOLARITY_GLOBAL": {
        "filename": "EQPOLARITY_GLOBAL.pth",
        "filename_hf": "EQPOLARITY/EQPOLARITY_GLOBAL.pth",
        "filename_ms": "EQPOLARITY/EQPOLARITY_GLOBAL.pth",
        "model_class": EQPolarityCCT,
        "input_len": 600,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": None
    },
    "DITINGMOTION_DITINGSCSN": {
        "filename": "DITINGMOTION_DITINGSCSN.pth",
        "filename_hf": "DITINGMOTION/DITINGMOTION_DITINGSCSN.pth",
        "filename_ms": "DITINGMOTION/DITINGMOTION_DITINGSCSN.pth",
        "model_class": DitingMotion,
        "input_len": 128,
        "num_classes": 3,
        "class_map": {0: "Up", 1: "Down", 2: "Unknown"},
        "output_index": 3
    },
    "DITING_GLOBAL": {
        "filename": "DITING_GLOBAL.pth",
        "filename_hf": "DITINGMOTION/DITING_GLOBAL.pth",
        "filename_ms": "DITINGMOTION/DITING_GLOBAL.pth",
        "model_class": DitingMotion,
        "input_len": 128,
        "num_classes": 3,
        "class_map": {0: "Up", 1: "Down", 2: "Unknown"},
        "output_index": 3
    },
    "APP_SCSN": {
        "filename": "APP_SCSN.pth",
        "filename_hf": "APP/APP_SCSN.pth",
        "filename_ms": "APP/APP_SCSN.pth",
        "model_class": PPNet,
        "input_len": 400,
        "num_classes": 3,
        "class_map": {0: "Up", 1: "Down", 2: "Unknown"},
        "output_index": 1
    },
    "APP_GLOBAL": {
        "filename": "APP_GLOBAL.pth",
        "filename_hf": "APP/APP_GLOBAL.pth",
        "filename_ms": "APP/APP_GLOBAL.pth",
        "model_class": PPNet,
        "input_len": 400,
        "num_classes": 3,
        "class_map": {0: "Up", 1: "Down", 2: "Unknown"},
        "output_index": 1
    },
    "CFM_SCSN": {
        "filename": "CFM_SCSN.pth",
        "filename_hf": "CFM/CFM_SCSN.pth",
        "filename_ms": "CFM/CFM_SCSN.pth",
        "model_class": CFM,
        "input_len": 160,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": None
    },
    "CFM_GLOBAL": {
        "filename": "CFM_GLOBAL.pth",
        "filename_hf": "CFM/CFM_GLOBAL.pth",
        "filename_ms": "CFM/CFM_GLOBAL.pth",
        "model_class": CFM,
        "input_len": 160,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": None
    },
    "POLARCAP_SCSN": {
        "filename": "POLARCAP_SCSN.pth",
        "filename_hf": "POLARCAP/POLARCAP_SCSN.pth",
        "filename_ms": "POLARCAP/POLARCAP_SCSN.pth",
        "model_class": PolarCAP,
        "input_len": 64,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": 1
    },
    "POLARCAP_GLOBAL": {
        "filename": "POLARCAP_GLOBAL.pth",
        "filename_hf": "POLARCAP/POLARCAP_GLOBAL.pth",
        "filename_ms": "POLARCAP/POLARCAP_GLOBAL.pth",
        "model_class": PolarCAP,
        "input_len": 64,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": 1
    },
    "RPNET_SCSN": {
        "filename": "RPNET_SCSN.pth",
        "filename_hf": "RPNET/RPNET_SCSN.pth",
        "filename_ms": "RPNET/RPNET_SCSN.pth",
        "model_class": RPNet,
        "input_len": 400,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": None
    },
    "RPNET_GLOBAL": {
        "filename": "RPNET_GLOBAL.pth",
        "filename_hf": "RPNET/RPNET_GLOBAL.pth",
        "filename_ms": "RPNET/RPNET_GLOBAL.pth",
        "model_class": RPNet,
        "input_len": 400,
        "num_classes": 2,
        "class_map": {0: "Up", 1: "Down"},
        "output_index": None
    }
}

class Predictor:
    """
    High-level interface for polarity prediction.
    极性预测的高级接口。
    
    Usage:
        >>> from seispolarity.inference import Predictor
        >>> model = Predictor("ROSS_GLOBAL")
        >>> preds = model.predict(waveforms)
    
    Available Models (可使用模型):
        - ROSS_SCSN: ROSS_SCSN.pth (3 classes: Up, Down, Unknown)
        - ROSS_GLOBAL: ROSS_GLOBAL.pth (3 classes: Up, Down, Unknown)
        - EQPOLARITY_SCSN: EQPOLARITY_SCSN.pth (2 classes: Up, Down)
        - EQPOLARITY_TXED: EQPOLARITY_TXED.pth (2 classes: Up, Down)
        - EQPOLARITY_GLOBAL: EQPOLARITY_GLOBAL.pth (2 classes: Up, Down)
        - DITINGMOTION_DITINGSCSN: DITINGMOTION_DITINGSCSN.pth (3 classes: Up, Down, Unknown)
        - DITING_GLOBAL: DITING_GLOBAL.pth (3 classes: Up, Down, Unknown)
        - APP_SCSN: APP_SCSN.pth (3 classes: Up, Down, Unknown)
        - APP_GLOBAL: APP_GLOBAL.pth (3 classes: Up, Down, Unknown)
        - CFM_SCSN: CFM_SCSN.pth (2 classes: Up, Down)
        - CFM_GLOBAL: CFM_GLOBAL.pth (2 classes: Up, Down)
        - POLARCAP_SCSN: POLARCAP_SCSN.pth (2 classes: Up, Down)
        - POLARCAP_GLOBAL: POLARCAP_GLOBAL.pth (2 classes: Up, Down)
        - RPNET_SCSN: RPNET_SCSN.pth (2 classes: Up, Down)
        - RPNET_GLOBAL: RPNET_GLOBAL.pth (2 classes: Up, Down)
    
    Note: Use the full model name (e.g., "ROSS_SCSN", "ROSS_GLOBAL") to initialize the predictor.
          使用完整模型名字初始化预测器（如 "ROSS_SCSN", "ROSS_GLOBAL"）。
    """
    
    def __init__(self, model_name: str = "ROSS_SCSN", device: Optional[str] = None, cache_dir: str = "./checkpoints_download", model_path: Optional[str] = None, force_ud: bool = False):
        """
        Initialize the predictor.
        初始化预测器。

        Args:
            model_name (str): Name of the model to use (default: "ROSS_SCSN").
                             Must be one of the available model names (e.g., "ROSS_SCSN", "ROSS_GLOBAL", "APP_GLOBAL").
                             必须使用完整模型名字（如 "ROSS_SCSN", "ROSS_GLOBAL", "APP_GLOBAL"）。
            device (str, optional): "cuda" or "cpu". If None, auto-detect.
            cache_dir (str): Directory to store downloaded models (default: "./checkpoints_download").
            model_path (str, optional): Manually specified path to the model file. If provided, skips download.
            force_ud (bool): 是否强制输出U/D（不输出X）。对于DiTingMotion模型，如果为True，则当模型预测为X时，
                             会选择U和D中概率较高的那个作为最终预测。

        Raises:
            ValueError: If model_name is not found in available models.
                       如果模型名称不在可用模型列表中，抛出异常。
        """

        # model_name 可以是配置键（如 "ross"）或完整文件名（如 "ROSS_SCSN.pth"）
        # 在 MODELS_CONFIG 中查找匹配的配置
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
            print("启用强制输出U/D模式：当预测为X时，选择U/D中概率较高的那个")
        
        # 1. Download/Load Checkpoint
        self.checkpoint_path = self._resolve_model_path(cache_dir, self.config["filename"], model_path)
        
        # 2. Initialize Model
        # 根据模型类型使用不同的初始化参数
        if self.config_key in ["EQPOLARITY_SCSN", "EQPOLARITY_TXED", "EQPOLARITY_GLOBAL"]:
            # EQPolarityCCT 需要 input_length 参数
            self.model = self.config["model_class"](input_length=self.config["input_len"])
        elif self.config_key in ["DITINGMOTION_DITINGSCSN", "DITING_GLOBAL"]:
            # DitingMotion 需要 input_channels 参数
            self.model = self.config["model_class"](input_channels=2)
        elif self.config_key in ["APP_SCSN", "APP_GLOBAL"]:
            # PPNet 需要 input_len, input_channels, num_classes 参数
            self.model = self.config["model_class"](
                input_len=self.config["input_len"],
                input_channels=1,
                num_classes=self.config["num_classes"]
            )
        elif self.config_key in ["POLARCAP_SCSN", "POLARCAP_GLOBAL"]:
            # PolarCAP 需要 drop_rate 参数
            self.model = self.config["model_class"](drop_rate=0.3)
        elif self.config_key in ["CFM_SCSN", "CFM_GLOBAL"]:
            # CFM 需要 sample_rate 参数
            self.model = self.config["model_class"](sample_rate=100.0)
        elif self.config_key in ["RPNET_SCSN", "RPNET_GLOBAL"]:
            # RPNet 需要 sample_rate 参数
            self.model = self.config["model_class"](sample_rate=100.0)
        else:
            # 其他模型（ross_scsn, ross_global）使用 num_fm_classes 参数
            self.model = self.config["model_class"](num_fm_classes=self.config["num_classes"])
        
        # 3. 加载权重（这会处理输出层形状不匹配的问题）
        self._load_weights(self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_model_path(self, cache_dir: str, filename: str, user_path: Optional[str]) -> str:
        """
        Resolve model path:
        1. User specified path (if provided, use it directly)
        2. Hugging Face (优先)
        3. ModelScope (国内备用)
        如果都失败，直接报错
        """
        # A. Check User Specified Path (最高优先级)
        if user_path:
            if os.path.exists(user_path):
                print(f"Using manually specified model: {user_path}")
                return user_path
            else:
                raise FileNotFoundError(f"Model path not found: {user_path}")

        # B. Try Hugging Face and ModelScope (网络下载)
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

        # 1. 检测 Hugging Face 网络连通性
        hf_accessible = False
        try:
            import socket
            # 尝试连接 huggingface.co 的 443 端口，超时 1 秒
            socket.setdefaulttimeout(1.0)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("huggingface.co", 443))
            hf_accessible = True
            print("Hugging Face network is accessible.")
        except Exception:
            print("Hugging Face network is not accessible, will use ModelScope.")
        finally:
            socket.setdefaulttimeout(None)

        # 2. Try Hugging Face (仅在网络可访问时尝试)
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

        # 2. Try ModelScope (国内备用)
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
                file_path=filename_ms,  # 带子文件夹的路径
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
            
            # 尝试加载权重
            try:
                self.model.load_state_dict(new_state_dict, strict=True)
                print("Weights loaded successfully (strict mode).")
            except RuntimeError as e:
                # 如果严格模式失败，检查是否是输出层形状不匹配
                if "output_layer" in str(e) and "size mismatch" in str(e):
                    print("Output layer size mismatch detected. Adapting model...")
                    
                    # 对于 EQPolarity 模型，动态调整输出层
                    if hasattr(self.model, 'output_layer'):
                        # 获取检查点中输出层的形状
                        checkpoint_output_weight = new_state_dict.get('output_layer.weight')
                        checkpoint_output_bias = new_state_dict.get('output_layer.bias')
                        
                        if checkpoint_output_weight is not None:
                            # 获取输出类别数
                            num_classes = checkpoint_output_weight.shape[0]
                            in_features = checkpoint_output_weight.shape[1]
                            
                            # 动态创建新的输出层
                            import torch.nn as nn
                            self.model.output_layer = nn.Linear(in_features, num_classes)
                            
                            # 重新尝试加载（非严格模式）
                            self.model.load_state_dict(new_state_dict, strict=False)
                            print(f"Weights loaded successfully (adapted output layer to {num_classes} classes).")
                        else:
                            raise RuntimeError("Cannot adapt output layer: weight not found in checkpoint")
                    else:
                        # 非严格模式加载（跳过不匹配的层）
                        self.model.load_state_dict(new_state_dict, strict=False)
                        print("Weights loaded successfully (non-strict mode).")
                else:
                    # 其他错误，重新抛出
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
        执行推理。
        
        Args:
            waveforms: Input data. 输入数据。
            return_probs: If True, return probabilities instead of class indices.
                         如果为True，返回概率值而不是类别索引。
            batch_size: Batch size for inference to avoid OOM.
                       批处理大小，避免内存溢出。
            force_ud: 是否强制输出U/D（不输出X）。如果为None，则使用初始化时的设置。
                     对于DiTingMotion模型，如果为True，则当模型预测为X时，
                     会选择U和D中概率较高的那个作为最终预测。
            
        Returns:
            np.ndarray: Predicted classes (or probabilities). 预测的类别（或概率值）。
            
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
        
        # 确定是否使用force_ud
        use_force_ud = force_ud if force_ud is not None else self.force_ud
        
        with torch.no_grad():
            for i in tqdm(range(0, n_samples, batch_size), desc="Predicting", disable=n_samples <= batch_size):
                batch = input_tensor[i : i + batch_size].to(self.device)
                logits = self.model(batch)
                
                # 处理不同形状的输出
                output_index = self.config.get("output_index", None)
                
                if isinstance(logits, (tuple, list)):
                    # 多输出模型
                    if len(logits) == 8:
                        # DitingMotion模型：使用融合输出（索引3）
                        fuse_output = logits[3]
                        probs = torch.softmax(fuse_output, dim=1)
                    elif len(logits) == 2:
                        # PPNet 或 PolarCAP：使用 output_index 指定的输出
                        if output_index is not None:
                            selected_output = logits[output_index]
                            probs = torch.softmax(selected_output, dim=1)
                        else:
                            # 默认使用第二个输出（分类输出）
                            probs = torch.softmax(logits[1], dim=1)
                    else:
                        # 其他多输出情况，使用最后一个输出
                        probs = torch.softmax(logits[-1], dim=1)
                elif logits.shape[1] == 1:
                    # 二分类的 sigmoid 输出 (batch, 1)
                    probs = torch.sigmoid(logits)
                    # 转换为两类的概率分布
                    probs = torch.cat([1 - probs, probs], dim=1)
                else:
                    # 多分类的 softmax 输出 (batch, num_classes)
                    probs = torch.softmax(logits, dim=1)
                
                # 处理force_ud（强制输出U/D）
                if use_force_ud and probs.shape[1] == 3:
                    # 对于三分类（U, D, X），强制输出U/D
                    # 如果预测为X（索引2），则选择U和D中概率较高的那个
                    preds = torch.argmax(probs, dim=1)
                    x_mask = preds == 2  # 找到预测为X的样本
                    
                    if x_mask.any():
                        # 对于预测为X的样本，选择U(0)和D(1)中概率较高的那个
                        ud_probs = probs[x_mask, :2]  # 只取U和D的概率
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
        在DataLoader上执行推理。
        
        Args:
            loader: PyTorch DataLoader yielding (waveforms, labels) or waveforms.
                   PyTorch DataLoader，输出 (waveforms, labels) 或仅 waveforms。
            return_probs: If True, return probabilities.
                         如果为True，返回概率值。
            force_ud: 是否强制输出U/D（不输出X）。如果为None，则使用初始化时的设置。
                     对于DiTingMotion模型，如果为True，则当模型预测为X时，
                     会选择U和D中概率较高的那个作为最终预测。
            
        Returns:
            (predictions, labels): 
                - predictions: (N,) indices or (N, C) probabilities. 预测的类别索引或概率值。
                - labels: (N,) ground truth labels if available, else None. 真实标签（如果可用），否则为None。
                
        Example:
            >>> from seispolarity.inference import Predictor
            >>> predictor = Predictor("ROSS_GLOBAL")
            >>> predictions, labels = predictor.predict_from_loader(dataloader)
        """
        self.model.eval()
        results = []
        labels_list = []
        
        device = self.device
        
        # 确定是否使用force_ud
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
                
                # 处理不同形状的输出
                output_index = self.config.get("output_index", None)
                
                if isinstance(logits, (tuple, list)):
                    # 多输出模型
                    if len(logits) == 8:
                        # DitingMotion模型：使用融合输出（索引3）
                        fuse_output = logits[3]
                        probs = torch.softmax(fuse_output, dim=1)
                    elif len(logits) == 2:
                        # PPNet 或 PolarCAP：使用 output_index 指定的输出
                        if output_index is not None:
                            selected_output = logits[output_index]
                            probs = torch.softmax(selected_output, dim=1)
                        else:
                            # 默认使用第二个输出（分类输出）
                            probs = torch.softmax(logits[1], dim=1)
                    else:
                        # 其他多输出情况，使用最后一个输出
                        probs = torch.softmax(logits[-1], dim=1)
                elif logits.shape[1] == 1:
                    # 二分类的 sigmoid 输出 (batch, 1)
                    probs = torch.sigmoid(logits)
                    # 转换为两类的概率分布
                    probs = torch.cat([1 - probs, probs], dim=1)
                else:
                    # 多分类的 softmax 输出 (batch, num_classes)
                    probs = torch.softmax(logits, dim=1)
                
                # 4. 处理force_ud（强制输出U/D）
                if use_force_ud and probs.shape[1] == 3:
                    # 对于三分类（U, D, X），强制输出U/D
                    # 如果预测为X（索引2），则选择U和D中概率较高的那个
                    preds = torch.argmax(probs, dim=1)
                    x_mask = preds == 2  # 找到预测为X的样本
                    
                    if x_mask.any():
                        # 对于预测为X的样本，选择U(0)和D(1)中概率较高的那个
                        ud_probs = probs[x_mask, :2]  # 只取U和D的概率
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

