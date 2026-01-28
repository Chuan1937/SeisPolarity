from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, List, Optional
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from seispolarity.generate import GenericGenerator


@dataclass
class TrainingConfig:
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 1e-3
    num_workers: int = 0
    train_val_split: float = 0.8  # 训练集比例（相对于总数据集）
    val_split: float = 0.1  # 验证集比例（相对于总数据集）
    test_split: float = 0.1  # 测试集比例（相对于总数据集）
    limit: Optional[int] = None
    label_key: str = "label"
    device: Optional[str] = None
    checkpoint_dir: str = "."
    save_best_only: bool = True
    patience: int = -1 # Early stopping patience. -1 means disabled.
    resume_checkpoint: Optional[str] = None
    loss_fn: Optional[Callable] = None # Custom loss function, defaults to nn.CrossEntropyLoss()
    output_index: Optional[int] = 0 # Index of output to use for loss if model returns tuple. Set to None to pass all outputs.
    metric_index: int = 0 # Index of output to use for metrics if model returns tuple.
    random_seed: Optional[int] = 42 # Random seed for dataset splitting and reproducibility


def default_device(config: TrainingConfig) -> torch.device:
    if config.device:
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MetadataToLabel:
    """Augmentation: moves metadata[label_key] into key 'y'."""

    def __init__(self, metadata_key: str = "label", key: str = "y", label_map=None):
        self.metadata_key = metadata_key
        self.key = key
        self.label_map = label_map or {"U": 0, "D": 1, "X": 2}

    def _to_numeric_label(self, raw_label):
        """Convert mixed/byte/string labels to int64 for collation."""
        arr = np.array(raw_label)

        # Numeric types are returned directly
        if arr.dtype.kind in ("i", "u", "f"):
            return arr.astype(np.int64)

        # Handle string/bytes/object labels by mapping to integers
        if arr.dtype.kind in ("S", "U", "O"):
            def to_char(x):
                if isinstance(x, bytes):
                    return x.decode()
                return str(x)

            chars = np.vectorize(to_char)(arr)
            
            if chars.ndim == 0:
                c = str(chars)
                mapped = self.label_map.get(c.upper(), self.label_map.get("X", 2))
                return np.array(mapped, dtype=np.int64)

            mapped = [self.label_map.get(c.upper(), self.label_map.get("X", 2)) for c in chars]
            return np.array(mapped, dtype=np.int64)

        # Fallback conversion
        try:
            return arr.astype(np.int64)
        except Exception:
            return np.array([self.label_map.get("X", 2)], dtype=np.int64)

    def __call__(self, state_dict):
        # 处理state_dict["X"]可能不是元组的情况
        x_value = state_dict.get("X", None)
        if isinstance(x_value, tuple) and len(x_value) == 2:
            _, metadata = x_value
        else:
            # 如果state_dict["X"]不是元组，尝试从其他地方获取metadata
            metadata = state_dict.get("metadata", {})
        
        raw_label = metadata.get(self.metadata_key, None)
        numeric_label = self._to_numeric_label(raw_label) if raw_label is not None else np.array([-1], dtype=np.int64)
        # 只返回标签数据，而不是元组，以避免DataLoader的collate问题
        state_dict[self.key] = numeric_label


class MultiLabelExtractor:
    """
    提取多个标签的增强类，适用于DitingMotion等多任务模型。
    
    可以同时提取polarity标签和clarity标签。
    """
    def __init__(self, polarity_key="label", clarity_key="clarity", 
                 polarity_label_map=None, clarity_label_map=None):
        self.polarity_key = polarity_key
        self.clarity_key = clarity_key
        self.polarity_label_map = polarity_label_map or {"U": 0, "D": 1, "X": 2}
        self.clarity_label_map = clarity_label_map or {"I": 0, "E": 1, "K": 2}
    
    def _to_numeric_label(self, raw_label, label_map):
        """Convert mixed/byte/string labels to int64 for collation."""
        if raw_label is None:
            return np.array([-1], dtype=np.int64)
            
        arr = np.array(raw_label)

        # Numeric types are returned directly
        if arr.dtype.kind in ("i", "u", "f"):
            return arr.astype(np.int64)

        # Handle string/bytes/object labels by mapping to integers
        if arr.dtype.kind in ("S", "U", "O"):
            def to_char(x):
                if isinstance(x, bytes):
                    return x.decode()
                return str(x)

            chars = np.vectorize(to_char)(arr)
            
            if chars.ndim == 0:
                c = str(chars)
                mapped = label_map.get(c.upper(), label_map.get("X", 2))
                return np.array(mapped, dtype=np.int64)

            mapped = [label_map.get(c.upper(), label_map.get("X", 2)) for c in chars]
            return np.array(mapped, dtype=np.int64)

        # Fallback conversion
        try:
            return arr.astype(np.int64)
        except Exception:
            return np.array([label_map.get("X", 2)], dtype=np.int64)
    
    def __call__(self, state_dict):
        # 处理state_dict["X"]可能不是元组的情况
        if isinstance(state_dict["X"], tuple) and len(state_dict["X"]) == 2:
            _, metadata = state_dict["X"]
        else:
            # 如果state_dict["X"]不是元组，尝试从其他地方获取metadata
            metadata = state_dict.get("metadata", {})
        
        # 提取polarity标签
        raw_polarity = metadata.get(self.polarity_key, None)
        polarity_label = self._to_numeric_label(raw_polarity, self.polarity_label_map)
        
        # 提取clarity标签
        raw_clarity = metadata.get(self.clarity_key, None)
        clarity_label = self._to_numeric_label(raw_clarity, self.clarity_label_map)
        
        # 将两个标签存储为元组（但不包含None），以避免DataLoader的collate问题
        state_dict["y"] = (polarity_label, clarity_label)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataset,
        config: TrainingConfig,
        train_augmentations: Optional[List[Callable]] = None,
        val_augmentations: Optional[List[Callable]] = None,
        test_augmentations: Optional[List[Callable]] = None,
        val_dataset=None,
        test_dataset=None,
    ):
        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = default_device(config)
        self.train_augmentations = train_augmentations or []
        self.val_augmentations = val_augmentations or []
        self.test_augmentations = test_augmentations or []
        
        # Check if loss function takes 'inputs' argument
        self._loss_takes_inputs = False
        if self.config.loss_fn is not None:
            import inspect
            try:
                # Use forward method signature if it's an nn.Module, else the callable itself
                fn = self.config.loss_fn.forward if hasattr(self.config.loss_fn, 'forward') else self.config.loss_fn
                sig = inspect.signature(fn)
                if 'inputs' in sig.parameters:
                    self._loss_takes_inputs = True
            except Exception:
                pass

        # Initialize Log File
        self.log_path = Path(config.checkpoint_dir) / "training_log.txt"
        self._init_log_file()

    def _init_log_file(self):
        """Initialize log file with configuration parameters."""
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            f.write("="*50 + "\n")
            f.write(f"Training Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            f.write("Configuration:\n")
            # Convert config to dict, handle non-serializable types if any
            cfg_dict = asdict(self.config)
            for k, v in cfg_dict.items():
                f.write(f"{k}: {v}\n")
            f.write("\n" + "="*50 + "\n")
            f.write(f"{'Epoch':<6} | {'Train Loss':<12} | {'Train Acc':<10} | {'Val Loss':<12} | {'Val Acc':<10} | {'Time':<20}\n")
            f.write("-" * 80 + "\n")

    def _log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, test_loss=None, test_acc=None):
        """Log epoch stats to file."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        if test_loss is not None and test_acc is not None:
            log_str = f"{epoch:<6} | {train_loss:<12.4f} | {train_acc:<10.2f} | {val_loss:<12.4f} | {val_acc:<10.2f} | {test_loss:<12.4f} | {test_acc:<10.2f} | {timestamp:<20}\n"
        else:
            log_str = f"{epoch:<6} | {train_loss:<12.4f} | {train_acc:<10.2f} | {val_loss:<12.4f} | {val_acc:<10.2f} | {timestamp:<20}\n"
        with open(self.log_path, "a") as f:
            f.write(log_str)

    def _build_loaders(self):
        cfg = self.config

        if self.val_dataset is not None and self.test_dataset is not None:
            # Use provided explicit splits
            train_dataset = self.dataset
            val_dataset = self.val_dataset
            test_dataset = self.test_dataset
        else:
            # Split dataset into train/val/test
            total_size = len(self.dataset)
            train_size = int(cfg.train_val_split * total_size)
            val_size = int(cfg.val_split * total_size)
            test_size = total_size - train_size - val_size
            
            # Ensure test_size is not negative
            if test_size < 0:
                raise ValueError("train_val_split + val_split must be <= 1")
            
            # 设置随机种子以确保可复现性
            if cfg.random_seed is not None:
                generator = torch.Generator().manual_seed(cfg.random_seed)
            else:
                generator = None
            
            # 处理test_size为0的情况
            if test_size == 0:
                # 只划分训练集和验证集
                train_dataset, val_dataset = random_split(
                    self.dataset, [train_size, val_size], generator=generator
                )
                test_dataset = None
            else:
                train_dataset, val_dataset, test_dataset = random_split(
                    self.dataset, [train_size, val_size, test_size], generator=generator
                )

        # Generators
        train_gen = GenericGenerator(train_dataset)
        val_gen = GenericGenerator(val_dataset)
        
        # 处理test_dataset为None的情况
        if test_dataset is not None:
            test_gen = GenericGenerator(test_dataset)
        else:
            test_gen = None

        # 根据模型类型选择标签提取器
        # 检查是否是DitingMotion模型（有8个输出）
        model_class_name = self.model.__class__.__name__
        is_diting_motion = model_class_name == "DitingMotion"
        
        if is_diting_motion:
            # 对于DitingMotion模型，使用MultiLabelExtractor提取polarity和clarity标签
            base_augmentation = MultiLabelExtractor(
                polarity_key=cfg.label_key,
                clarity_key="clarity"
            )
        else:
            # 对于其他模型，使用MetadataToLabel提取单个标签
            base_augmentation = MetadataToLabel(metadata_key=cfg.label_key)
        
        # 检查数据集是否已经有数据增强
        # 如果数据集已经有数据增强，我们只添加MetadataToLabel
        # 否则添加所有增强（包括用户传入的）
        
        # 训练集增强
        if hasattr(train_dataset, 'augmentations') and train_dataset.augmentations:
            # 数据集已经有增强，只添加MetadataToLabel
            train_gen.add_augmentations([base_augmentation])
        else:
            # 数据集没有增强，添加所有增强
            train_augmentations = [base_augmentation] + self.train_augmentations
            train_gen.add_augmentations(train_augmentations)
        
        # 验证集增强
        if hasattr(val_dataset, 'augmentations') and val_dataset.augmentations:
            # 数据集已经有增强，只添加MetadataToLabel
            val_gen.add_augmentations([base_augmentation])
        else:
            # 数据集没有增强，添加所有增强
            val_augmentations = [base_augmentation] + self.val_augmentations
            val_gen.add_augmentations(val_augmentations)
        
        # 测试集增强（如果存在测试集）
        if test_gen is not None:
            if hasattr(test_dataset, 'augmentations') and test_dataset.augmentations:
                # 数据集已经有增强，只添加MetadataToLabel
                test_gen.add_augmentations([base_augmentation])
            else:
                # 数据集没有增强，添加所有增强
                test_augmentations = [base_augmentation] + self.test_augmentations
                test_gen.add_augmentations(test_augmentations)

        train_loader = DataLoader(
            train_gen,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=self._collate_fn,
        )
        val_loader = DataLoader(
            val_gen,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=self._collate_fn,
        )
        # 创建测试集DataLoader（如果存在测试集）
        if test_gen is not None:
            test_loader = DataLoader(
                test_gen,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                collate_fn=self._collate_fn,
            )
        else:
            # 创建一个空的测试集DataLoader
            test_loader = DataLoader(
                [],  # 空数据集
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
            )
        
        return train_loader, val_loader, test_loader

    @staticmethod
    def _collate_fn(batch):
        """
        自定义 collate 函数来处理生成器的输出。
        生成器返回的字典包含：
        - "X": (波形数据, 元数据字典) 或 波形数据
        - "y": 标签数据 或 (polarity_label, clarity_label)
        """
        # 提取 X 和 y
        X_list = []
        y_list = []
        
        for sample in batch:
            # 提取 X（波形数据）
            if "X" in sample:
                x_val = sample["X"]
                if isinstance(x_val, tuple) and len(x_val) == 2:
                    X_list.append(x_val[0])  # 波形数据
                else:
                    X_list.append(x_val)
            
            # 提取 y（标签）
            if "y" in sample:
                y_list.append(sample["y"])
        
        # 转换为张量
        if X_list:
            # 检查是否所有 X 都是 numpy 数组
            if all(isinstance(x, np.ndarray) for x in X_list):
                X_batch = torch.tensor(np.stack(X_list))
            else:
                # 如果已经是张量，直接 stack
                X_batch = torch.stack(X_list) if isinstance(X_list[0], torch.Tensor) else torch.tensor(X_list)
        else:
            X_batch = torch.tensor([])
        
        if y_list:
            # 检查 y 是否是元组（多标签情况）
            if all(isinstance(y, tuple) for y in y_list):
                # 多标签情况：y 是 (polarity_label, clarity_label)
                polarity_labels = [y[0] for y in y_list]
                clarity_labels = [y[1] for y in y_list]
                
                # 转换为张量
                polarity_batch = torch.tensor(np.stack(polarity_labels)) if isinstance(polarity_labels[0], np.ndarray) else torch.stack(polarity_labels)
                clarity_batch = torch.tensor(np.stack(clarity_labels)) if isinstance(clarity_labels[0], np.ndarray) else torch.stack(clarity_labels)
                
                y_batch = (polarity_batch, clarity_batch)
            else:
                # 单标签情况
                y_batch = torch.tensor(np.stack(y_list)) if isinstance(y_list[0], np.ndarray) else torch.stack(y_list)
        else:
            y_batch = torch.tensor([])
        
        return {"X": X_batch, "y": y_batch}

    def train(self):
        cfg = self.config
        device = self.device
        print(f"Using device: {device}")

        train_loader, val_loader, test_loader = self._build_loaders()
        print(f"Dataset sizes - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

        self.model.to(device)

        if cfg.resume_checkpoint:
            if Path(cfg.resume_checkpoint).exists():
                print(f"Resuming training from checkpoint: {cfg.resume_checkpoint}")
                self.model.load_state_dict(torch.load(cfg.resume_checkpoint, map_location=device))
            else:
                print(f"Checkpoint {cfg.resume_checkpoint} not found. Starting from scratch.")

        # Use custom loss function if provided, otherwise use default CrossEntropyLoss
        if cfg.loss_fn is not None:
            criterion = cfg.loss_fn
            print(f"Using custom loss function: {criterion.__class__.__name__}")
        else:
            criterion = nn.CrossEntropyLoss()
            print("Using default loss function: CrossEntropyLoss")
        
        optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        best_val_acc = -1.0
        patience_counter = 0 # Counter for early stopping

        for epoch in range(cfg.epochs):
            # Train
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]")
            for batch in pbar:
                inputs = batch["X"].to(device)
                batch_labels = batch["y"]
                
                # 处理多标签情况：batch["y"]可能是元组(polarity_labels, clarity_labels)
                if isinstance(batch_labels, (tuple, list)) and len(batch_labels) == 2:
                    # 分别移动两个标签到设备
                    polarity_labels = batch_labels[0].to(device)
                    clarity_labels = batch_labels[1].to(device)
                    labels = (polarity_labels, clarity_labels)
                else:
                    # 单标签情况
                    labels = batch_labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # 提取正确的输出用于损失计算
                if isinstance(outputs, (tuple, list)):
                    loss_output = outputs[cfg.output_index] if cfg.output_index is not None else outputs
                else:
                    loss_output = outputs
                
                # 稳健性处理：如果输出是 (batch, 1) 而标签是 (batch,)，自动处理
                curr_labels = labels
                if not isinstance(loss_output, (tuple, list)):
                    if loss_output.ndim == 2 and loss_output.shape[1] == 1 and curr_labels.ndim == 1:
                        curr_labels = curr_labels.unsqueeze(1).float()
                    elif isinstance(criterion, (nn.BCEWithLogitsLoss, nn.BCELoss)):
                        curr_labels = curr_labels.float()
                        if curr_labels.ndim == 1:
                            curr_labels = curr_labels.unsqueeze(1)
                
                # 计算损失，考虑是否需要传递原始输入
                if self._loss_takes_inputs:
                    loss = criterion(loss_output, curr_labels, inputs=inputs)
                else:
                    loss = criterion(loss_output, curr_labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # 计算准确率
                if isinstance(outputs, (tuple, list)):
                    metric_out = outputs[cfg.metric_index]
                else:
                    metric_out = outputs
                if isinstance(labels, (tuple, list)) and len(labels) == 2:
                    current_labels = labels[0]
                else:
                    current_labels = labels

                # 稳健的准确率计算：处理 2 分类 (batch, 1) 和多分类 (batch, C)
                if metric_out.ndim == 2 and metric_out.shape[1] == 1:
                    # 二分类 (batch, 1)，使用 0.5 阈值
                    predicted = (torch.sigmoid(metric_out) > 0.5).long().view(-1)
                else:
                    # 多分类 (batch, C)
                    _, predicted = torch.max(metric_out.data, 1)
                
                total += current_labels.size(0)
                correct += (predicted == current_labels.view(-1)).sum().item()
                
                pbar.set_postfix({"loss": running_loss / (pbar.n + 1), "acc": 100 * correct / total})

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total if total else 0.0

            # Validate
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Val]"):
                    inputs = batch["X"].to(device)
                    batch_labels = batch["y"]
                    
                    # 处理多标签情况
                    if isinstance(batch_labels, (tuple, list)) and len(batch_labels) == 2:
                        # 分别移动两个标签到设备
                        polarity_labels = batch_labels[0].to(device)
                        clarity_labels = batch_labels[1].to(device)
                        labels = (polarity_labels, clarity_labels)
                    else:
                        # 单标签情况
                        labels = batch_labels.to(device)
                    
                    outputs = self.model(inputs)
                    
                    # 提取正确的输出用于损失计算
                    if isinstance(outputs, (tuple, list)):
                        loss_output = outputs[cfg.output_index] if cfg.output_index is not None else outputs
                    else:
                        loss_output = outputs
                    
                    # 稳健性处理
                    curr_labels = labels
                    if not isinstance(loss_output, (tuple, list)):
                        if loss_output.ndim == 2 and loss_output.shape[1] == 1 and curr_labels.ndim == 1:
                            curr_labels = curr_labels.unsqueeze(1).float()
                        elif isinstance(criterion, (nn.BCEWithLogitsLoss, nn.BCELoss)):
                            curr_labels = curr_labels.float()
                            if curr_labels.ndim == 1:
                                curr_labels = curr_labels.unsqueeze(1)
                    
                    # 计算损失
                    if self._loss_takes_inputs:
                        loss = criterion(loss_output, curr_labels, inputs=inputs)
                    else:
                        loss = criterion(loss_output, curr_labels)
                    val_loss += loss.item()
                    
                    # 计算准确率
                    if isinstance(outputs, (tuple, list)):
                        metric_out = outputs[cfg.metric_index]
                    else:
                        metric_out = outputs
                    if isinstance(labels, (tuple, list)) and len(labels) == 2:
                        current_labels = labels[0]
                    else:
                        current_labels = labels

                    if metric_out.ndim == 2 and metric_out.shape[1] == 1:
                        predicted = (torch.sigmoid(metric_out) > 0.5).long().view(-1)
                    else:
                        _, predicted = torch.max(metric_out.data, 1)
                    
                    total += current_labels.size(0)
                    correct += (predicted == current_labels.view(-1)).sum().item()
            val_loss /= len(val_loader)
            val_acc = 100 * correct / total if total else 0.0

            # Test evaluation (optional, can be done less frequently to save time)
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            test_acc = 0.0
            
            # 只有在有测试数据时才进行评估
            if len(test_loader.dataset) > 0:
                self.model.eval()
                with torch.no_grad():
                    for batch in test_loader:
                        inputs = batch["X"].to(device)
                        batch_labels = batch["y"]
                        
                        # 处理多标签情况
                        if isinstance(batch_labels, (tuple, list)) and len(batch_labels) == 2:
                            # 分别移动两个标签到设备
                            polarity_labels = batch_labels[0].to(device)
                            clarity_labels = batch_labels[1].to(device)
                            labels = (polarity_labels, clarity_labels)
                        else:
                            # 单标签情况
                            labels = batch_labels.to(device)
                        
                        outputs = self.model(inputs)
                        
                        # 提取正确的输出用于损失计算
                        if isinstance(outputs, (tuple, list)):
                            loss_output = outputs[cfg.output_index] if cfg.output_index is not None else outputs
                        else:
                            loss_output = outputs
                        
                        # 稳健性处理
                        curr_labels = labels
                        if not isinstance(loss_output, (tuple, list)):
                            if loss_output.ndim == 2 and loss_output.shape[1] == 1 and curr_labels.ndim == 1:
                                curr_labels = curr_labels.unsqueeze(1).float()
                            elif isinstance(criterion, (nn.BCEWithLogitsLoss, nn.BCELoss)):
                                curr_labels = curr_labels.float()
                                if curr_labels.ndim == 1:
                                    curr_labels = curr_labels.unsqueeze(1)
                        
                        # 计算损失
                        if self._loss_takes_inputs:
                            loss = criterion(loss_output, curr_labels, inputs=inputs)
                        else:
                            loss = criterion(loss_output, curr_labels)
                        test_loss += loss.item()
                        
                        # 计算准确率
                        if isinstance(outputs, (tuple, list)):
                            metric_out = outputs[cfg.metric_index]
                        else:
                            metric_out = outputs
                        if isinstance(labels, (tuple, list)) and len(labels) == 2:
                            current_labels = labels[0]
                        else:
                            current_labels = labels

                        if metric_out.ndim == 2 and metric_out.shape[1] == 1:
                            predicted = (torch.sigmoid(metric_out) > 0.5).long().view(-1)
                        else:
                            _, predicted = torch.max(metric_out.data, 1)
                        
                        test_total += current_labels.size(0)
                        test_correct += (predicted == current_labels.view(-1)).sum().item()
                test_loss /= len(test_loader)
                test_acc = 100 * test_correct / test_total if test_total else 0.0
            
            # 打印训练进度
            if len(test_loader.dataset) > 0:
                print(
                    f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                    f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
                )
            else:
                print(
                    f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )
            
            # Write to log file
            self._log_epoch(epoch+1, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)

            # Checkpoint & Early Stopping
            if cfg.save_best_only:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_checkpoint(epoch + 1, best=True)
                    patience_counter = 0 # Reset counter if improved
                else:
                    patience_counter += 1
            else:
                self._save_checkpoint(epoch + 1, best=False)
                # Note: if not save_best_only, logic for "patience" is ambiguous.
                # Assuming patience is based on valid accuracy improvement regardless of saving strategy.
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            # Check Early Stopping
            if cfg.patience > 0 and patience_counter >= cfg.patience:
                print(f"Early stop triggered! No improvement for {cfg.patience} epochs.")
                break

        # Final test evaluation (only if there is test data)
        final_test_loss = 0.0
        final_test_acc = 0.0
        
        if len(test_loader.dataset) > 0:
            final_test_loss, final_test_acc = self.evaluate(test_loader, criterion, device)
            print(f"Final Test Performance - Loss: {final_test_loss:.4f}, Acc: {final_test_acc:.2f}%")
        else:
            print("No test data available for final evaluation.")
        
        return best_val_acc, final_test_acc

    def evaluate(self, data_loader, criterion, device=None):
        """Evaluate model on a given data loader."""
        if device is None:
            device = self.device
        
        # 检查DataLoader是否为空
        if len(data_loader.dataset) == 0:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch["X"].to(device)
                batch_labels = batch["y"]
                
                # 处理多标签情况
                if isinstance(batch_labels, (tuple, list)) and len(batch_labels) == 2:
                    # 分别移动两个标签到设备
                    polarity_labels = batch_labels[0].to(device)
                    clarity_labels = batch_labels[1].to(device)
                    labels = (polarity_labels, clarity_labels)
                else:
                    # 单标签情况
                    labels = batch_labels.to(device)
                
                outputs = self.model(inputs)
                
                # 提取正确的输出用于损失计算
                if isinstance(outputs, (tuple, list)):
                    loss_output = outputs[self.config.output_index] if self.config.output_index is not None else outputs
                else:
                    loss_output = outputs
                
                # 稳健性处理
                curr_labels = labels
                if not isinstance(loss_output, (tuple, list)):
                    if loss_output.ndim == 2 and loss_output.shape[1] == 1 and curr_labels.ndim == 1:
                        curr_labels = curr_labels.unsqueeze(1).float()
                    elif isinstance(criterion, (nn.BCEWithLogitsLoss, nn.BCELoss)):
                        curr_labels = curr_labels.float()
                        if curr_labels.ndim == 1:
                            curr_labels = curr_labels.unsqueeze(1)
                
                # 计算损失
                if self._loss_takes_inputs:
                    loss = criterion(loss_output, curr_labels, inputs=inputs)
                else:
                    loss = criterion(loss_output, curr_labels)
                total_loss += loss.item()
                
                # 计算准确率
                if isinstance(outputs, (tuple, list)):
                    metric_out = outputs[self.config.metric_index]
                else:
                    metric_out = outputs
                if isinstance(labels, (tuple, list)) and len(labels) == 2:
                    current_labels = labels[0]
                else:
                    current_labels = labels

                if metric_out.ndim == 2 and metric_out.shape[1] == 1:
                    predicted = (torch.sigmoid(metric_out) > 0.5).long().view(-1)
                else:
                    _, predicted = torch.max(metric_out.data, 1)
                
                total += current_labels.size(0)
                correct += (predicted == current_labels.view(-1)).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total if total else 0.0
        
        return avg_loss, accuracy

    def _save_checkpoint(self, epoch: int, best: bool = False):
        tag = "best" if best else f"epoch_{epoch}"
        path = f"{self.config.checkpoint_dir}/datasets_{tag}.pth"
        torch.save(self.model.state_dict(), path)
        print(f"Saved checkpoint to {path}")
