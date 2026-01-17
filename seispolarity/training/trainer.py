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
    output_index: int = 0 # Index of output to use for metrics if model returns tuple


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
        _, metadata = state_dict["X"]
        raw_label = metadata.get(self.metadata_key, None)
        numeric_label = self._to_numeric_label(raw_label) if raw_label is not None else np.array([-1], dtype=np.int64)
        state_dict[self.key] = (numeric_label, None)


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
            
            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset, [train_size, val_size, test_size]
            )

        # Generators
        train_gen = GenericGenerator(train_dataset)
        val_gen = GenericGenerator(val_dataset)
        test_gen = GenericGenerator(test_dataset)

        # 基础数据增强：将标签从metadata移动到'y'键
        base_augmentation = MetadataToLabel(metadata_key=cfg.label_key)
        
        # 训练集增强：基础增强 + 训练集特定增强
        train_augmentations = [base_augmentation] + self.train_augmentations
        train_gen.add_augmentations(train_augmentations)
        
        # 验证集增强：基础增强 + 验证集特定增强
        val_augmentations = [base_augmentation] + self.val_augmentations
        val_gen.add_augmentations(val_augmentations)
        
        # 测试集增强：基础增强 + 测试集特定增强
        test_augmentations = [base_augmentation] + self.test_augmentations
        test_gen.add_augmentations(test_augmentations)

        train_loader = DataLoader(
            train_gen,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
        val_loader = DataLoader(
            val_gen,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        test_loader = DataLoader(
            test_gen,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        return train_loader, val_loader, test_loader

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
                labels = batch["y"].to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                metric_out = outputs[cfg.output_index] if isinstance(outputs, (tuple, list)) else outputs
                _, predicted = torch.max(metric_out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
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
                    labels = batch["y"].to(device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    metric_out = outputs[cfg.output_index] if isinstance(outputs, (tuple, list)) else outputs
                    _, predicted = torch.max(metric_out.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_loss /= len(val_loader)
            val_acc = 100 * correct / total if total else 0.0

            # Test evaluation (optional, can be done less frequently to save time)
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            self.model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch["X"].to(device)
                    labels = batch["y"].to(device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    metric_out = outputs[cfg.output_index] if isinstance(outputs, (tuple, list)) else outputs
                    _, predicted = torch.max(metric_out.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            test_loss /= len(test_loader)
            test_acc = 100 * test_correct / test_total if test_total else 0.0

            print(
                f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
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

        # Final test evaluation
        final_test_loss, final_test_acc = self.evaluate(test_loader, criterion, device)
        print(f"Final Test Performance - Loss: {final_test_loss:.4f}, Acc: {final_test_acc:.2f}%")
        
        return best_val_acc, final_test_acc

    def evaluate(self, data_loader, criterion, device=None):
        """Evaluate model on a given data loader."""
        if device is None:
            device = self.device
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch["X"].to(device)
                labels = batch["y"].to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                metric_out = outputs[self.config.output_index] if isinstance(outputs, (tuple, list)) else outputs
                _, predicted = torch.max(metric_out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total if total else 0.0
        
        return avg_loss, accuracy

    def _save_checkpoint(self, epoch: int, best: bool = False):
        tag = "best" if best else f"epoch_{epoch}"
        path = f"{self.config.checkpoint_dir}/scsn_{tag}.pth"
        torch.save(self.model.state_dict(), path)
        print(f"Saved checkpoint to {path}")
