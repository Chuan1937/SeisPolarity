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
from seispolarity.generate import Normalize, ChangeDtype, FixedWindow


@dataclass
class TrainingConfig:
    h5_path: str
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 1e-3
    num_workers: int = 0
    train_val_split: float = 0.8
    limit: Optional[int] = None
    p0: int = 100
    windowlen: int = 400
    picker_p: Optional[int] = None  # P拾取点位置，用于约束或参考
    label_key: str = "label"
    device: Optional[str] = None
    checkpoint_dir: str = "."
    save_best_only: bool = True
    patience: int = -1 # Early stopping patience. -1 means disabled.
    resume_checkpoint: Optional[str] = None


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

    def __init__(self, metadata_key: str = "label", key: str = "y"):
        self.metadata_key = metadata_key
        self.key = key

    def __call__(self, state_dict):
        _, metadata = state_dict["X"]
        label = np.array(metadata[self.metadata_key])
        state_dict[self.key] = (label, None)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataset,
        config: TrainingConfig,
        extra_augmentations: Optional[List[Callable]] = None,
        val_dataset=None,
    ):
        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = default_device(config)
        self.extra_augmentations = extra_augmentations or []
        
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

    def _log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Log epoch stats to file."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_str = f"{epoch:<6} | {train_loss:<12.4f} | {train_acc:<10.2f} | {val_loss:<12.4f} | {val_acc:<10.2f} | {timestamp:<20}\n"
        with open(self.log_path, "a") as f:
            f.write(log_str)

    def _build_loaders(self):
        cfg = self.config

        if self.val_dataset is not None:
            # Use provided explicit split
            train_dataset = self.dataset
            val_dataset = self.val_dataset
        else:
            # Split dataset
            train_size = int(cfg.train_val_split * len(self.dataset))
            val_size = len(self.dataset) - train_size
            train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

        # Generators
        train_gen = GenericGenerator(train_dataset)
        val_gen = GenericGenerator(val_dataset)

        augmentations = [
            FixedWindow(p0=cfg.p0, windowlen=cfg.windowlen),
            Normalize(),
            MetadataToLabel(metadata_key=cfg.label_key),
            ChangeDtype(np.float32, "X"),
            ChangeDtype(np.longlong, "y"),
        ] + self.extra_augmentations

        train_gen.add_augmentations(augmentations)
        val_gen.add_augmentations(augmentations)

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
        return train_loader, val_loader

    def train(self):
        cfg = self.config
        device = self.device
        print(f"Using device: {device}")

        train_loader, val_loader = self._build_loaders()

        self.model.to(device)

        if cfg.resume_checkpoint:
            if Path(cfg.resume_checkpoint).exists():
                print(f"Resuming training from checkpoint: {cfg.resume_checkpoint}")
                self.model.load_state_dict(torch.load(cfg.resume_checkpoint, map_location=device))
            else:
                print(f"Checkpoint {cfg.resume_checkpoint} not found. Starting from scratch.")

        criterion = nn.CrossEntropyLoss()
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
                _, predicted = torch.max(outputs.data, 1)
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
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_loss /= len(val_loader)
            val_acc = 100 * correct / total if total else 0.0

            print(
                f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Write to log file
            self._log_epoch(epoch+1, train_loss, train_acc, val_loss, val_acc)

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

        return best_val_acc

    def _save_checkpoint(self, epoch: int, best: bool = False):
        tag = "best" if best else f"epoch_{epoch}"
        path = f"{self.config.checkpoint_dir}/scsn_{tag}.pth"
        torch.save(self.model.state_dict(), path)
        print(f"Saved checkpoint to {path}")
