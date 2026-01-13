"""Train Ross (2018 JGR) polarity model on SCSN HDF5 using unified Trainer API.

This script uses the seispolarity.training.Trainer interface.
It supports:
1. Unified training loop (Trainer).
2. Parallel data loading (via num_workers).
3. Optional RAM preloading (preload=True) for maximum speed.
4. Explicit Train/Val split support.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from seispolarity.data.scsn import SCSNDataset
from seispolarity.models.scsn import SCSN
from seispolarity.training import Trainer, TrainingConfig



"""
Main execution entry point.
主执行入口函数。
"""
# =========================================================================
# 1. Configuration (配置参数)
# =========================================================================

# --- Paths (路径设置) ---
# Path to HDF5 files. Set to None to auto-download from Hugging Face.
# 设置为 None 可从 Hugging Face 自动下载默认数据。
TRAIN_H5 = r"/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5"
TEST_H5  = r"/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5"
OUT_DIR  = "./checkpoints_ross_scsn"

# --- Performance (性能优化) ---
PRELOAD_RAM = False  # Load all data into RAM for speed / 是否全量加载到内存

# Auto-configure workers:
# If Preload=True, usage of multiple workers adds overhead (forking RAM), so use 0.
# If Preload=False (Disk), use multiple workers to hide IO latency.
NUM_WORKERS = 8 if not PRELOAD_RAM else 0

# --- Hyperparameters (超参数) ---
EPOCHS          = 50
BATCH_SIZE      = 3000
LR              = 1e-3
PATIENCE        = 5
TRAIN_VAL_SPLIT = 0.9   # 90% Training / 10% Validation

# --- Data Processing (数据处理) ---
# Ross (2018) Settings:
# Input window: 400 samples (centered at P-arrival)
# 输入窗口: 400 采样点 (以 P 波到达为中心)
SCSN_P_PICK_INDEX = 300  # P-wave index in raw data / 原始数据中 P 波位置
WINDOWLEN         = 400  # Model input length / 模型输入长度
P0                = 100  # Start crop index / 裁剪起始点 (300 - 400/2 = 100)

DEVICE = None  # None(Auto), "cuda", "cpu"

# =========================================================================
# 2. Dataset Preparation (数据集准备)
# =========================================================================
print(f"{'='*40}\nInitializing Datasets / 初始化数据集\n{'='*40}")

# Load Training Data (Will be split into Train/Val later)
# 加载训练数据 (用于训练和验证)
train_full_ds = SCSNDataset(
    TRAIN_H5, 
    preload=PRELOAD_RAM, 
    split="train"
)

# Load Test Data (Held-out set)
# 加载测试数据 (仅用于最终测试，不参与训练)
test_ds = SCSNDataset(
    TEST_H5, 
    preload=PRELOAD_RAM, 
    split="test"
)

# =========================================================================
# 3. Setup Trainer (配置训练器)
# =========================================================================

# Create configuration object
# 创建训练配置对象
config = TrainingConfig(
    h5_path=str(TRAIN_H5), 
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LR,
    num_workers=NUM_WORKERS,
    p0=P0,
    windowlen=WINDOWLEN,
    picker_p=SCSN_P_PICK_INDEX,  # Pass P-pick index for reference / 传入P波拾取点
    device=DEVICE,
    checkpoint_dir=OUT_DIR,
    label_key="label",
    train_val_split=TRAIN_VAL_SPLIT,
    patience=PATIENCE
)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Initialize Model (Ross 2018 CNN)
# 初始化模型
model = SCSN(num_fm_classes=3)

# Initialize Trainer
# 初始化训练管理器
# val_dataset=None triggers auto-splitting based on train_val_split
# val_dataset=None 会触发基于 train_val_split 的自动划分
trainer = Trainer(
    model=model,
    dataset=train_full_ds,
    val_dataset=None, 
    config=config
)

# =========================================================================
# 4. Execution (开始训练)
# =========================================================================
print(f"\n{'='*40}\nStarting Training / 开始训练\n{'='*40}")
print(f"Model: Ross (SCSN) CNN")
print(f"Split: {TRAIN_VAL_SPLIT*100:.0f}% Train / {(1-TRAIN_VAL_SPLIT)*100:.0f}% Val")
print(f"Device: {config.device}")

best_acc = trainer.train()

print(f"\n{'='*40}\nTraining Finished / 训练完成\n{'='*40}")
print(f"Best Validation Accuracy: {best_acc:.2f}%")
print(f"Test Set Size: {len(test_ds)} (Evaluation pending)")

