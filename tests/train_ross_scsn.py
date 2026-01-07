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


def main() -> int:
    # =========================================================================
    # 配置区域 (Configuration)
    # =========================================================================
    # 数据路径 (Set to None to auto-download from Hugging Face)
    TRAIN_H5 = r"/mnt/c/Users/yuan/seispolarity/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5"
    TEST_H5 = r"/mnt/c/Users/yuan/seispolarity/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5" # 原 VAL_H5 改为 TEST_H5，因为这是测试集
    OUT_DIR = "./checkpoints_ross_scsn"
    
    # 性能优化
    PRELOAD_RAM = True  # 是否将数据一次性加载到内存 (推荐 True，除非内存不足)
    NUM_WORKERS = 4     # 数据加载进程数
    
    # 训练超参数
    EPOCHS = 8
    PATIENCE = 5
    BATCH_SIZE = 3000
    LR = 1e-3
    LIMIT = None        # 测试用，例如 1000
    TRAIN_VAL_SPLIT = 0.9 # 90% 训练, 10% 验证 (从训练集中划分)
    
    # 模型与数据处理
    SCSN_P_PICK_INDEX = 300
    P0 = 100            # Ross paper: center ~300, +/-200 => start at 100 for length 400
    WINDOWLEN = 400
    DEVICE = None       # None (auto), "cuda", "cpu"
    # =========================================================================

    # 1. 准备数据集 (SCSNDataset 内部处理内存检查和预加载)
    # Note: SCSNDataset loads raw 600-sample traces.
    # The Trainer applies FixedWindow(p0=100, windowlen=400) augmentation,
    # which crops the center 400 samples (100-500). No conflict exists.
    
    # 加载训练集文件 (将被 Trainer 划分为 Train/Val)
    train_full_ds = SCSNDataset(TRAIN_H5, limit=LIMIT, preload=PRELOAD_RAM, split="train")
    
    # 加载测试集文件 (仅用于最终测试，不参与训练过程中的验证)
    test_ds = SCSNDataset(TEST_H5, limit=LIMIT, preload=PRELOAD_RAM, split="test")

    # 2. 配置 Trainer
    config = TrainingConfig(
        h5_path=str(TRAIN_H5), 
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LR,
        num_workers=NUM_WORKERS,
        limit=LIMIT,
        p0=P0,
        windowlen=WINDOWLEN,
        picker_p=SCSN_P_PICK_INDEX,
        device=DEVICE,
        checkpoint_dir=OUT_DIR,
        label_key="label",
        train_val_split=TRAIN_VAL_SPLIT # 设置划分比例
    )
    
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # 3. 初始化模型
    model = SCSN(num_fm_classes=3)
    
    # 4. 启动训练
    # 注意：val_dataset=None，让 Trainer 自动从 train_full_ds 中划分验证集
    trainer = Trainer(
        model=model,
        dataset=train_full_ds,
        val_dataset=None, 
        config=config
    )
    
    print("Starting training with unified Trainer API...")
    print(f"Splitting training data: {TRAIN_VAL_SPLIT*100}% Train, {(1-TRAIN_VAL_SPLIT)*100:.1f}% Validation")
    best_acc = trainer.train()
    print(f"Training finished. Best Validation Accuracy: {best_acc:.2f}%")
    
    # TODO: 可以在这里添加使用 test_ds 进行最终测试的代码
    print(f"Test dataset loaded with {len(test_ds)} samples. (Evaluation not implemented in this script)")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
