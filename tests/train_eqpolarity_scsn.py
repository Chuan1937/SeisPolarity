"""Train EQPolarity model on SCSN HDF5 using unified Trainer API.

This script demonstrates:
1. Using EQPolarity (Transformer-based) model.
2. Training on filtered SCSN data (Labels 0 and 1 only, binary classification).
3. Adapting the binary model to 2 output neurons for compatibility with CrossEntropyLoss trainer.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from seispolarity.data.scsn import SCSNDataset
from seispolarity.models.eqpolarity import EQPolarityCCT
from seispolarity.training import Trainer, TrainingConfig

def main() -> int:
    # =========================================================================
    # 配置区域 (Configuration)
    # =========================================================================
    # 数据路径 (Set to None to auto-download from Hugging Face)
    TRAIN_H5 = r"/mnt/c/Users/yuan/seispolarity/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5"
    TEST_H5 = r"/mnt/c/Users/yuan/seispolarity/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5"
    OUT_DIR = "./checkpoints_eqpolarity_scsn"
    
    # 性能优化
    PRELOAD_RAM = True
    NUM_WORKERS = 4
    
    # 训练超参数
    EPOCHS = 10
    BATCH_SIZE = 600 # Transformer usually takes more memory, maybe lower batch size
    LR = 1e-4 # Transformer usually needs lower LR
    LIMIT = None
    TRAIN_VAL_SPLIT = 0.9

    # 模型与数据处理
    SCSN_P_PICK_INDEX = 300 # SCSN数据 P波理论位置
    
    # EQPolarity 使用完整的 600 点输入，不做额外裁剪
    # 因此 P0 设为 0，WINDOWLEN 设为 600
    P0 = 0
    WINDOWLEN = 600  
    
    DEVICE = None      
    
    # EQPolarity Specifics
    ALLOWED_LABELS = [0, 1] # Binary classification (Up vs Down)
    # =========================================================================

    # 1. 准备数据集 (Filtered for labels 0 and 1)
    # Note: SCSNDataset loads raw 600-sample traces.
    print(f"Initializing datasets. Keeping only labels {ALLOWED_LABELS}...")
    train_full_ds = SCSNDataset(TRAIN_H5, limit=LIMIT, preload=PRELOAD_RAM, split="train", allowed_labels=ALLOWED_LABELS)
    test_ds = SCSNDataset(TEST_H5, limit=LIMIT, preload=PRELOAD_RAM, split="test", allowed_labels=ALLOWED_LABELS)

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
        picker_p=SCSN_P_PICK_INDEX, # 传入P波拾取点
        device=DEVICE,
        checkpoint_dir=OUT_DIR,
        label_key="label",
        train_val_split=TRAIN_VAL_SPLIT
    )
    
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # 3. 初始化模型
    # Note: Input length is 400 (after cropping).
    model = EQPolarityCCT(input_length=WINDOWLEN)
    
    # [CRITICAL ADAPTATION]
    # The default EQPolarityCCT outputs 1 neuron (for BCE loss).
    # The generic Trainer uses CrossEntropyLoss which expects [Batch, NumClasses].
    # So we replace the final layer to output 2 neurons.
    in_features = model.output_layer.in_features
    model.output_layer = nn.Linear(in_features, 2)
    print(f"Model adapted for Binary CrossEntropy via 2-class logits. Output layer: {model.output_layer}")

    # 4. 启动训练
    trainer = Trainer(
        model=model,
        dataset=train_full_ds,
        val_dataset=None, 
        config=config
    )
    
    print("Starting training EQPolarity on SCSN...")
    best_acc = trainer.train()
    print(f"Training finished. Best Validation Accuracy: {best_acc:.2f}%")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
