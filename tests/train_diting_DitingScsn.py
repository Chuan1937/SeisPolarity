import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
import numpy as np
from seispolarity.data.base import WaveformDataset
from seispolarity.models.diting_motion import DitingMotion
from seispolarity.training import Trainer, TrainingConfig
from seispolarity.generate import (
    FocalLoss, 
    Normalize, 
    Demean, 
    DifferentialFeatures, 
    ChangeDtype,
    GaussianNoise,
    OneOf,
    RandomTimeShift
)

class MultiHeadFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weights=None):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma)
        # Weights for o3, o4, o5, ofuse. 
        self.weights = weights or [1.0, 1.0, 1.0, 1.0]

    def forward(self, inputs, targets):
        if isinstance(inputs, (tuple, list)) and len(inputs) >= 4:
            loss = 0.0
            # Only compute loss for the first 4 polarity outputs
            for i in range(4):
                loss += self.weights[i] * self.focal_loss(inputs[i], targets)
            return loss
        return self.focal_loss(inputs, targets)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 配置
SCSN_PATH_train = r"/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5"
SCSN_PATH_test = r"/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5"
Diting_path = r"/home/yuan/code/SeisPolarity/datasets/DiTing/DiTing_augmented_merge.hdf5"
OUT_DIR = "./checkpoints_diting_DitingScsn"

# 训练参数
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
NUM_WORKERS = 4

# SCSN数据参数
PRELOAD = True
ALLOWED_LABELS = [0, 1, 2]
SCSN_P0 = 236
SCSN_WINDOWLEN = 128

# 创建SCSN数据集
scsn_dataset_train = WaveformDataset(
    path=SCSN_PATH_train,
    name="SCSN_Train",
    preload=PRELOAD,
    allowed_labels=ALLOWED_LABELS,
    data_key="X",
    label_key="Y",
    clarity_key=None,
    pick_key=None,
    metadata_keys=[],
    window_p0=SCSN_P0,      # 裁剪起始点
    window_len=SCSN_WINDOWLEN  # 裁剪长度
)

scsn_dataset_test = WaveformDataset(
    path=SCSN_PATH_test,
    name="SCSN_Test",
    preload=PRELOAD,
    allowed_labels=ALLOWED_LABELS,
    data_key="X",
    label_key="Y",
    clarity_key=None,
    pick_key=None,
    metadata_keys=[],
    window_p0=SCSN_P0,      # 裁剪起始点
    window_len=SCSN_WINDOWLEN  # 裁剪长度
)


# Diting数据参数
DITING_P0 = 0
DITING_WINDOWLEN = 128

diting_dataset = WaveformDataset(
    path=Diting_path,
    name="Diting_Train",
    preload=PRELOAD,
    allowed_labels=ALLOWED_LABELS,  
    data_key="X",
    label_key="Y",
    clarity_key="Z",
    pick_key="p_pick",
    metadata_keys=[],
    window_p0=DITING_P0,      # 裁剪起始点
    window_len=DITING_WINDOWLEN  # 裁剪长度
)


# 对scsn_dataset 添加clarity，全部设置为'K'
scsn_dataset_train.add_clarity_labels(clarity_value='K')
scsn_dataset_test.add_clarity_labels(clarity_value='K')


# 合并数据集
combined_dataset = scsn_dataset_train + scsn_dataset_test + diting_dataset

# 训练数据增强流水线（包含随机增强）
train_augmentations = [
    RandomTimeShift(key="X", max_shift=25, mode="reflect"),
    Demean(key="X", axis=-1),
    Normalize(key="X", amp_norm_axis=-1, amp_norm_type="std"),
    DifferentialFeatures(key="X", axis=-1, append=True),
    ChangeDtype(np.float32, key="X"),
    
]

val_augmentations = [
    Demean(key="X", axis=-1),
    Normalize(key="X", amp_norm_axis=-1, amp_norm_type="std"),
    DifferentialFeatures(key="X", axis=-1, append=True),
    ChangeDtype(np.float32, key="X"),
]


test_augmentations = [
    Demean(key="X", axis=-1),
    Normalize(key="X", amp_norm_axis=-1, amp_norm_type="std"),
    DifferentialFeatures(key="X", axis=-1, append=True),
    ChangeDtype(np.float32, key="X"),
]

# 训练配置 - 使用FocalLoss
config = TrainingConfig(
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LR,
    num_workers=NUM_WORKERS,
    device=DEVICE,
    checkpoint_dir=OUT_DIR,
    label_key="label", 
    train_val_split=0.9,  # 训练集比例
    val_split=0.1,        # 验证集比例
    test_split=0.0,       # 测试集比例
    patience=5,
    loss_fn=MultiHeadFocalLoss(gamma=2.0),
    output_index=3
)

# 创建输出目录
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# 初始化模型 - 输入通道数为2（原始波形 + 差分符号特征）
model = DitingMotion(input_channels=2)

# 创建训练器，传入数据增强
trainer = Trainer(
    model=model, 
    dataset=combined_dataset, 
    val_dataset=None, 
    test_dataset=None,
    config=config,
    train_augmentations=train_augmentations,
    val_augmentations=val_augmentations,
    test_augmentations=test_augmentations
)

# 开始训练
best_val_acc, final_test_acc = trainer.train()

# 保存最终模型
final_model_path = Path(OUT_DIR) / "final_model.pth"
torch.save(model.state_dict(), final_model_path)

# 清理
if not PRELOAD and hasattr(combined_dataset, 'dataset') and hasattr(combined_dataset.dataset, 'close'):
    combined_dataset.dataset.close()