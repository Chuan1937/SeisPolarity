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
    DitingMotionLoss,
    Normalize, 
    Demean, 
    DifferentialFeatures, 
    ChangeDtype,
    GaussianNoise,
    OneOf,
    RandomTimeShift
)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 配置
SCSN_PATH_train = r"/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5"
SCSN_PATH_test = r"/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5"
Diting_path = r"/home/yuan/code/SeisPolarity/datasets/DiTing/DiTing_augmented_merge.hdf5"
OUT_DIR = "./checkpoints_diting_DitingScsn"

# 训练参数 - 匹配论文中的设置
EPOCHS = 100  # 论文中训练80个epoch
BATCH_SIZE = 800  # 论文中使用批量大小32
LR = 3e-4  # 论文中使用学习率0.0003
NUM_WORKERS = 8

# SCSN数据参数
PRELOAD = True
ALLOWED_LABELS = [0, 1, 2]
SCSN_P0 = 236
SCSN_WINDOWLEN = 128

# Diting数据参数
DITING_P0 = 0
DITING_WINDOWLEN = 128

# 训练数据增强流水线（包含随机增强）
train_augmentations = [
    RandomTimeShift(key="X", max_shift=5, mode="reflect"),
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
    window_len=SCSN_WINDOWLEN,  # 裁剪长度
    augmentations=train_augmentations  # 直接在数据集上应用训练数据增强
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
    window_len=SCSN_WINDOWLEN,  # 裁剪长度
    augmentations=val_augmentations  # 测试集使用验证数据增强
)

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
    window_len=DITING_WINDOWLEN,  # 裁剪长度
    augmentations=train_augmentations  # Diting数据集也使用训练数据增强
)

# 对scsn_dataset 添加clarity，全部设置为'K'
scsn_dataset_train.add_clarity_labels(clarity_value='K')
scsn_dataset_test.add_clarity_labels(clarity_value='K')

# 合并数据集
combined_dataset = scsn_dataset_train + scsn_dataset_test + diting_dataset

# 训练配置 - 使用DitingMotionLoss
config = TrainingConfig(
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LR,
    num_workers=NUM_WORKERS,
    device=DEVICE,
    checkpoint_dir=OUT_DIR,
    label_key="label", 
    train_val_split=0.75,  # 训练集比例
    val_split=0.10,        # 验证集比例
    test_split=0.15,       # 测试集比例
    patience=5, 
    loss_fn=DitingMotionLoss(gamma=2.0, has_clarity_labels=True),
    output_index=3,  # 使用ofuse（索引3）作为评估指标
    random_seed=36  # 设置随机种子以确保可复现性
)

# 创建输出目录
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# 初始化模型 - 输入通道数为2（原始波形 + 差分符号特征）
# 使用更低的dropout率（0.15）以防止过拟合
model = DitingMotion(input_channels=2, dropout_rate=0.15)

# 创建训练器，数据增强已在数据集层面处理
trainer = Trainer(
    model=model, 
    dataset=combined_dataset, 
    val_dataset=None, 
    test_dataset=None,
    config=config
  
)

# 开始训练
best_val_acc, final_test_acc = trainer.train()

# 保存最终模型
final_model_path = Path(OUT_DIR) / "final_model.pth"
torch.save(model.state_dict(), final_model_path)

# 清理
if not PRELOAD and hasattr(combined_dataset, 'dataset') and hasattr(combined_dataset.dataset, 'close'):
    combined_dataset.dataset.close()