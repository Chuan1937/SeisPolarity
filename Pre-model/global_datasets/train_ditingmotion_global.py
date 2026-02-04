import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import torch
import torch.nn as nn
from seispolarity.data.base import WaveformDataset
from seispolarity.models.diting_motion import DitingMotion
from seispolarity.training import Trainer, TrainingConfig
from seispolarity import Normalize,RandomTimeShift,Demean,DifferentialFeatures,ChangeDtype,DitingMotionLoss
import numpy as np
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

crop_left = 64
crop_right = 64
ALLOWED_LABELS = [0, 1, 2]
# Txed Datasets
txed_path = r"/home/yuan/code/SeisPolarity/datasets/Txed/Txed_polarity.hdf5"

txed_datasets = WaveformDataset(
    path=txed_path,
    name="TXED",
    preload=True,
    allowed_labels=ALLOWED_LABELS,
    data_key="X",
    label_key="Y",
    pick_key="p_pick",  # 使用p_pick作为P波到达点
    clarity_key=None,
    p_pick_position=None,
    crop_left=crop_left,
    crop_right=crop_right
)

# 数据增强
# 1-20hz 带通滤波
# 归一化处理
txed_datasets_augmentations = [
     RandomTimeShift(key="X", max_shift=5, shift_unit="samples",p_pick_key="p_pick"),
    Demean(key="X", axis=-1),
    Normalize(key="X", amp_norm_axis=-1, amp_norm_type="std"),
    DifferentialFeatures(key="X", axis=-1, append=True),
    ChangeDtype(np.float32, key="X"),
]

txed_datasets.add_augmentations(txed_datasets_augmentations)    
txed_datasets.add_clarity_labels(clarity_value='K')

# SCSN Datasets
scsn_train_path = r"/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5"
scsn_test_path = r"/mnt/f/AI_Seismic_Data/scsn/scsn_p_2018_2023_6sec_0.5r_fm_test.hdf5"
scsn_train_datasets = WaveformDataset(
    path=scsn_train_path,
    name="SCSN_Train",
    preload=True,
    allowed_labels=ALLOWED_LABELS,
    data_key="X",
    label_key="Y",
    clarity_key=None,
    p_pick_position=300,
    crop_left=crop_left,
    crop_right=crop_right  
)

scsn_test_datasets = WaveformDataset(
    path=scsn_test_path,
    name="SCSN_Test",
    preload=True,
    allowed_labels=ALLOWED_LABELS,
    data_key="X",
    label_key="Y",
    clarity_key=None,
    p_pick_position=300,
    crop_left=crop_left,   # 300 - 200 = 100
    crop_right=crop_right   # 300 + 200 = 500 (共 400 个采样点)
)

scsn_train_augmentations = [
    RandomTimeShift(key="X", max_shift=5, shift_unit="samples", p_pick_key=None),
    Demean(key="X", axis=-1),
    Normalize(key="X", amp_norm_axis=-1, amp_norm_type="std"),
    DifferentialFeatures(key="X", axis=-1, append=True),
    ChangeDtype(np.float32, key="X"),
]

scsn_train_datasets.add_augmentations(scsn_train_augmentations)
scsn_train_datasets.add_clarity_labels(clarity_value='K')

scsn_test_augmentations = [
    RandomTimeShift(key="X", max_shift=5, shift_unit="samples", p_pick_key=None),
    Demean(key="X", axis=-1),
    Normalize(key="X", amp_norm_axis=-1, amp_norm_type="std"),
    DifferentialFeatures(key="X", axis=-1, append=True),
    ChangeDtype(np.float32, key="X"),
]

scsn_test_datasets.add_augmentations(scsn_test_augmentations)
scsn_test_datasets.add_clarity_labels(clarity_value='K')

scsn_datasets = scsn_train_datasets + scsn_test_datasets


# Instance Datasets
instance_datasets = r"/home/yuan/code/SeisPolarity/datasets/Instance/Instance_polarity.hdf5"
instance_datasets = WaveformDataset(
    path=instance_datasets,
    name="Instance",
    preload=True,
    allowed_labels=ALLOWED_LABELS,
    data_key="X",
    label_key="Y",
    clarity_key=None,
    p_pick_position=None,
    pick_key="p_pick",  # 使用p_pick作为P波到达点
    crop_left=crop_left,   # 300 - 200 = 100
    crop_right=crop_right   # 300 + 200 = 500 (共 400 个采样点)
)

instance_datasets_augmentations = [
     RandomTimeShift(key="X", max_shift=5, shift_unit="samples",p_pick_key="p_pick"),
    Demean(key="X", axis=-1),
    Normalize(key="X", amp_norm_axis=-1, amp_norm_type="std"),
    DifferentialFeatures(key="X", axis=-1, append=True),
    ChangeDtype(np.float32, key="X"),
]

instance_datasets.add_augmentations(instance_datasets_augmentations)  
instance_datasets.add_clarity_labels(clarity_value='K')      

# PNW Datasets
pnw_datasets = r"/home/yuan/code/SeisPolarity/datasets/PNW/pnw_polarity.hdf5"
pnw_datasets = WaveformDataset(
    path=pnw_datasets,
    name="PNW",
    preload=True,
    allowed_labels=ALLOWED_LABELS,
    data_key="X",
    label_key="Y",
    clarity_key=None,
    p_pick_position=None,
    pick_key="p_pick",  # 使用p_pick作为P波到达点
    crop_left=crop_left,   # 300 - 200 = 100
    crop_right=crop_right   # 300 + 200 = 500 (共 400 个采样点)
)

pnw_datasets_augmentations = [
     RandomTimeShift(key="X", max_shift=5, shift_unit="samples",p_pick_key="p_pick"),
    Demean(key="X", axis=-1),
    Normalize(key="X", amp_norm_axis=-1, amp_norm_type="std"),
    DifferentialFeatures(key="X", axis=-1, append=True),
    ChangeDtype(np.float32, key="X"),
]

pnw_datasets.add_augmentations(pnw_datasets_augmentations)  
pnw_datasets.add_clarity_labels(clarity_value='K')      

# DiTing Datasets
diting_dataset = r"/home/yuan/code/SeisPolarity/datasets/DiTing/DiTing_polarity.hdf5"
diting_datasets = WaveformDataset(
    path=diting_dataset,
    name="DiTing",
    preload=True,
    allowed_labels=ALLOWED_LABELS,
    data_key="X",
    label_key="Y",
    clarity_key="Z",
    p_pick_position=None,
    pick_key="p_pick",  # 使用p_pick作为P波到达点
    crop_left=crop_left,   # 300 - 200 = 100
    crop_right=crop_right   # 300 + 200 = 500 (共 400 个采样点)
)

diting_datasets_augmentations = [
     RandomTimeShift(key="X", max_shift=5, shift_unit="samples",p_pick_key="p_pick"),
    Demean(key="X", axis=-1),
    Normalize(key="X", amp_norm_axis=-1, amp_norm_type="std"),
    DifferentialFeatures(key="X", axis=-1, append=True),
    ChangeDtype(np.float32, key="X"),
]

diting_datasets.add_augmentations(diting_datasets_augmentations)  

datasets = txed_datasets + scsn_datasets + instance_datasets + pnw_datasets + diting_datasets

# 训练配置
config = TrainingConfig(
    batch_size=1024,
    epochs=100,
    learning_rate=1e-3,
    num_workers=4,
    device=DEVICE,
    checkpoint_dir="./checkpoints_ditingmotion_global",
    label_key="label", 
    train_val_split=0.8,  # 训练集比例
    val_split=0.1,        # 验证集比例
    test_split=0.1,       # 测试集比例
    patience=5,
    loss_fn=DitingMotionLoss(gamma=2.0, has_clarity_labels=False),
    output_index=None,     # 为 None 则将全部 8 个输出传递给 DitingMotionLoss
    metric_index=3,        # 使用 ofuse（索引 3）进行准确率评估
    random_seed=36  # 设置随机种子以确保可复现性
)

# 创建输出目录
Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

# 初始化模型和训练器
model = DitingMotion(input_channels=2, dropout_rate=0.15)
trainer = Trainer(model=model, dataset=datasets, val_dataset=None, test_dataset=None, config=config)

# 开始训练
best_val_acc, final_test_acc = trainer.train()

# 保存最终模型
final_model_path = Path(config.checkpoint_dir) / "final_model.pth"
torch.save(model.state_dict(), final_model_path)
