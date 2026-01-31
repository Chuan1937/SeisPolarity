import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import torch
import torch.nn as nn
from seispolarity.data.base import WaveformDataset
from seispolarity.models.cfm import cfm
from seispolarity.training import Trainer, TrainingConfig
from seispolarity import BandpassFilter, Normalize, PolarityInversion, RandomTimeShift, Demean, Detrend, Stretching
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

crop_left = 80
crop_right = 80
ALLOWED_LABELS = [0, 1]
# Txed Datasets
txed_path = r"/home/yuan/code/SeisPolarity/datasets/Txed/txed_polarity.hdf5"

txed_datasets = WaveformDataset(
    path=txed_path,
    name="TXED",
    preload=True,
    allowed_labels=ALLOWED_LABELS,
    data_key="X",
    label_key="Y",
    pick_key="p_pick",  # 使用p_pick作为P波到达点
    p_pick_position=None,
    crop_left=crop_left,
    crop_right=crop_right
)

# 数据增强
# 1-20hz 带通滤波
# 归一化处理
txed_datasets_augmentations = [
    Demean(key="X", axis=-1),
    Detrend(key="X", axis=-1, type="linear"),
    # BandpassFilter(key="X", lowcut=1, highcut=20, fs=100, order=4),
    # Stretching(key="X", original_fs=100, stretch_factors=[2, 3], 
    #            target_samples=400, p_pick_key="p_pick", crop_left=crop_left, crop_right=crop_right),
    RandomTimeShift(key="X", max_shift=10, shift_unit="samples",p_pick_key="p_pick"),
    Normalize(key="X", amp_norm_axis=-1, amp_norm_type="std"),
    PolarityInversion(key="X", label_key="label", label_map={'U': 1, 'D': 1, 'X': 1})
]

txed_datasets.add_augmentations(txed_datasets_augmentations)    

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
    p_pick_position=300,
    crop_left=crop_left,   # 300 - 200 = 100
    crop_right=crop_right   # 300 + 200 = 500 (共 400 个采样点)
)

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
    p_pick_position=None,
    pick_key="p_pick",  # 使用p_pick作为P波到达点
    crop_left=crop_left,   # 300 - 200 = 100
    crop_right=crop_right   # 300 + 200 = 500 (共 400 个采样点)
)

instance_datasets_augmentations = [
    Demean(key="X", axis=-1),
    Detrend(key="X", axis=-1, type="linear"),
    # BandpassFilter(key="X", lowcut=1, highcut=20, fs=100, order=4),
    # Stretching(key="X", original_fs=100, stretch_factors=[2, 3], 
    #            target_samples=400, p_pick_key="p_pick", crop_left=crop_left, crop_right=crop_right),
    RandomTimeShift(key="X", max_shift=10, shift_unit="samples",p_pick_key="p_pick"),
    Normalize(key="X", amp_norm_axis=-1, amp_norm_type="std"),
    PolarityInversion(key="X", label_key="label", label_map={'U': 1, 'D': 1, 'X': 1})
]

instance_datasets.add_augmentations(instance_datasets_augmentations)  

# PNW Datasets
pnw_datasets = r"/home/yuan/code/SeisPolarity/datasets/PNW/pnw_polarity.hdf5"
pnw_datasets = WaveformDataset(
    path=pnw_datasets,
    name="PNW",
    preload=True,
    allowed_labels=ALLOWED_LABELS,
    data_key="X",
    label_key="Y",
    p_pick_position=None,
    pick_key="p_pick",  # 使用p_pick作为P波到达点
    crop_left=crop_left,   # 300 - 200 = 100
    crop_right=crop_right   # 300 + 200 = 500 (共 400 个采样点)
)

pnw_datasets_augmentations = [
    Demean(key="X", axis=-1),
    Detrend(key="X", axis=-1, type="linear"),
    # BandpassFilter(key="X", lowcut=1, highcut=20, fs=100, order=4),
    # Stretching(key="X", original_fs=100, stretch_factors=[2, 3], 
    #            target_samples=400, p_pick_key="p_pick", crop_left=crop_left, crop_right=crop_right),
    RandomTimeShift(key="X", max_shift=10, shift_unit="samples",p_pick_key="p_pick"),
    Normalize(key="X", amp_norm_axis=-1, amp_norm_type="std"),
    PolarityInversion(key="X", label_key="label", label_map={'U': 1, 'D': 1, 'X': 1})
]

pnw_datasets.add_augmentations(pnw_datasets_augmentations)  

# DiTing Datasets
diting_dataset = r"/home/yuan/code/SeisPolarity/datasets/DiTing/DiTing_polarity.hdf5"
diting_datasets = WaveformDataset(
    path=diting_dataset,
    name="DiTing",
    preload=True,
    allowed_labels=ALLOWED_LABELS,
    data_key="X",
    label_key="Y",
    p_pick_position=None,
    pick_key="p_pick",  # 使用p_pick作为P波到达点
    crop_left=crop_left,   # 300 - 200 = 100
    crop_right=crop_right   # 300 + 200 = 500 (共 400 个采样点)
)

diting_datasets_augmentations = [
    Demean(key="X", axis=-1),
    Detrend(key="X", axis=-1, type="linear"),
    # BandpassFilter(key="X", lowcut=1, highcut=20, fs=100, order=4),
    # Stretching(key="X", original_fs=100, stretch_factors=[2, 3], 
    #            target_samples=400, p_pick_key="p_pick", crop_left=crop_left, crop_right=crop_right),
    RandomTimeShift(key="X", max_shift=10, shift_unit="samples",p_pick_key="p_pick"),
    Normalize(key="X", amp_norm_axis=-1, amp_norm_type="std"),
    PolarityInversion(key="X", label_key="label", label_map={'U': 1, 'D': 1, 'X': 1})
]

diting_datasets.add_augmentations(diting_datasets_augmentations)  

datasets = txed_datasets + scsn_datasets + instance_datasets + pnw_datasets + diting_datasets

# 训练配置
config = TrainingConfig(
    batch_size=256,
    epochs=50,
    learning_rate=1e-3,
    num_workers=4,
    device=DEVICE,
    checkpoint_dir="./checkpoints_cfm_global",
    label_key="label", 
    train_val_split=0.8,
    val_split=0.1,
    test_split=0.1,
    patience=5,
    random_seed=36,
    loss_fn=nn.BCEWithLogitsLoss()
)

# 模型与训练
model = cfm(sample_rate=100.0)

trainer = Trainer(model=model, dataset=datasets, val_dataset=None, test_dataset=None, config=config)

# 开始训练
best_val_acc, final_test_acc = trainer.train()

# 保存最终模型
final_model_path = Path(config.checkpoint_dir) / "final_model.pth"
torch.save(model.state_dict(), final_model_path)

# 清理
if not datasets.preload and hasattr(datasets, 'dataset') and hasattr(datasets.dataset, 'close'):
    datasets.dataset.close()
