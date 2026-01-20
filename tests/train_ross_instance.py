import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import numpy as np
from seispolarity import WaveformDataset, Demean, Normalize, RandomTimeShift,PolarityInversion
from seispolarity.models.scsn import SCSN
from seispolarity.training import Trainer, TrainingConfig

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 配置
DATA_PATH = r"/home/yuan/code/SeisPolarity/datasets/Instance/Instance_polarity.hdf5"
OUT_DIR = "./checkpoints_ross_instance"

# 训练参数
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
NUM_WORKERS = 4

# 数据参数
PRELOAD = True
ALLOWED_LABELS = [0, 1, 2]
CROP_LEFT = 200  # p_pick左侧裁剪200个样本点（2秒）
CROP_RIGHT = 200  # p_pick右侧裁剪200个样本点（2秒）

# 设置随机种子以确保可复现性
RANDOM_SEED = 36
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 数据增强流水线（与其他训练文件格式一致）
train_augmentations = [
    RandomTimeShift(key="X", max_shift=5, mode="reflect"),
    Demean(key="X", axis=-1),
    Normalize(key="X", amp_norm_axis=-1, amp_norm_type="std"),
]

# 创建原始数据集（与其他训练文件格式一致）
dataset = WaveformDataset(
    path=DATA_PATH,
    name="Instance_Train",
    preload=PRELOAD,
    allowed_labels=ALLOWED_LABELS,
    data_key="X",
    label_key="Y",
    clarity_key=None,
    pick_key="p_pick",  # 使用p_pick作为P波到达点
    pick_position=None,
    metadata_keys=[],
    crop_left=CROP_LEFT,      # p_pick左侧裁剪长度
    crop_right=CROP_RIGHT,    # p_pick右侧裁剪长度
)

dataset.add_augmentations(train_augmentations)
# 添加极性反转增强到数据集
polarity_inversion = PolarityInversion(key="X", label_key="label", label_map={'U': 0, 'D': 1, 'X': 2})
dataset.add_augmentation(polarity_inversion)

# 训练配置
config = TrainingConfig(
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LR,
    num_workers=NUM_WORKERS,
    device=DEVICE,
    checkpoint_dir=OUT_DIR,
    label_key="label", 
    train_val_split=0.8,  # 训练集比例
    val_split=0.1,        # 验证集比例
    test_split=0.1,       # 测试集比例
    patience=5,
    random_seed=RANDOM_SEED
)

# 创建输出目录
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# 初始化模型和训练器
model = SCSN(num_fm_classes=3)
trainer = Trainer(model=model, dataset=dataset, val_dataset=None, test_dataset=None, config=config)

# 开始训练
best_val_acc, final_test_acc = trainer.train()

# 保存最终模型
final_model_path = Path(OUT_DIR) / "final_model.pth"
torch.save(model.state_dict(), final_model_path)
print(f"\n模型已保存到: {final_model_path}")

# 清理
if not PRELOAD and hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'close'):
    dataset.dataset.close()
