import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn

from seispolarity.data.base import WaveformDataset
from seispolarity.models.eqpolarity import EQPolarityCCT
from seispolarity.training import Trainer, TrainingConfig

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 配置
DATA_PATH = r"/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5"
OUT_DIR = "./checkpoints_eqpolarity_scsn"

# 训练参数
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-4
NUM_WORKERS = 4

# 数据参数
PRELOAD = True
ALLOWED_LABELS = [0, 1]
CROP_LEFT = 300  # p_pick左侧裁剪长度（p_pick=300, 开始点=0）
CROP_RIGHT = 300  # p_pick右侧裁剪长度（结束点=600）

# 创建数据集
dataset = WaveformDataset(
    path=DATA_PATH,
    name="SCSN_Train",
    preload=PRELOAD,
    allowed_labels=ALLOWED_LABELS,
    # SCSN数据使用标准键名：X为数据，Y为标签
    data_key="X",
    label_key="Y",
    clarity_key=None,
    pick_key=None,  # SCSN数据集没有p_pick字段
    metadata_keys=[],  # SCSN不需要额外的元数据键
    p_pick_position=300,      # SCSN数据集的固定P波位置（第300个样本点）
    crop_left=CROP_LEFT,      # p_pick左侧裁剪长度
    crop_right=CROP_RIGHT     # p_pick右侧裁剪长度
)

# 训练配置
config = TrainingConfig(
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LR,
    num_workers=NUM_WORKERS,
    device=DEVICE,
    checkpoint_dir=OUT_DIR,
    label_key="label",  # MetadataToLabel会从元数据中提取'label'字段
    train_val_split=0.9,  # 训练集比例
    val_split=0.1,        # 验证集比例
    test_split=0.0,       # 测试集比例
    patience=5,
    random_seed=42  # 设置随机种子以确保可复现性
)

# 创建输出目录
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# 初始化模型和训练器
model = EQPolarityCCT(input_length=600)

# 适配输出层为2类
in_features = model.output_layer.in_features
model.output_layer = nn.Linear(in_features, 2)

trainer = Trainer(model=model, dataset=dataset, val_dataset=None, test_dataset=None, config=config)

# 开始训练
best_val_acc, final_test_acc = trainer.train()

# 保存最终模型
final_model_path = Path(OUT_DIR) / "final_model.pth"
torch.save(model.state_dict(), final_model_path)

# 清理
if not PRELOAD and hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'close'):
    dataset.dataset.close()
