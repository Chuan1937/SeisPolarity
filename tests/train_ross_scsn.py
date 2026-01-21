import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
from seispolarity.data.base import WaveformDataset
from seispolarity.models.scsn import SCSN
from seispolarity.training import Trainer, TrainingConfig

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 配置

# ==================== 数据集路径配置 ====================
# 方式1: 使用本地文件 (当前使用)
# DATA_PATH = r"/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5"
#
# 方式2: 使用自动下载 (取消注释下面的代码)
# from seispolarity import get_dataset_path
# DATA_PATH = get_dataset_path(
#     dataset_name="SCSN",
#     subset="train",  # 训练集
#     cache_dir="./datasets_download",  # 自定义缓存目录
#     use_hf=False  # 默认使用 ModelScope，设为 True 使用 Hugging Face
# )

# 当前使用本地文件路径
DATA_PATH = r"/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5"
OUT_DIR = "./checkpoints_ross_scsn"

# 训练参数
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
NUM_WORKERS = 4

# 数据参数
PRELOAD = True
ALLOWED_LABELS = [0, 1, 2]
CROP_LEFT = 200  # p_pick左侧裁剪长度（p_pick=300, 开始点=100）
CROP_RIGHT = 200  # p_pick右侧裁剪长度（结束点=500）

# 创建数据集
dataset = WaveformDataset(
    path=DATA_PATH,
    name="SCSN_Train",
    preload=PRELOAD,
    allowed_labels=ALLOWED_LABELS,
    data_key="X",
    label_key="Y",
    clarity_key=None,
    pick_key=None,  # SCSN数据集没有p_pick字段
    metadata_keys=[],
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
    label_key="label", 
    train_val_split=0.8,  # 训练集比例
    val_split=0.1,        # 验证集比例
    test_split=0.1,       # 测试集比例
    patience=5,
    random_seed=36  # 设置随机种子以确保可复现性
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

# 清理
if not PRELOAD and hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'close'):
    dataset.dataset.close()