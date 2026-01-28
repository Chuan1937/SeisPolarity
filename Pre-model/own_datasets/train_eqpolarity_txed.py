import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import torch
import torch.nn as nn
from seispolarity.data.base import WaveformDataset
from seispolarity.models.eqpolarity import EQPolarityCCT
from seispolarity.training import Trainer, TrainingConfig
from seispolarity import BandpassFilter, Normalize
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# from seispolarity import get_dataset_path
# DATA_PATH = get_dataset_path(
#     dataset_name="TXED",
#     subset="default",  # 训练集
#     cache_dir="./datasets_download",  # 自定义缓存目录
#     use_hf=False  # 默认使用 ModelScope，设为 True 使用 Hugging Face
# )
# from seispolarity.data.txed import TXED
# processor = TXED(
#         csv_path='/path/to/datasets/TXED/TXED.csv',
#         hdf5_path='/path/to/datasets/TXED/TXED.hdf5',
#         output_polarity='/path/to/datasets/TXED/'
#     )
# processor.process()

# 导入数据
txed_path = r"/home/yuan/code/SeisPolarity/datasets/Txed/txed_polarity.hdf5"

datasets = WaveformDataset(
    path=txed_path,
    name="TXED",
    preload=True,
    allowed_labels=[0, 1],
    data_key="X",
    label_key="Y",
    p_pick_position=300,
    crop_left=300,
    crop_right=300
)

# 数据增强
# 1-20hz 带通滤波
# 归一化处理
train_augmentations = [
    BandpassFilter(key="X", lowcut=1, highcut=20, fs=100, order=4),
    Normalize(key="X", amp_norm_axis=-1, amp_norm_type="std"),
]

datasets.add_augmentations(train_augmentations)

# 训练配置
config = TrainingConfig(
    batch_size=256,
    epochs=100,
    learning_rate=1e-3,
    num_workers=4,
    device=DEVICE,
    checkpoint_dir="./checkpoints_eqpolarity_txed",
    label_key="label", 
    train_val_split=0.8,  # 训练集比例
    val_split=0.1,        # 验证集比例
    test_split=0.1,       # 测试集比例
    patience=5,
    random_seed=36  # 设置随机种子以确保可复现性
)

# 创建输出目录
Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

# 初始化模型和训练器
model = EQPolarityCCT(input_length=600)

# 适配输出层为2类
in_features = model.output_layer.in_features
model.output_layer = nn.Linear(in_features, 2)

trainer = Trainer(model=model, dataset=datasets, val_dataset=None, test_dataset=None, config=config)

# 开始训练
best_val_acc, final_test_acc = trainer.train()

# 保存最终模型
final_model_path = Path(config.checkpoint_dir) / "final_model.pth"
torch.save(model.state_dict(), final_model_path)

# 清理
if not datasets.preload and hasattr(datasets, 'dataset') and hasattr(datasets.dataset, 'close'):
    datasets.dataset.close()
