import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from seispolarity.data.base import WaveformDataset
from seispolarity.models.scsn import SCSN
from seispolarity.training import Trainer, TrainingConfig

# 配置
DATA_PATH = r"/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5"
OUT_DIR = "./checkpoints_ross_scsn"

# 训练参数
EPOCHS = 2
BATCH_SIZE = 128
LR = 1e-3
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 数据参数
PRELOAD = True
ALLOWED_LABELS = [0, 1, 2]
MAX_SAMPLES = None
P0 = 100
WINDOWLEN = 400

# 创建数据集
dataset = WaveformDataset(
    path=DATA_PATH,
    name="SCSN_Train",
    preload=PRELOAD,
    allowed_labels=ALLOWED_LABELS
)

# 限制样本数（快速测试）
if MAX_SAMPLES and len(dataset) > MAX_SAMPLES:
    from torch.utils.data import Subset
    import numpy as np
    indices = np.random.choice(len(dataset), MAX_SAMPLES, replace=False)
    dataset = Subset(dataset, indices)

# 训练配置
config = TrainingConfig(
    h5_path=DATA_PATH,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LR,
    num_workers=NUM_WORKERS,
    p0=P0,
    windowlen=WINDOWLEN,
    picker_p=300,
    device=DEVICE,
    checkpoint_dir=OUT_DIR,
    label_key="label",
    train_val_split=0.9,
    patience=2
)

# 创建输出目录
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# 初始化模型和训练器
model = SCSN(num_fm_classes=3)
trainer = Trainer(model=model, dataset=dataset, val_dataset=None, config=config)

# 开始训练
if __name__ == "__main__":
    best_acc = trainer.train()
    
    # 保存最终模型
    final_model_path = Path(OUT_DIR) / "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    
    # 清理
    if not PRELOAD and hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'close'):
        dataset.dataset.close()