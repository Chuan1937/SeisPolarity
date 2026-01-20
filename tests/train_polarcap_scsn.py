import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from seispolarity.data.base import WaveformDataset
from seispolarity.models.polarCAP import PolarCAP, PolarCAPLoss
from seispolarity.training import Trainer, TrainingConfig

# ==========================================
# 1. 配置参数
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DATA_PATH = r"/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5"
OUT_DIR = "./checkpoints_polarcap_scsn"

# 训练参数
EPOCHS = 50
BATCH_SIZE = 512
LR = 1e-3
NUM_WORKERS = 4

# 数据参数
ALLOWED_LABELS = [0, 1]  # PolarCAP 暂时处理二分类
CROP_LEFT = 32           # PolarCAP 输入长度为 64 (300-32 到 300+32)
CROP_RIGHT = 32

# ==========================================
# 2. 主程序
# ==========================================
if __name__ == "__main__":
    # 创建输出目录
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # 创建数据集
    dataset = WaveformDataset(
        path=DATA_PATH,
        name="SCSN_Train",
        preload=True,
        allowed_labels=ALLOWED_LABELS,
        data_key="X",
        label_key="Y",
        p_pick_position=300,
        crop_left=CROP_LEFT,
        crop_right=CROP_RIGHT
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
        train_val_split=0.8,
        val_split=0.1,
        test_split=0.1,
        patience=10,
        random_seed=36,
        loss_fn=PolarCAPLoss(), # 使用专门的 PolarCAPLoss
        output_index=None,      # 设置为 None 以便将所有输出 (dec, p) 传递给损失函数
        metric_index=1          # 使用第 2 个输出 (p) 进行准确率评估
    )

    # 初始化模型 (PolarCAP 输入通道为 1)
    model = PolarCAP()

    # 初始化标准训练器
    trainer = Trainer(model=model, dataset=dataset, config=config)

    # 开始训练
    print("\nStarting PolarCAP standard training on SCSN...")
    best_val_acc, final_test_acc = trainer.train()

    # 保存最终模型和配置
    final_model_path = Path(OUT_DIR) / "polarcap_final.pth"
    torch.save({'model_state_dict': model.state_dict(), 'config': config}, final_model_path)
    print(f"训练完成。最佳验证准确率: {best_val_acc:.2f}%, 最终测试准确率: {final_test_acc:.2f}%")

    # 清理
    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'close'):
        dataset.dataset.close()
