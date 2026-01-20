import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
from seispolarity.data.base import WaveformDataset
from seispolarity.models.rpnet import rpnet
from seispolarity.training import Trainer, TrainingConfig

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 配置
DATA_PATH = r"/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5"
OUT_DIR = "./checkpoints_rpnet_scsn"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# 数据集 (RPNet 需要 400 个采样点)
dataset = WaveformDataset(
    path=DATA_PATH,
    name="SCSN_Train",
    preload=True,
    allowed_labels=[0, 1],
    data_key="X",
    label_key="Y",
    p_pick_position=300,
    crop_left=200,   # 300 - 200 = 100
    crop_right=200   # 300 + 200 = 500 (共 400 个采样点)
)

# 训练配置
config = TrainingConfig(
    batch_size=128,  # RPNet 较大，调小一些 batch_size 避免 OOM
    epochs=50,
    learning_rate=1e-4, # RPNet 这种深层网络建议小学习率
    num_workers=4,
    device=DEVICE,
    checkpoint_dir=OUT_DIR,
    label_key="label", 
    train_val_split=0.8,
    val_split=0.1,
    test_split=0.1,
    patience=10,
    random_seed=36,
    loss_fn=nn.CrossEntropyLoss()
)

# 模型与训练
model = rpnet(sample_rate=100.0)
trainer = Trainer(model=model, dataset=dataset, config=config)

print("\nStarting RPNet training...")
best_val_acc, final_test_acc = trainer.train()

# 保存
final_model_path = Path(OUT_DIR) / "rpnet_final.pth"
torch.save({'model_state_dict': model.state_dict(), 'config': config}, final_model_path)
print(f"RPNet 模型已保存到: {final_model_path}")

if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'close'):
    dataset.dataset.close()
