import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
from seispolarity.data.base import WaveformDataset
from seispolarity.models.diting_motioned import DitingMotioned
from seispolarity.training import Trainer, TrainingConfig

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 配置
SCSN_PATH = r"/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5"
Diting_path = r"/home/yuan/code/SeisPolarity/datasets/DiTing/DiTing_augmented_merge.hdf5"
OUT_DIR = "./checkpoints_diting_scsn"

# 训练参数
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
NUM_WORKERS = 4

#创建数据集
scsn = WaveformDataset()
