import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) 
import numpy as np
from seispolarity.inference import Predictor
from seispolarity.data.base import WaveformDataset
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 测试数据路径 (如有需请修改)
TEST_FILE = "/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5"
WINDOW_P0 = 100
WINDOW_LEN = 400

LOCAL_MODEL_PATH = r"/home/yuan/code/SeisPolarity/pretrained_model/Ross/Ross_SCSN.pth"
predictor = Predictor(model_name="ross", model_path=LOCAL_MODEL_PATH, device=DEVICE)

USE_PRELOAD = True 

dataset = WaveformDataset(
    path=TEST_FILE,
    name="SCSN_Test",
    preload=USE_PRELOAD,
    allowed_labels=[0, 1, 2],
    # SCSN数据使用标准键名：X为数据，Y为标签
    data_key="X",
    label_key="Y",
    clarity_key=None,
    pick_key=None,
    metadata_keys=[],  # SCSN不需要额外的元数据键
    window_p0=WINDOW_P0,      # 裁剪起始点
    window_len=WINDOW_LEN  # 裁剪长度
)


# 使用统一的 WaveformDataset 接口
loader = dataset.get_dataloader(
    batch_size=2048,  
    num_workers=4 if not USE_PRELOAD else 0, 
    shuffle=False
)

probabilities, labels = predictor.predict_from_loader(loader, return_probs=True)

predictions = np.argmax(probabilities, axis=1)

