import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add project root to sys.path

import torch
import numpy as np
from seispolarity.inference import Predictor
from seispolarity.data.base import WaveformDataset

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 测试数据路径
TEST_FILE = "/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5"
WINDOW_P0 = 0      
WINDOW_LEN = 600   

LOCAL_MODEL_PATH = r"/home/yuan/code/SeisPolarity/pretrained_model/Eqpolarity/Eqpolarity_SCSN.pth"
predictor = Predictor(model_name="eqpolarity", model_path=LOCAL_MODEL_PATH, device=DEVICE)

USE_PRELOAD = True 

dataset = WaveformDataset(
    path=TEST_FILE,
    name="SCSN_Test",
    preload=USE_PRELOAD,
    allowed_labels=[0, 1],  # EQPolarity 是二分类：Up vs Down
    # SCSN数据使用标准键名：X为数据，Y为标签
    data_key="X",
    label_key="Y",
    clarity_key=None,
    pick_key=None,
    metadata_keys=[],  # SCSN不需要额外的元数据键
    window_p0=WINDOW_P0,      # 裁剪起始点
    window_len=WINDOW_LEN  # 裁剪长度
)

loader = dataset.get_dataloader(
    batch_size=2048,  # Large batch size for inference
    num_workers=4 if not USE_PRELOAD else 0,  # 与训练时保持一致
    shuffle=False  # 推理时不需要打乱
)

probabilities, labels = predictor.predict_from_loader(loader, return_probs=True)
predictions = np.argmax(probabilities, axis=1)

