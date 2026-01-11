import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add project root to sys.path

import numpy as np
from seispolarity.inference import Predictor
from seispolarity.data.scsn import SCSNDataset

# 测试数据路径 (如有需请修改)
TEST_FILE = "/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5"
WINDOW_P0 = 100
WINDOW_LEN = 400

print("Initializing Predictor...")
# 1. Initialize Predictor
LOCAL_MODEL_PATH = r"/home/yuan/code/SeisPolarity/pretrained_model/Ross/Ross_SCSN.pth"
predictor = Predictor(model_name="ross", model_path=LOCAL_MODEL_PATH)

USE_PRELOAD = True 

print(f"Initializing Dataset (Preload={USE_PRELOAD})...")

dataset = SCSNDataset(h5_path=TEST_FILE, preload=USE_PRELOAD, split="test")

# 2. Setup Data Loader
# This automatically handles parallel loading, windowing (600->400), and normalization.
loader = dataset.get_dataloader(
    window_p0=WINDOW_P0,
    window_len=WINDOW_LEN,
    batch_size=2048,  # Large batch size for inference
    num_workers=8 if not USE_PRELOAD else 0
)

print(f"Starting prediction on {len(dataset)} samples...")


# 3. Stream & Predict (High-Level Interface)
# 使用 Predictor.predict_from_loader 处理整个 DataLoader 循环，更加简洁高效
# 相比手动循环，它在内部处理了 Tensor 转换、设备迁移和结果聚合

probabilities, labels = predictor.predict_from_loader(loader, return_probs=True)

predictions = np.argmax(probabilities, axis=1)

