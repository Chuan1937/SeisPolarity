import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) 
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from seispolarity.inference import Predictor
from seispolarity.data.base import WaveformDataset
from seispolarity.generate import (
    Demean, 
    Normalize, 
    DifferentialFeatures, 
    ChangeDtype
)

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 模型路径 - 使用训练好的模型
LOCAL_MODEL_PATH = r"/home/yuan/code/SeisPolarity/pretrained_model/DiTingMotion/DiTingMotion_DitingScsn.pth"

# 测试数据路径
TEST_FILE = "/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5"
WINDOW_P0 = 236      
WINDOW_LEN = 128   

# 数据增强（与训练时验证集一致）
test_augmentations = [
    Demean(key="X", axis=-1),
    Normalize(key="X", amp_norm_axis=-1, amp_norm_type="std"),
    DifferentialFeatures(key="X", axis=-1, append=True),
    ChangeDtype(np.float32, key="X"),
]

# 强制输出U/D模式（不能有X）
FORCE_UD = True

predictor = Predictor(model_name="diting_motion", model_path=LOCAL_MODEL_PATH, device=DEVICE, force_ud=FORCE_UD)

# 创建数据集（与训练时一致的处理）
dataset = WaveformDataset(
    path=TEST_FILE,
    name="SCSN_Test",
    preload=True,
    allowed_labels=[0, 1, 2],  # U, D, X
    data_key="X",
    label_key="Y",
    clarity_key=None,
    pick_key=None,
    metadata_keys=[],
    window_p0=WINDOW_P0,      # 裁剪起始点
    window_len=WINDOW_LEN,    # 裁剪长度
    augmentations=test_augmentations  # 应用数据增强
)

# 创建数据加载器
loader = dataset.get_dataloader(
    batch_size=1024,
    num_workers=0,  # preload=True时使用0个worker
    shuffle=False
)

# 进行预测（显式传递force_ud参数）
print("\n进行预测（强制U/D模式）...")
probabilities, labels = predictor.predict_from_loader(loader, return_probs=True, force_ud=FORCE_UD)

# 手动应用force_ud逻辑（因为predict_from_loader在return_probs=True时返回原始概率）
predictions = np.argmax(probabilities, axis=1)

if FORCE_UD:
    # 应用force_ud：将X预测转换为U或D中概率较高的那个
    x_mask = predictions == 2
    if x_mask.any():
        # 对于预测为X的样本，选择U和D中概率较高的那个
        ud_probs = probabilities[x_mask, :2]  # 只取U和D的概率
        ud_preds = np.argmax(ud_probs, axis=1)
        predictions[x_mask] = ud_preds
        
        # 也更新概率，将X的概率重新分配给U和D
        # 保持U和D的相对比例，将X的概率按比例分配给U和D
        for i in np.where(x_mask)[0]:
            total_ud = probabilities[i, 0] + probabilities[i, 1]
            if total_ud > 0:
                probabilities[i, 0] = probabilities[i, 0] / total_ud
                probabilities[i, 1] = probabilities[i, 1] / total_ud
            probabilities[i, 2] = 0.0  # X的概率设为0


# 创建掩码：只选择真实标签为U或D的样本
ud_mask = (labels == 0) | (labels == 1)
labels_ud = labels[ud_mask]
predictions_ud = predictions[ud_mask]

# 保存预测结果
results_df = pd.DataFrame({
    'true_label': labels,
    'predicted_label': predictions,
    'prob_U': probabilities[:, 0],
    'prob_D': probabilities[:, 1],
    'prob_X': probabilities[:, 2]
})
