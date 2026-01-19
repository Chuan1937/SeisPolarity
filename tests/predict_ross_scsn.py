import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) 
import numpy as np
from seispolarity.inference import Predictor
from seispolarity.data.base import WaveformDataset
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 测试数据路径 (如有需请修改)
TEST_FILE = "/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5"
CROP_LEFT = 200  # p_pick左侧裁剪长度（p_pick=300, 开始点=100）
CROP_RIGHT = 200  # p_pick右侧裁剪长度（结束点=500）

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
    pick_key=None,  # SCSN数据集没有p_pick字段
    metadata_keys=[],  # SCSN不需要额外的元数据键
    p_pick_position=300,      # SCSN数据集的固定P波位置（第300个样本点）
    crop_left=CROP_LEFT,      # p_pick左侧裁剪长度
    crop_right=CROP_RIGHT     # p_pick右侧裁剪长度
)


# 使用统一的 WaveformDataset 接口
loader = dataset.get_dataloader(
    batch_size=2048,  
    num_workers=4 if not USE_PRELOAD else 0, 
    shuffle=False
)

probabilities, labels = predictor.predict_from_loader(loader, return_probs=True)

predictions = np.argmax(probabilities, axis=1)

# 计算混淆矩阵
cm = confusion_matrix(labels, predictions)

print("\n" + "="*60)
print("Ross模型预测结果 - 混淆矩阵")
print("="*60)

# 打印数值混淆矩阵
print(f"\n混淆矩阵 (样本数):")
print(f"真实标签\\预测标签 | 0 (Up) | 1 (Down) | 2 (Unknown)")
print("-" * 50)
for i in range(3):
    label_name = ['Up', 'Down', 'Unknown'][i]
    row_str = f"      {i} ({label_name})     |"
    for j in range(3):
        row_str += f" {cm[i, j]:7d} |"
    print(row_str)

# 计算准确率
accuracy = np.sum(np.diag(cm)) / np.sum(cm)
print(f"\n总体准确率: {accuracy:.4f}")

# 打印分类报告
print("\n分类报告:")
print(classification_report(labels, predictions, target_names=['Up', 'Down', 'Unknown']))

