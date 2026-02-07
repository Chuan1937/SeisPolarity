import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add project root to sys.path

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from seispolarity.inference import Predictor
from seispolarity.data.base import WaveformDataset

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ==================== 数据集路径配置 ====================
# 方式1: 使用本地文件 (当前使用)
# TEST_FILE = "/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5"
#
# 方式2: 使用自动下载 (取消注释下面的代码)
# from seispolarity import get_dataset_path
# TEST_FILE = get_dataset_path(
#     dataset_name="SCSN",
#     subset="test",  # 测试集
#     cache_dir="./datasets_download",  # 自定义缓存目录
#     use_hf=False  # 默认使用 ModelScope，设为 True 使用 Hugging Face
# )

# 当前使用本地文件路径
TEST_FILE = "/mnt/f/AI_Seismic_Data/scsn/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5"
CROP_LEFT = 300  # p_pick左侧裁剪长度（p_pick=300, 开始点=0）
CROP_RIGHT = 300  # p_pick右侧裁剪长度（结束点=600）

# 网络下载模型
predictor = Predictor(model_name="EQPOLARITY_SCSN.pth",device=DEVICE)
# 本地加载模型示例
# predictor = Predictor(model_name="EQPOLARITY_SCSN.pth", model_path="/home/yuan/code/SeisPolarity/pretrained_model/EQPOLARITY/EQPOLARITY_SCSN.pth", device=DEVICE)

dataset = WaveformDataset(
    path=TEST_FILE,
    name="SCSN_Test",
    preload=True,
    allowed_labels=[0, 1],  # EQPolarity 是二分类：Up vs Down
    # SCSN数据使用标准键名：X为数据，Y为标签
    data_key="X",
    label_key="Y",
    clarity_key=None,
    pick_key=None,
    metadata_keys=[],  # SCSN不需要额外的元数据键
    p_pick_position=300,      # SCSN数据集的固定P波位置（第300个样本点）
    crop_left=CROP_LEFT,      # p_pick左侧裁剪长度
    crop_right=CROP_RIGHT     # p_pick右侧裁剪长度
)

loader = dataset.get_dataloader(
    batch_size=2048,  # Large batch size for inference
    num_workers=4 if not USE_PRELOAD else 0,  # 与训练时保持一致
    shuffle=False  # 推理时不需要打乱
)

probabilities, labels = predictor.predict_from_loader(loader, return_probs=True)
predictions = np.argmax(probabilities, axis=1)

# 计算混淆矩阵
cm = confusion_matrix(labels, predictions)

print("\n" + "="*60)
print("Eqpolarity模型预测结果 - 混淆矩阵")
print("="*60)

# 打印数值混淆矩阵
print(f"\n混淆矩阵 (样本数):")
print(f"真实标签\\预测标签 | 0 (Up) | 1 (Down)")
print("-" * 40)
for i in range(2):
    row_str = f"      {i} ({'Up' if i==0 else 'Down'})     |"
    for j in range(2):
        row_str += f" {cm[i, j]:7d} |"
    print(row_str)

# 计算准确率
accuracy = np.sum(np.diag(cm)) / np.sum(cm)
print(f"\n总体准确率: {accuracy:.4f}")

# 打印分类报告
print("\n分类报告:")
print(classification_report(labels, predictions, target_names=['Up', 'Down']))

