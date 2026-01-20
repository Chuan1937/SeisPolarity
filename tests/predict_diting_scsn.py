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
CROP_LEFT = 64  # p_pick左侧裁剪长度（p_pick=300, 开始点=236）
CROP_RIGHT = 64  # p_pick右侧裁剪长度（结束点=364）   

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
    p_pick_position=300,      # SCSN数据集的固定P波位置（第300个样本点）
    crop_left=CROP_LEFT,      # p_pick左侧裁剪长度
    crop_right=CROP_RIGHT,    # p_pick右侧裁剪长度
    augmentations=None  # 不在初始化时添加增强，后续使用点操作添加
)

# 使用点操作方式添加数据增强
dataset.add_augmentations(test_augmentations)

# 创建数据加载器
loader = dataset.get_dataloader(
    batch_size=800,
    num_workers=8,  # preload=True时使用0个worker
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


# 简洁的混淆矩阵输出
print("\n" + "="*60)
print("DiTingMotion模型预测结果 - 简洁混淆矩阵")
print("="*60)

if len(labels_ud) > 0:
    cm = confusion_matrix(labels_ud, predictions_ud, labels=[0, 1])
    print(f"\n混淆矩阵 (U/D样本):")
    print(f"真实标签\\预测标签 | 0 (U) | 1 (D)")
    print("-" * 40)
    for i in range(2):
        label_name = ['U', 'D'][i]
        row_str = f"      {i} ({label_name})     |"
        for j in range(2):
            row_str += f" {cm[i, j]:7d} |"
        print(row_str)
    
    accuracy = accuracy_score(labels_ud, predictions_ud)
    print(f"\n总体准确率 (U/D样本): {accuracy:.4f}")
    
    # 打印分类报告
    print("\n分类报告 (U/D样本):")
    print(classification_report(labels_ud, predictions_ud, target_names=['U', 'D']))
    
else:
    print("警告：没有找到U/D样本进行评估")

print("="*60)
