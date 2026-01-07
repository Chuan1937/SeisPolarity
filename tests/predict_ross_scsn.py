"""
Example script: Using the Unified Inference Interface to predict Polarity.
示例脚本: 使用统一推断接口进行极性预测。

This script demonstrates:
1. Auto-downloading the pretrained Ross(SCSN) model.
2. Loading test waveforms from HDF5.
3. Running predictions with minimal code.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add project root to sys.path

import h5py
import numpy as np
from seispolarity.inference import Predictor

# Path to test data (Adjust if needed)
# 测试数据路径 (如有需请修改)
TEST_FILE = "/mnt/c/Users/yuan/seispolarity/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5"

def main():
    print("Initializing Predictor...")
    # 1. Initialize Predictor (Auto-downloads model if missing)
    # 1. 初始化预测器 (如果模型缺失会自动下载)
    predictor = Predictor(model_name="ross")

    print(f"Loading data from {TEST_FILE}...")
    # 2. Load some sample data
    # 2. 加载一些样本数据
    try:
        with h5py.File(TEST_FILE, "r") as f:
            # Load first 100 waveforms
            # shape: (N, 600)
            waveforms = f["valid/data"][:100].astype(np.float32)
            labels = f["valid/label"][:100].astype(np.int64)
            
        print(f"Loaded {len(waveforms)} waveforms. Shape: {waveforms.shape}")
        
    except FileNotFoundError:
        print(f"Error: Test file not found at {TEST_FILE}")
        print("Creating dummy random data for demonstration...")
        waveforms = np.random.randn(100, 600).astype(np.float32)
        labels = None

    # 3. Predict
    # 3. 进行预测
    # predictor.predict handles cropping (600->400) and normalization automatically
    # predictor.predict 会自动处理裁剪(600->400)和归一化
    print("Running prediction...")
    predictions = predictor.predict(waveforms)
    probabilities = predictor.predict(waveforms, return_probs=True)

    # 4. Show results
    # 4. 显示结果
    print("\nResults (First 10 samples):")
    print(f"{'Index':<6} | {'True':<6} | {'Pred':<6} | {'Confidence':<10}")
    print("-" * 40)
    
    for i in range(10):
        true_label = labels[i] if labels is not None else "N/A"
        pred = predictions[i]
        conf = probabilities[i][pred]
        
        # Map: 0=Up, 1=Down, 2=Unknown (Standard SCSN mapping usually)
        print(f"{i:<6} | {true_label:<6} | {pred:<6} | {conf:.4f}")

    if labels is not None:
        acc = (predictions == labels).mean() * 100
        print(f"\nBatch Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
