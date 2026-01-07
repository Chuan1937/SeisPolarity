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

import numpy as np
from seispolarity.inference import Predictor
from seispolarity.data.scsn import SCSNDataset
from seispolarity.generate import FixedWindow, Normalize

# Path to test data (Adjust if needed)
# 测试数据路径 (如有需请修改)
TEST_FILE = "/mnt/c/Users/yuan/seispolarity/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5"
WINDOW_P0 = 100
WINDOW_LEN = 400
MAX_SAMPLES = 100

def load_waveforms(limit=MAX_SAMPLES):
    """Load and preprocess waveforms using existing dataset + augmentation utilities."""
    dataset = SCSNDataset(h5_path=TEST_FILE, limit=limit, preload=False)
    window = FixedWindow(p0=WINDOW_P0, windowlen=WINDOW_LEN)
    normalizer = Normalize(amp_norm_axis=-1, amp_norm_type="peak")

    waveforms = []
    labels = []

    for idx in range(len(dataset)):
        waveform, metadata = dataset[idx]
        # Apply shared FixedWindow + Normalize pipeline (same as training)
        state = {"X": (waveform, metadata)}
        window(state)
        normalizer(state)
        processed_waveform, processed_metadata = state["X"]

        waveforms.append(processed_waveform.squeeze())  # (1, 400) -> (400,)
        labels.append(processed_metadata.get("label", -1) if processed_metadata else -1)

    return np.stack(waveforms), np.array(labels)

def main():
    print("Initializing Predictor...")
    # 1. Initialize Predictor
    LOCAL_MODEL_PATH = None  # e.g. "./checkpoints_ross_scsn/scsn_best.pth"
    predictor = Predictor(model_name="ross", model_path=LOCAL_MODEL_PATH)

    print(f"Loading data from {TEST_FILE} via SCSNDataset + FixedWindow...")
    try:
        waveforms, labels = load_waveforms(limit=MAX_SAMPLES)
        print(f"Loaded {len(waveforms)} waveforms. Shape: {waveforms.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Predict
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
