import pytest
import numpy as np
import copy
from seispolarity.generate import (
    Normalize, Filter, AddGap, GaussianNoise, ChangeDtype, 
    ChannelDropout, RandomArrayRotation, RotateHorizontalComponents,
    FixedWindow, SlidingWindow, RandomWindow,
    ProbabilisticLabeller, DetectionLabeller
)

@pytest.fixture
def sample_state_dict():
    # Create a dummy trace: 3 channels, 1000 samples
    # Shape (3, 1000) -> (Channels, Samples)
    data = np.random.randn(3, 1000).astype(np.float32)
    metadata = {
        "trace_sampling_rate_hz": 100,
        "trace_component_order": "ZNE",
        "trace_start_time": "2021-01-01T00:00:00.000000Z",
        "trace_p_arrival_sample": 500,
        "trace_s_arrival_sample": 800,
    }
    return {"X": (data, metadata)}

def test_normalize(sample_state_dict):
    aug = Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak")
    aug(sample_state_dict)
    data, _ = sample_state_dict["X"]
    assert np.abs(np.mean(data, axis=-1)).max() < 1e-5
    assert np.abs(data).max() <= 1.0 + 1e-5

def test_gaussian_noise(sample_state_dict):
    aug = GaussianNoise(scale=(0.1, 0.1))
    original_data = sample_state_dict["X"][0].copy()
    aug(sample_state_dict)
    data, _ = sample_state_dict["X"]
    assert not np.allclose(data, original_data)
    assert data.shape == original_data.shape

def test_change_dtype(sample_state_dict):
    aug = ChangeDtype(dtype=np.float64)
    aug(sample_state_dict)
    data, _ = sample_state_dict["X"]
    assert data.dtype == np.float64

def test_add_gap(sample_state_dict):
    aug = AddGap(axis=-1)
    aug(sample_state_dict)
    data, _ = sample_state_dict["X"]
    # Check if there is a sequence of zeros
    # This is probabilistic, but with default settings it should add a gap
    # We can check if there are any zeros that weren't there before, 
    # but random data is unlikely to be exactly zero.
    is_zero = np.all(data == 0, axis=0)
    assert np.any(is_zero)

def test_channel_dropout(sample_state_dict):
    aug = ChannelDropout(axis=0)
    aug(sample_state_dict)
    data, _ = sample_state_dict["X"]
    assert data.shape == (3, 1000)

def test_random_array_rotation(sample_state_dict):
    aug = RandomArrayRotation(axis=-1)
    original_data = sample_state_dict["X"][0].copy()
    aug(sample_state_dict)
    data, _ = sample_state_dict["X"]
    assert data.shape == original_data.shape

def test_rotate_horizontal_components(sample_state_dict):
    aug = RotateHorizontalComponents()
    aug(sample_state_dict)
    data, _ = sample_state_dict["X"]
    assert data.shape == (3, 1000)

def test_fixed_window(sample_state_dict):
    aug = FixedWindow(p0=100, windowlen=200)
    aug(sample_state_dict)
    data, metadata = sample_state_dict["X"]
    assert data.shape == (3, 200)
    assert metadata["trace_p_arrival_sample"] == 400  # 500 - 100

def test_sliding_window(sample_state_dict):
    aug = SlidingWindow(timestep=100, windowlen=200)
    aug(sample_state_dict)
    data, metadata = sample_state_dict["X"]
    # (1000 - 200) // 100 + 1 = 9 windows
    assert data.shape == (9, 3, 200)
    assert len(metadata["trace_p_arrival_sample"]) == 9

def test_random_window(sample_state_dict):
    aug = RandomWindow(windowlen=200, low=0, high=500)
    aug(sample_state_dict)
    data, metadata = sample_state_dict["X"]
    assert data.shape == (3, 200)

def test_probabilistic_labeller(sample_state_dict):
    # Note: Labeller usually expects 'y' key to write to, but reads from 'X'
    aug = ProbabilisticLabeller(label_columns={"trace_p_arrival_sample": "P", "trace_s_arrival_sample": "S"}, dim=0)
    aug(sample_state_dict)
    labels, _ = sample_state_dict["y"]
    # Labels: P, S, Noise -> 3 channels
    assert labels.shape == (3, 1000)
    # Check if labels sum to 1 (approximately)
    assert np.allclose(labels.sum(axis=0), 1.0)

def test_detection_labeller(sample_state_dict):
    aug = DetectionLabeller(p_phases=["trace_p_arrival_sample"], s_phases=["trace_s_arrival_sample"])
    aug(sample_state_dict)
    labels, _ = sample_state_dict["y"]
    # Detection labeller output shape: (1, 1000)
    assert labels.shape == (1, 1000)
