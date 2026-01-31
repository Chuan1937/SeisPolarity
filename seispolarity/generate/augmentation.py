import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

# ==============================================================================
# Signal Processing Augmentations (信号处理增强)
# ==============================================================================

class Demean:
    """Remove the mean from the waveform.
    
    对于地震波形数据，通常沿时间轴（axis=-1）去除均值。
    """
    def __init__(self, key="X", axis=-1):
        self.key = key
        self.axis = axis
    def __call__(self, state_dict):
        if self.key in state_dict:
            value = state_dict[self.key]
            
            # 处理元组情况：当value是(waveform, metadata)元组时
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            mean = np.mean(x, axis=self.axis, keepdims=True)
            x = x - mean
            
            # 根据原始类型返回
            if is_tuple:
                state_dict[self.key] = (x, meta)
            else:
                state_dict[self.key] = x

class Detrend:
    """Remove the linear trend from the waveform.
    
    对于地震波形数据，通常沿时间轴（axis=-1）去除线性趋势。
    使用 scipy.signal.detrend 函数去除线性趋势。
    
    Args:
        key: 状态字典中的键
        axis: 去趋势的轴，通常为 -1（时间轴）
        type: 去趋势类型，'linear' 或 'constant'
    """
    def __init__(self, key="X", axis=-1, type="linear"):
        self.key = key
        self.axis = axis
        self.type = type
    def __call__(self, state_dict):
        if self.key in state_dict:
            value = state_dict[self.key]
            
            # 处理元组情况：当value是(waveform, metadata)元组时
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            x = signal.detrend(x, axis=self.axis, type=self.type)
            
            # 根据原始类型返回
            if is_tuple:
                state_dict[self.key] = (x, meta)
            else:
                state_dict[self.key] = x

class Stretching:
    """Stretch the waveform by upsampling and cropping.
    
    通过上采样和截取实现数据拉伸，模拟低频信号。
    原始数据频率 5-15 Hz，拉伸 2-3 倍后频率约为 2-5 Hz。
    
    处理流程：
    1. 从原始采样率上采样到更高的采样率（如 100 Hz -> 200 Hz 或 300 Hz）
    2. 截取包含 P 波的固定长度采样点（默认 400 个采样点）
    
    Args:
        key: 状态字典中的键
        original_fs: 原始采样率，默认 100 Hz
        stretch_factors: 拉伸因子列表，如 [2, 3] 表示随机选择 2 倍或 3 倍拉伸
        target_samples: 截取的采样点数，默认 400
        p_pick_key: metadata 中 P 波到时的键名
        crop_left: P 波左侧截取的采样点数
        crop_right: P 波右侧截取的采样点数
    """
    def __init__(self, key="X", original_fs=100, stretch_factors=[2, 3], 
                 target_samples=400, p_pick_key="p_pick", crop_left=None, crop_right=None):
        self.key = key
        self.original_fs = original_fs
        self.stretch_factors = stretch_factors
        self.target_samples = target_samples
        self.p_pick_key = p_pick_key
        self.crop_left = crop_left
        self.crop_right = crop_right
    
    def __call__(self, state_dict):
        if self.key in state_dict:
            value = state_dict[self.key]
            
            # 处理元组情况：当value是(waveform, metadata)元组时
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            # 确保x至少是2D数组
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # 随机选择拉伸因子
            stretch_factor = np.random.choice(self.stretch_factors)
            
            # 计算目标采样率
            target_fs = self.original_fs * stretch_factor
            
            # 获取原始长度（时间点数）
            original_length = x.shape[-1]
            
            # 上采样到目标采样率
            original_time = np.linspace(0, original_length / self.original_fs, original_length)
            target_length = int(original_length * stretch_factor)
            target_time = np.linspace(0, original_length / self.original_fs, target_length)
            
            # 对每个通道进行插值
            stretched_data = np.zeros((x.shape[0], target_length))
            for i in range(x.shape[0]):
                stretched_data[i] = np.interp(target_time, original_time, x[i])
            
            # 截取包含 P 波的区域
            if meta is not None and self.p_pick_key in meta:
                p_pick = meta[self.p_pick_key]
                # 将原始采样点位置映射到拉伸后的位置
                p_pick_stretched = int(p_pick * stretch_factor)
                
                # 计算 crop_left 和 crop_right
                if self.crop_left is None:
                    crop_left = self.target_samples // 2
                else:
                    crop_left = int(self.crop_left * stretch_factor)
                
                if self.crop_right is None:
                    crop_right = self.target_samples // 2
                else:
                    crop_right = int(self.crop_right * stretch_factor)
                
                # 计算截取范围
                start_idx = p_pick_stretched - crop_left
                end_idx = p_pick_stretched + crop_right
                
                # 确保索引在有效范围内
                if start_idx < 0:
                    start_idx = 0
                    end_idx = min(target_length, self.target_samples)
                elif end_idx > target_length:
                    end_idx = target_length
                    start_idx = max(0, end_idx - self.target_samples)
                
                # 截取数据
                cropped_data = stretched_data[:, start_idx:end_idx]
                
                # 如果截取长度不足，进行填充
                if cropped_data.shape[-1] < self.target_samples:
                    pad_length = self.target_samples - cropped_data.shape[-1]
                    pad_left = pad_length // 2
                    pad_right = pad_length - pad_left
                    cropped_data = np.pad(cropped_data, 
                                          ((0, 0), (pad_left, pad_right)), 
                                          mode='constant')
                
                # 更新 metadata 中的采样率和 P 波到时
                meta['sampling_rate'] = target_fs
                meta[self.p_pick_key] = self.target_samples // 2  # 将 P 波设置在中间
                
                x = cropped_data
            else:
                # 如果没有 P 波信息，直接截取中间区域
                start_idx = (target_length - self.target_samples) // 2
                end_idx = start_idx + self.target_samples
                if start_idx < 0:
                    start_idx = 0
                    end_idx = min(target_length, self.target_samples)
                elif end_idx > target_length:
                    end_idx = target_length
                    start_idx = max(0, end_idx - self.target_samples)
                
                x = stretched_data[:, start_idx:end_idx]
                
                # 更新采样率
                if meta is not None:
                    meta['sampling_rate'] = target_fs
            
            # 根据原始类型返回
            if is_tuple:
                state_dict[self.key] = (x, meta)
            else:
                state_dict[self.key] = x

class BandpassFilter:
    """Apply bandpass or highpass filter to the waveform.
    
    使用巴特沃斯滤波器对波形进行滤波。
    当 highcut 为 None 时，实现高通滤波；否则实现带通滤波。
    
    Args:
        key: 状态字典中的键
        lowcut: 低截止频率
        highcut: 高截止频率，如果为None则实现高通滤波
        fs: 采样率，如果为None则从metadata中读取
        order: 滤波器阶数
        zerophase: 是否使用零相位滤波
    """
    def __init__(self, key="X", lowcut=1.0, highcut=20.0, fs=None, order=4, zerophase=True):
        self.key = key
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.zerophase = zerophase
    
    def __call__(self, state_dict):
        if self.key in state_dict:
            value = state_dict[self.key]
            
            # 处理元组情况：当value是(waveform, metadata)元组时
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            # 确定采样率
            if self.fs is None and meta is not None and 'sampling_rate' in meta:
                fs = meta['sampling_rate']
            elif self.fs is not None:
                fs = self.fs
            else:
                raise ValueError("Sampling rate must be provided either directly or in metadata")
            
            # 确保x至少是2D数组
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # 设计滤波器
            nyquist = 0.5 * fs
            
            if self.highcut is None:
                # 高通滤波
                high = self.lowcut / nyquist
                if high >= 1.0:
                    raise ValueError(f"Invalid cutoff frequency: {self.lowcut} Hz for sampling rate {fs} Hz")
                sos = signal.butter(self.order, high, btype='high', output='sos')
            else:
                # 带通滤波
                low = self.lowcut / nyquist
                high = self.highcut / nyquist
                if low >= 1.0 or high >= 1.0 or low >= high:
                    raise ValueError(f"Invalid frequency range: [{self.lowcut}, {self.highcut}] Hz for sampling rate {fs} Hz")
                sos = signal.butter(self.order, [low, high], btype='band', output='sos')
            
            # 对每个通道应用滤波
            filtered_data = np.zeros_like(x)
            for i in range(x.shape[0]):
                if self.zerophase:
                    filtered_data[i] = signal.sosfiltfilt(sos, x[i])
                else:
                    filtered_data[i] = signal.sosfilt(sos, x[i])
            
            # 根据原始类型返回
            if is_tuple:
                state_dict[self.key] = (filtered_data, meta)
            else:
                state_dict[self.key] = filtered_data

class Normalize:
    """Normalize the waveform amplitude."""
    def __init__(self, key="X", amp_norm_axis=-1, amp_norm_type="peak", eps=1e-12):
        self.key = key
        self.axis = amp_norm_axis
        self.type = amp_norm_type
        self.eps = eps
    def __call__(self, state_dict):
        if self.key in state_dict:
            value = state_dict[self.key]
            
            # 处理元组情况：当value是(waveform, metadata)元组时
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            if self.type == "std":
                mean = np.mean(x, axis=self.axis, keepdims=True)
                std = np.std(x, axis=self.axis, keepdims=True)
                x = (x - mean) / (std + self.eps)
            elif self.type == "peak":
                peak = np.max(np.abs(x), axis=self.axis, keepdims=True)
                x = x / (peak + self.eps)
            
            # 根据原始类型返回
            if is_tuple:
                state_dict[self.key] = (x, meta)
            else:
                state_dict[self.key] = x

class DifferentialFeatures:
    """Computes the differential of the waveform and its sign.
    
    对于地震波形数据，通常形状为 (C, N) 或 (N,)，其中：
    - C: 通道数（通常为1）
    - N: 时间点数量（如128）
    
    当append=True时，将差分符号特征作为新通道添加。
    例如：输入形状 (1, 128) -> 输出形状 (2, 128)
    """
    def __init__(self, key="X", axis=-1, append=True):
        self.key = key
        self.axis = axis
        self.append = append
    def __call__(self, state_dict):
        if self.key in state_dict:
            value = state_dict[self.key]
            
            # 处理元组情况：当value是(waveform, metadata)元组时
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            # 确保x至少是2D数组
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # 计算差分（沿时间轴）
            diff = np.diff(x, axis=self.axis)
            
            # 填充以保持原始长度（在开始处填充0）
            pad_shape = [(0, 0)] * x.ndim
            if self.axis == -1:
                idx = x.ndim - 1
            else:
                idx = self.axis
            pad_shape[idx] = (1, 0) # 在开始处填充1个0
            
            diff = np.pad(diff, pad_shape, mode='constant')
            sign_diff = np.sign(diff)
            
            if self.append:
                # 沿通道轴（axis=0）拼接
                new_x = np.concatenate([x, sign_diff], axis=0)
                # 根据原始类型返回
                if is_tuple:
                    state_dict[self.key] = (new_x, meta)
                else:
                    state_dict[self.key] = new_x
            else:
                # 当append=False时，将差分符号特征存储为新键
                if is_tuple:
                    # 如果原始是元组，我们需要保持元组结构
                    state_dict[self.key] = (x, meta)
                state_dict[self.key + "_diff_sign"] = sign_diff

class RandomTimeShift:
    """对 P 波到时标签进行随机扰动。
    
    对 metadata 中的 p_pick 进行±max_shift范围内的均匀随机平移。
    这样可以模拟真实情况中 P 波到时拾取的不确定性，使模型对 P 波对齐误差具有鲁棒性。
    
    注意：这个增强不会改变波形数据，只修改 metadata 中的 p_pick 标签。
    波形的裁剪会在后续的 WaveformDataset 中根据扰动后的 p_pick 进行。
    
    例如：采样率100Hz，±0.5s对应±50个采样点。
    
    Args:
        key: 状态字典中的键（用于获取 waveform 和 metadata）
        max_shift: 最大平移量
        shift_unit: 平移单位，'samples'（采样点）或 'seconds'（秒）
        sampling_rate: 采样率，当 shift_unit='seconds' 时使用
        p_pick_key: metadata 中 p_pick 的键名
        fixed_p_pick: 固定的 P 波到时（采样点数），当 metadata 中没有 p_pick_key 时使用
                      例如 SCSN 数据的 P 波固定在某个位置
    """
    def __init__(self, key="X", max_shift=5, shift_unit="samples", sampling_rate=100, p_pick_key="p_pick", fixed_p_pick=None):
        self.key = key
        self.max_shift = max_shift
        self.shift_unit = shift_unit
        self.sampling_rate = sampling_rate
        self.p_pick_key = p_pick_key
        self.fixed_p_pick = fixed_p_pick
    
    def __call__(self, state_dict):
        if self.key in state_dict:
            value = state_dict[self.key]
            
            # 处理元组情况：当value是(waveform, metadata)元组时
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            # 只修改metadata中的p_pick
            if meta is not None:
                # 获取 P 波到时：优先从 metadata 中读取，否则使用 fixed_p_pick
                if self.p_pick_key in meta:
                    p_pick = meta[self.p_pick_key]
                    has_p_pick_key = True
                elif self.fixed_p_pick is not None:
                    p_pick = self.fixed_p_pick
                    has_p_pick_key = False
                else:
                    # 既没有 p_pick_key 也没有 fixed_p_pick，不做任何操作
                    pass
                
                # 应用平移
                if has_p_pick_key or (self.fixed_p_pick is not None):
                    # 根据shift_unit计算随机平移量
                    if self.shift_unit == "seconds":
                        shift_seconds = np.random.uniform(-self.max_shift, self.max_shift)
                        shift_samples = int(shift_seconds * self.sampling_rate)
                    else:  # samples
                        shift_samples = np.random.randint(-self.max_shift, self.max_shift + 1)
                    
                    # 应用平移
                    if has_p_pick_key:
                        # 更新 metadata 中的 p_pick
                        meta[self.p_pick_key] = p_pick + shift_samples
                    else:
                        # 使用 fixed_p_pick，创建或更新 metadata 中的 p_pick
                        meta[self.p_pick_key] = p_pick + shift_samples
            
            # 根据原始类型返回
            if is_tuple:
                state_dict[self.key] = (x, meta)
            else:
                state_dict[self.key] = x

# ==============================================================================
# Data Type and Format Augmentations (数据类型和格式增强)
# ==============================================================================

class ChangeDtype:
    """Change the dtype of the data in the state dict."""
    def __init__(self, dtype, key="X"):
        self.dtype = dtype
        self.key = key
    def __call__(self, state_dict):
        if self.key in state_dict:
            value = state_dict[self.key]
            
            # 处理元组情况：当value是(waveform, metadata)元组时
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            if hasattr(x, "astype"):
                x = x.astype(self.dtype)
            
            # 根据原始类型返回
            if is_tuple:
                state_dict[self.key] = (x, meta)
            else:
                state_dict[self.key] = x

class Copy:
    """Copy a key in the state dict."""
    def __init__(self, source_key="X", target_key="X_copy"):
        self.source_key = source_key
        self.target_key = target_key
    def __call__(self, state_dict):
        if self.source_key in state_dict:
            value = state_dict[self.source_key]
            
            # 处理元组情况：当value是(waveform, metadata)元组时
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            if hasattr(x, "copy"):
                x_copy = x.copy()
            else:
                x_copy = x
            
            # 根据原始类型返回
            if is_tuple:
                state_dict[self.target_key] = (x_copy, meta)
            else:
                state_dict[self.target_key] = x_copy
                state_dict[self.target_key] = copy.deepcopy(x)

class FilterKeys:
    """Filter keys in the state dict, keeping only specified keys."""
    def __init__(self, keep_keys):
        self.keep_keys = keep_keys
    def __call__(self, state_dict):
        keys = list(state_dict.keys())
        for key in keys:
            if key not in self.keep_keys:
                del state_dict[key]

# ==============================================================================
# Noise and Distortion Augmentations (噪声和失真增强)
# ==============================================================================

class GaussianNoise:
    """Add Gaussian noise to the waveform."""
    def __init__(self, key="X", std=0.1):
        self.key = key
        self.std = std
    def __call__(self, state_dict):
        if self.key in state_dict:
            value = state_dict[self.key]
            
            # 处理元组情况：当value是(waveform, metadata)元组时
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            noise = np.random.normal(0, self.std, x.shape)
            x = x + noise
            
            # 根据原始类型返回
            if is_tuple:
                state_dict[self.key] = (x, meta)
            else:
                state_dict[self.key] = x

class ChannelDropout:
    """Randomly dropout channels (set to zero)."""
    def __init__(self, key="X", dropout_prob=0.1):
        self.key = key
        self.dropout_prob = dropout_prob
    def __call__(self, state_dict):
        if self.key in state_dict:
            value = state_dict[self.key]
            
            # 处理元组情况：当value是(waveform, metadata)元组时
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            if x.ndim >= 2:
                # 对每个通道应用独立的dropout
                mask = np.random.binomial(1, 1 - self.dropout_prob, size=x.shape[0])
                mask = mask.reshape(-1, *([1] * (x.ndim - 1)))
                x = x * mask
            
            # 根据原始类型返回
            if is_tuple:
                state_dict[self.key] = (x, meta)
            else:
                state_dict[self.key] = x

class PolarityInversion:
    """反转波形极性并相应地更新标签。
    
    这个增强专门用于极性分类任务：
    - 对于U（正向）标签：反转波形，标签变为D（负向）
    - 对于D（负向）标签：反转波形，标签变为U（正向）
    - 对于X（不确定）标签：不进行反转
    
    注意：这个增强假设标签映射为 {'U': 0, 'D': 1, 'X': 2}
    """
    def __init__(self, key="X", label_key="label", label_map={'U': 0, 'D': 1, 'X': 2}):
        self.key = key
        self.label_key = label_key
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}
    
    def __call__(self, state_dict):
        if self.key in state_dict:
            value = state_dict[self.key]
            
            # 处理元组情况：当value是(waveform, metadata)元组时
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            # 检查是否有标签信息
            if meta is not None and self.label_key in meta:
                label = meta[self.label_key]
                
                # 只对U和D标签进行反转
                if label == self.label_map['U']:  # U -> D
                    x = -x  # 反转波形
                    meta[self.label_key] = self.label_map['D']
                elif label == self.label_map['D']:  # D -> U
                    x = -x  # 反转波形
                    meta[self.label_key] = self.label_map['U']
                # X标签保持不变
            
            # 根据原始类型返回
            if is_tuple:
                state_dict[self.key] = (x, meta)
            else:
                state_dict[self.key] = x

# ==============================================================================
# Loss Functions (损失函数)
# ==============================================================================
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MultiHeadFocalLoss(nn.Module):
    """Multi-head Focal Loss for models with multiple outputs."""
    def __init__(self, gamma=2.0, weights=None):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma)
        # Weights for o3, o4, o5, ofuse. 
        self.weights = weights or [1.0, 1.0, 1.0, 1.0]

    def forward(self, inputs, targets):
        if isinstance(inputs, (tuple, list)) and len(inputs) >= 4:
            loss = 0.0
            # Only compute loss for the first 4 polarity outputs
            for i in range(4):
                loss += self.weights[i] * self.focal_loss(inputs[i], targets)
            return loss
        return self.focal_loss(inputs, targets)

class DitingMotionLoss(nn.Module):
    """
    DitingMotion模型的专用损失函数。
    
    DitingMotion有8个输出：
    1-4: polarity输出 (o3, o4, o5, ofuse)
    5-8: clarity输出 (o3_clarity, o4_clarity, o5_clarity, ofuse_clarity)
    
    这个损失函数可以处理以下情况：
    1. 只有polarity标签：计算前4个输出的损失
    2. 同时有polarity和clarity标签：计算所有8个输出的损失
    """
    def __init__(self, gamma=2.0, polarity_weights=None, clarity_weights=None, 
                 has_clarity_labels=True):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma)
        # polarity输出的权重
        self.polarity_weights = polarity_weights or [1.0, 1.0, 1.0, 1.0]
        # clarity输出的权重
        self.clarity_weights = clarity_weights or [1.0, 1.0, 1.0, 1.0]
        self.has_clarity_labels = has_clarity_labels
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型输出，应该是8个张量的元组/列表
            targets: 目标标签，可以是：
                - 单个张量：只有polarity标签
                - 元组/列表：(polarity_targets, clarity_targets)
        """
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 8:
            raise ValueError(f"DitingMotionLoss expects 8 outputs, got {len(inputs) if isinstance(inputs, (tuple, list)) else 1}")
        
        # 处理目标标签
        if isinstance(targets, (tuple, list)) and len(targets) == 2:
            polarity_targets, clarity_targets = targets
            has_clarity = True
        else:
            polarity_targets = targets
            clarity_targets = None
            has_clarity = False
        
        total_loss = 0.0
        
        # 计算polarity损失（前4个输出）
        for i in range(4):
            loss = self.focal_loss(inputs[i], polarity_targets)
            total_loss += self.polarity_weights[i] * loss
        
        # 计算clarity损失（后4个输出），如果有clarity标签
        if self.has_clarity_labels and has_clarity and clarity_targets is not None:
            for i in range(4):
                loss = self.focal_loss(inputs[i+4], clarity_targets)
                total_loss += self.clarity_weights[i] * loss
        
        return total_loss

# ==============================================================================
# Utility Augmentations (实用增强工具)
# ==============================================================================

class OneOf:
    """Randomly apply one of the given augmentations."""
    def __init__(self, augmentations, probabilities=None):
        self.augmentations = augmentations
        self.probabilities = probabilities
    def __call__(self, state_dict):
        if not self.augmentations:
            return
        # Simple selection for now
        idx = np.random.choice(len(self.augmentations))
        self.augmentations[idx](state_dict)






