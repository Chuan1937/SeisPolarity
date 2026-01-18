import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """对时间窗进行随机平移。
    
    对时间窗的中心进行±max_shift范围内的均匀随机平移。
    假设数据形状为 (C, N) 或 (N,)，其中N是时间点数量。
    
    例如：采样率50Hz，±0.5s对应±25个采样点。
    """
    def __init__(self, key="X", max_shift=25, mode="reflect"):
        """
        :param key: 状态字典中的键
        :param max_shift: 最大平移量（采样点）
        :param mode: 填充模式，可选 'constant', 'edge', 'reflect', 'symmetric'
        """
        self.key = key
        self.max_shift = max_shift
        self.mode = mode
    
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
            
            # 生成随机平移量（包括0）
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            
            if shift == 0:
                # 不进行平移
                return
            
            # 获取数据形状
            n_channels, n_time = x.shape[0], x.shape[-1]
            
            if shift > 0:
                # 向右平移（时间向前）
                # 左侧填充，右侧截断
                pad_width = [(0, 0)] * x.ndim
                pad_width[-1] = (shift, 0)  # 在时间轴左侧填充
                
                # 使用填充
                x_padded = np.pad(x, pad_width, mode=self.mode)
                # 截取原始长度
                x_shifted = x_padded[:, :n_time]
            else:
                # 向左平移（时间向后）
                shift = abs(shift)
                # 右侧填充，左侧截断
                pad_width = [(0, 0)] * x.ndim
                pad_width[-1] = (0, shift)  # 在时间轴右侧填充
                
                # 使用填充
                x_padded = np.pad(x, pad_width, mode=self.mode)
                # 截取原始长度
                x_shifted = x_padded[:, shift:shift + n_time]
            
            # 根据原始类型返回
            if is_tuple:
                state_dict[self.key] = (x_shifted, meta)
            else:
                state_dict[self.key] = x_shifted

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






