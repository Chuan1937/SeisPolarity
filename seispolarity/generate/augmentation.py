import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

class Demean:
    """Remove the mean from the waveform.
    
    For seismic waveform data, typically remove mean along time axis (axis=-1).
    """
    def __init__(self, key="X", axis=-1):
        self.key = key
        self.axis = axis
    def __call__(self, state_dict):
        if self.key in state_dict:
            value = state_dict[self.key]
            
            # Handle tuple case: when value is (waveform, metadata) tuple
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            mean = np.mean(x, axis=self.axis, keepdims=True)
            x = x - mean
            
            # Return based on original type
            if is_tuple:
                state_dict[self.key] = (x, meta)
            else:
                state_dict[self.key] = x

class Detrend:
    """Remove the linear trend from the waveform.
    
    For seismic waveform data, typically remove linear trend along time axis (axis=-1).
    Use scipy.signal.detrend to remove linear trend.
    
    Args:
        key: Key in state dictionary
        axis: Axis for detrending, typically -1 (time axis)
        type: Detrend type, 'linear' or 'constant'
    """
    def __init__(self, key="X", axis=-1, type="linear"):
        self.key = key
        self.axis = axis
        self.type = type
    def __call__(self, state_dict):
        if self.key in state_dict:
            value = state_dict[self.key]
            
            # Handle tuple case: when value is (waveform, metadata) tuple
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            x = signal.detrend(x, axis=self.axis, type=self.type)
            
            # Return based on original type
            if is_tuple:
                state_dict[self.key] = (x, meta)
            else:
                state_dict[self.key] = x

class Stretching:
    """Stretch the waveform by upsampling and cropping.
    
    Implement data stretching by upsampling and cropping to simulate low-frequency signals.
    Original data frequency 5-15 Hz, after 2-3x stretching frequency is approximately 2-5 Hz.
    
    Processing flow:
    1. Upsample from original sampling rate to higher rate (e.g., 100 Hz -> 200 Hz or 300 Hz)
    2. Crop fixed-length samples containing P-wave (default 400 samples)
    
    Args:
        key: Key in state dictionary
        original_fs: Original sampling rate, default 100 Hz
        stretch_factors: List of stretch factors, e.g., [2, 3] means randomly select 2x or 3x stretching
        target_samples: Number of samples to crop, default 400
        p_pick_key: Key name for P-wave arrival time in metadata
        crop_left: Number of samples to crop left of P-wave
        crop_right: Number of samples to crop right of P-wave
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
            
            # Handle tuple case: when value is (waveform, metadata) tuple
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            # Ensure x is at least 2D array
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # Randomly select stretch factor
            stretch_factor = np.random.choice(self.stretch_factors)
            
            # Calculate target sampling rate
            target_fs = self.original_fs * stretch_factor
            
            # Get original length (number of time points)
            original_length = x.shape[-1]
            
            # Upsample to target sampling rate
            original_time = np.linspace(0, original_length / self.original_fs, original_length)
            target_length = int(original_length * stretch_factor)
            target_time = np.linspace(0, original_length / self.original_fs, target_length)
            
            # Interpolate for each channel
            stretched_data = np.zeros((x.shape[0], target_length))
            for i in range(x.shape[0]):
                stretched_data[i] = np.interp(target_time, original_time, x[i])
            
            # Crop region containing P-wave
            if meta is not None and self.p_pick_key in meta:
                p_pick = meta[self.p_pick_key]
                # Map original sample positions to stretched positions
                p_pick_stretched = int(p_pick * stretch_factor)
                
                # Calculate crop_left and crop_right
                if self.crop_left is None:
                    crop_left = self.target_samples // 2
                else:
                    crop_left = int(self.crop_left * stretch_factor)
                
                if self.crop_right is None:
                    crop_right = self.target_samples // 2
                else:
                    crop_right = int(self.crop_right * stretch_factor)
                
                # Calculate crop range
                start_idx = p_pick_stretched - crop_left
                end_idx = p_pick_stretched + crop_right
                
                # Ensure indices are within valid range
                if start_idx < 0:
                    start_idx = 0
                    end_idx = min(target_length, self.target_samples)
                elif end_idx > target_length:
                    end_idx = target_length
                    start_idx = max(0, end_idx - self.target_samples)
                
                # Crop data
                cropped_data = stretched_data[:, start_idx:end_idx]
                
                # If cropped length is insufficient, pad
                if cropped_data.shape[-1] < self.target_samples:
                    pad_length = self.target_samples - cropped_data.shape[-1]
                    pad_left = pad_length // 2
                    pad_right = pad_length - pad_left
                    cropped_data = np.pad(cropped_data, 
                                          ((0, 0), (pad_left, pad_right)), 
                                          mode='constant')
                
                # Update sampling rate and P-wave arrival time in metadata
                meta['sampling_rate'] = target_fs
                meta[self.p_pick_key] = self.target_samples // 2  # Place P-wave in center
                
                x = cropped_data
            else:
                # If no P-wave information, directly crop center region
                start_idx = (target_length - self.target_samples) // 2
                end_idx = start_idx + self.target_samples
                if start_idx < 0:
                    start_idx = 0
                    end_idx = min(target_length, self.target_samples)
                elif end_idx > target_length:
                    end_idx = target_length
                    start_idx = max(0, end_idx - self.target_samples)
                
                x = stretched_data[:, start_idx:end_idx]
                
                # Update sampling rate
                if meta is not None:
                    meta['sampling_rate'] = target_fs
            
            # Return based on original type
            if is_tuple:
                state_dict[self.key] = (x, meta)
            else:
                state_dict[self.key] = x

class BandpassFilter:
    """Apply bandpass or highpass filter to the waveform.
    
    Use Butterworth filter to filter the waveform.
    When highcut is None, implement highpass filter; otherwise implement bandpass filter.
    
    Args:
        key: Key in state dictionary
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency, if None then implement highpass filter
        fs: Sampling rate, if None read from metadata
        order: Filter order
        zerophase: Whether to use zero-phase filtering
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
            
            # Handle tuple case: when value is (waveform, metadata) tuple
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            # Determine sampling rate
            if self.fs is None and meta is not None and 'sampling_rate' in meta:
                fs = meta['sampling_rate']
            elif self.fs is not None:
                fs = self.fs
            else:
                raise ValueError("Sampling rate must be provided either directly or in metadata")
            
            # Ensure x is at least 2D array
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # Design filter
            nyquist = 0.5 * fs
            
            if self.highcut is None:
                # Highpass filter
                high = self.lowcut / nyquist
                if high >= 1.0:
                    raise ValueError(f"Invalid cutoff frequency: {self.lowcut} Hz for sampling rate {fs} Hz")
                sos = signal.butter(self.order, high, btype='high', output='sos')
            else:
                # Bandpass filter
                low = self.lowcut / nyquist
                high = self.highcut / nyquist
                if low >= 1.0 or high >= 1.0 or low >= high:
                    raise ValueError(f"Invalid frequency range: [{self.lowcut}, {self.highcut}] Hz for sampling rate {fs} Hz")
                sos = signal.butter(self.order, [low, high], btype='band', output='sos')
            
            # Apply filter to each channel
            filtered_data = np.zeros_like(x)
            for i in range(x.shape[0]):
                if self.zerophase:
                    filtered_data[i] = signal.sosfiltfilt(sos, x[i])
                else:
                    filtered_data[i] = signal.sosfilt(sos, x[i])
            
            # Return based on original type
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
            
            # Handle tuple case: when value is (waveform, metadata) tuple
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
            
            # Return based on original type
            if is_tuple:
                state_dict[self.key] = (x, meta)
            else:
                state_dict[self.key] = x

class DifferentialFeatures:
    """Computes the differential of the waveform and its sign.
    
    For seismic waveform data, typically shape is (C, N) or (N,), where:
    - C: Number of channels (typically 1)
    - N: Number of time points (e.g., 128)
    
    When append=True, add differential sign feature as new channel.
    Example: input shape (1, 128) -> output shape (2, 128)
    """
    def __init__(self, key="X", axis=-1, append=True):
        self.key = key
        self.axis = axis
        self.append = append
    def __call__(self, state_dict):
        if self.key in state_dict:
            value = state_dict[self.key]
            
            # Handle tuple case: when value is (waveform, metadata) tuple
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            # Ensure x is at least 2D array
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # Calculate differential (along time axis)
            diff = np.diff(x, axis=self.axis)
            
            # Pad to maintain original length (pad with 0 at beginning)
            pad_shape = [(0, 0)] * x.ndim
            if self.axis == -1:
                idx = x.ndim - 1
            else:
                idx = self.axis
            pad_shape[idx] = (1, 0) # Pad with 1 zero at beginning
            
            diff = np.pad(diff, pad_shape, mode='constant')
            sign_diff = np.sign(diff)
            
            if self.append:
                # Concatenate along channel axis (axis=0)
                new_x = np.concatenate([x, sign_diff], axis=0)
                # Return based on original type
                if is_tuple:
                    state_dict[self.key] = (new_x, meta)
                else:
                    state_dict[self.key] = new_x
            else:
                # When append=False, store differential sign feature as new key
                if is_tuple:
                    # If original is tuple, we need to maintain tuple structure
                    state_dict[self.key] = (x, meta)
                state_dict[self.key + "_diff_sign"] = sign_diff

class RandomTimeShift:
    """Randomly perturb P-wave arrival time labels.
    
    Apply uniform random shift to p_pick in metadata within ±max_shift range.
    This simulates P-wave arrival time picking uncertainty in real situations, making model robust to P-wave alignment errors.
    
    Note: This augmentation does not change waveform data, only modifies p_pick label in metadata.
    Waveform cropping will be done in subsequent WaveformDataset based on perturbed p_pick.
    
    Example: sampling rate 100Hz, ±0.5s corresponds to ±50 samples.
    
    Args:
        key: Key in state dictionary (used to get waveform and metadata)
        max_shift: Maximum shift amount
        shift_unit: Shift unit, 'samples' (samples) or 'seconds' (seconds)
        sampling_rate: Sampling rate, used when shift_unit='seconds'
        p_pick_key: Key name for p_pick in metadata
        fixed_p_pick: Fixed P-wave arrival time (samples), used when metadata has no p_pick_key
                      e.g., P-wave in SCSN data is fixed at a certain position
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
            
            # Handle tuple case: when value is (waveform, metadata) tuple
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            # Only modify p_pick in metadata
            if meta is not None:
                # Get P-wave arrival time: prioritize reading from metadata, otherwise use fixed_p_pick
                has_p_pick_key = False
                p_pick = None
                
                if self.p_pick_key in meta:
                    p_pick = meta[self.p_pick_key]
                    has_p_pick_key = True
                elif self.fixed_p_pick is not None:
                    p_pick = self.fixed_p_pick
                    has_p_pick_key = False
                
                # Apply shift
                if has_p_pick_key or (self.fixed_p_pick is not None):
                    # Calculate random shift based on shift_unit
                    if self.shift_unit == "seconds":
                        shift_seconds = np.random.uniform(-self.max_shift, self.max_shift)
                        shift_samples = int(shift_seconds * self.sampling_rate)
                    else:  # samples
                        shift_samples = np.random.randint(-self.max_shift, self.max_shift + 1)
                    
                    # Apply shift
                    if has_p_pick_key:
                        # Update p_pick in metadata
                        meta[self.p_pick_key] = p_pick + shift_samples
                    else:
                        # Use fixed_p_pick, create or update p_pick in metadata
                        meta[self.p_pick_key] = p_pick + shift_samples
            
            # Return based on original type
            if is_tuple:
                state_dict[self.key] = (x, meta)
            else:
                state_dict[self.key] = x

class ChangeDtype:
    """Change the dtype of the data in the state dict."""
    def __init__(self, dtype, key="X"):
        self.dtype = dtype
        self.key = key
    def __call__(self, state_dict):
        if self.key in state_dict:
            value = state_dict[self.key]
            
            # Handle tuple case: when value is (waveform, metadata) tuple
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            if hasattr(x, "astype"):
                x = x.astype(self.dtype)
            
            # Return based on original type
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
            
            # Handle tuple case: when value is (waveform, metadata) tuple
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
            
            # Return based on original type
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

class GaussianNoise:
    """Add Gaussian noise to the waveform."""
    def __init__(self, key="X", std=0.1):
        self.key = key
        self.std = std
    def __call__(self, state_dict):
        if self.key in state_dict:
            value = state_dict[self.key]
            
            # Handle tuple case: when value is (waveform, metadata) tuple
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            noise = np.random.normal(0, self.std, x.shape)
            x = x + noise
            
            # Return based on original type
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
            
            # Handle tuple case: when value is (waveform, metadata) tuple
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            if x.ndim >= 2:
                # Apply independent dropout to each channel
                mask = np.random.binomial(1, 1 - self.dropout_prob, size=x.shape[0])
                mask = mask.reshape(-1, *([1] * (x.ndim - 1)))
                x = x * mask
            
            # Return based on original type
            if is_tuple:
                state_dict[self.key] = (x, meta)
            else:
                state_dict[self.key] = x

class PolarityInversion:
    """Invert waveform polarity and update labels accordingly.
    
    This augmentation is specifically for polarity classification tasks:
    - For U (positive) labels: invert waveform, label becomes D (negative)
    - For D (negative) labels: invert waveform, label becomes U (positive)
    - For X (uncertain) labels: no inversion
    
    Note: This augmentation assumes label mapping is {'U': 0, 'D': 1, 'X': 2}
    """
    def __init__(self, key="X", label_key="label", label_map={'U': 0, 'D': 1, 'X': 2}):
        self.key = key
        self.label_key = label_key
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}
    
    def __call__(self, state_dict):
        if self.key in state_dict:
            value = state_dict[self.key]
            
            # Handle tuple case: when value is (waveform, metadata) tuple
            if isinstance(value, tuple) and len(value) == 2:
                x, meta = value
                is_tuple = True
            else:
                x = value
                meta = None
                is_tuple = False
            
            # Check if there is label information
            if meta is not None and self.label_key in meta:
                label = meta[self.label_key]
                
                # Only invert U and D labels
                if label == self.label_map['U']:  # U -> D
                    x = -x  # Invert waveform
                    meta[self.label_key] = self.label_map['D']
                elif label == self.label_map['D']:  # D -> U
                    x = -x  # Invert waveform
                    meta[self.label_key] = self.label_map['U']
                # X label remains unchanged
            
            # Return based on original type
            if is_tuple:
                state_dict[self.key] = (x, meta)
            else:
                state_dict[self.key] = x

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
    Specialized loss function for DitingMotion model.
    
    DitingMotion has 8 outputs:
    1-4: polarity outputs (o3, o4, o5, ofuse)
    5-8: clarity outputs (o3_clarity, o4_clarity, o5_clarity, ofuse_clarity)
    
    This loss function can handle the following cases:
    1. Only polarity labels: compute loss for first 4 outputs
    2. Both polarity and clarity labels: compute loss for all 8 outputs
    """
    def __init__(self, gamma=2.0, polarity_weights=None, clarity_weights=None, 
                 has_clarity_labels=True):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma)
        # Weights for polarity outputs
        self.polarity_weights = polarity_weights or [1.0, 1.0, 1.0, 1.0]
        # Weights for clarity outputs
        self.clarity_weights = clarity_weights or [1.0, 1.0, 1.0, 1.0]
        self.has_clarity_labels = has_clarity_labels
        
    def forward(self, outputs, targets, **kwargs):
        """
        Args:
            outputs: Model output, should be tuple/list of 8 tensors
            targets: Target labels, can be:
                - Single tensor: only polarity labels
                - Tuple/list: (polarity_targets, clarity_targets)
            **kwargs: Extra parameters (for receiving inputs keyword argument)
        """
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 8:
            raise ValueError(f"DitingMotionLoss expects 8 outputs, got {len(outputs) if isinstance(outputs, (tuple, list)) else 1}")
        
        # Process target labels
        if isinstance(targets, (tuple, list)) and len(targets) == 2:
            polarity_targets, clarity_targets = targets
            has_clarity = True
        else:
            polarity_targets = targets
            clarity_targets = None
            has_clarity = False
        
        total_loss = 0.0
        
        # Compute polarity loss (first 4 outputs)
        for i in range(4):
            loss = self.focal_loss(outputs[i], polarity_targets)
            total_loss += self.polarity_weights[i] * loss
        
        # Compute clarity loss (last 4 outputs), if clarity labels exist
        if self.has_clarity_labels and has_clarity and clarity_targets is not None:
            for i in range(4):
                loss = self.focal_loss(outputs[i+4], clarity_targets)
                total_loss += self.clarity_weights[i] * loss
        
        return total_loss

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






