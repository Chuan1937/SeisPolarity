import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BasePolarityModel
from seispolarity.annotations import PickList, PolarityLabel, Pick


class CFM(BasePolarityModel, nn.Module):
    """
    CFM (Convolutional Feature Model) for polarity detection.
    PyTorch implementation based on the provided Keras model architecture description.
    
    Model architecture (as per user-provided TensorFlow summary):
    1. conv1d (Conv1D): (None, 160, 32), params=192 (kernel_size=5, same padding, 1 in_channel)
    2. dropout (Dropout): (None, 160, 32)
    
    Reference:
        Messuti, G. et al. CFM: a convolutional neural network for first-motion polarity classification of seismic records 
        in volcanic and tectonic areas. Frontiers in Earth Science 11, 1223686 (2023).
    
    Author:
        Model weights converted and maintained by He XingChen (Chinese, Han ethnicity), https://github.com/Chuan1937
    
    3. conv1d_1 (Conv1D): (None, 157, 64), params=8,256 (kernel_size=4, valid padding)
    4. max_pooling1d (MaxPooling1D): (None, 78, 64)
    5. conv1d_2 (Conv1D): (None, 76, 128), params=24,704 (kernel_size=3, valid padding)
    6. max_pooling1d_1 (MaxPooling1D): (None, 38, 128)
    7. conv1d_3 (Conv1D): (None, 38, 256), params=164,096 (kernel_size=5, same padding)
    8. dropout_1 (Dropout): (None, 38, 256)
    9. conv1d_4 (Conv1D): (None, 36, 128), params=98,432 (kernel_size=3, valid padding)
    10. max_pooling1d_2 (MaxPooling1D): (None, 18, 128)
    11. flatten (Flatten): (None, 2304)
    12. dense (Dense): (None, 50), params=115,250
    13. dense_1 (Dense): (None, 1), params=51
    """
    
    def __init__(self, sample_rate=100.0, **kwargs):
        """
        Initialize CFM model.
        
        Args:
            sample_rate: Sampling rate in Hz
            **kwargs: Additional arguments for BasePolarityModel
        """
        # Set n_components=1 as the param count indicates 1 input channel
        BasePolarityModel.__init__(self, name="CFM", sample_rate=sample_rate, n_components=1, **kwargs)
        nn.Module.__init__(self)
        
        # Layer 1: Input (B, 1, 160) -> (B, 32, 160)
        # kernel_size=5, same padding means padding=2
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.dropout1 = nn.Dropout(0.2)
        
        # Layer 2: (B, 32, 160) -> (B, 64, 157)
        # 160 - 4 + 1 = 157 (valid padding)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # 157 // 2 = 78
        
        # Layer 3: (B, 64, 78) -> (B, 128, 76)
        # 78 - 3 + 1 = 76 (valid padding)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # 76 // 2 = 38
        
        # Layer 4: (B, 128, 38) -> (B, 256, 38)
        # kernel_size=5, same padding means padding=2
        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Layer 5: (B, 256, 38) -> (B, 128, 36)
        # 38 - 3 + 1 = 36 (valid padding)
        self.conv5 = nn.Conv1d(256, 128, kernel_size=3, padding=0)
        self.pool3 = nn.MaxPool1d(kernel_size=2)  # 36 // 2 = 18
        
        # Final layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2304, 50)
        self.fc2 = nn.Linear(50, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 1, 160)
        
        Returns:
            Output tensor of shape (batch, 1)
        """
        # Layer 1
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        
        # Layer 2
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        # Layer 3
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        
        # Layer 4
        x = F.relu(self.conv4(x))
        x = self.dropout2(x)
        
        # Layer 5
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        
        # Flatten and FC
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def forward_tensor(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for a batch tensor."""
        return self.forward(tensor)
    
    def build_picks(self, raw_output: torch.Tensor, **kwargs) -> PickList:
        """
        Convert raw model output to picks with polarity labels.
        Binary classification: >0.5 probability (after sigmoid) is UP, otherwise DOWN.
        """
        picks = PickList()
        probs = torch.sigmoid(raw_output)
        
        for i in range(probs.shape[0]):
            prob = probs[i].item()
            # Mapping probability to polarity
            polarity = PolarityLabel.UP if prob > 0.5 else PolarityLabel.DOWN
            
            picks.append(
                Pick(
                    trace_id="unknown",
                    time=None,
                    confidence=float(prob if prob > 0.5 else 1 - prob),
                    polarity=polarity,
                    extra={"score": float(raw_output[i].item())}
                )
            )
        return picks


def cfm(**kwargs):
    """
    Factory function to create CFM model.
    
    Args:
        **kwargs: Arguments for CFM constructor
    
    Returns:
        CFM model instance
    """
    return CFM(**kwargs)