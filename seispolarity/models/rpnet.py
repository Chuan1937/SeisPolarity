import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BasePolarityModel
from seispolarity.annotations import PickList, PolarityLabel, Pick

class InceptionModule(nn.Module):
    """
    Inception module implementation based on Figure 1(b) in the paper [cite: 125].
    Contains convolutional kernels of different scales (1x1, 3x3, 5x5) and max pooling.
    """
    def __init__(self, in_channels, c1, c2_in, c2_out, c3_in, c3_out, c4_out):
        super(InceptionModule, self).__init__()
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Conv1d(in_channels, c1, kernel_size=1)
        
        # Branch 2: 1x1 convolution + 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, c2_in, kernel_size=1),
            nn.Conv1d(c2_in, c2_out, kernel_size=3, padding=1)
        )
        
        # Branch 3: 1x1 convolution + 5x5 convolution
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, c3_in, kernel_size=1),
            nn.Conv1d(c3_in, c3_out, kernel_size=5, padding=2)
        )
        
        # Branch 4: 3x3 max pooling + 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, c4_out, kernel_size=1)
        )

    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        f4 = self.branch4(x)
        # Concatenate along channel dimension [cite: 103]
        return torch.cat([f1, f2, f3, f4], dim=1)

class RPNet(BasePolarityModel, nn.Module):
    """
    RPNet main model architecture [cite: 76, 454].
    Input shape: (batch_size, 1, 400) [cite: 78].
    
    Reference:
        Han, J., Kim, S. & Sheen, D.-H. RPNet: Robust P-Wave First-motion polarity determination using deep learning.
        Seismological Research Letters (2025).
    
    Author:
        Model weights converted and maintained by He XingChen (Chinese, Han ethnicity), https://github.com/Chuan1937
    """
    def __init__(self, sample_rate=100.0, **kwargs):
        BasePolarityModel.__init__(self, name="RPNet", sample_rate=sample_rate, n_components=1, **kwargs)
        nn.Module.__init__(self)
        
        # 1. Feature extraction stage [cite: 98, 455]
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Two Inception modules [cite: 82, 83]
        self.inception1 = InceptionModule(64, 64, 64, 96, 16, 32, 32) # Output channels: 64+96+32+32=224
        self.inception2 = InceptionModule(224, 128, 128, 192, 32, 96, 64) # Output channels: 480
        
        # 2. P-wave focusing stage [cite: 99, 457]
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=480, hidden_size=480, batch_first=True, bidirectional=False)
        self.self_attention = nn.MultiheadAttention(embed_dim=480, num_heads=8, batch_first=True)
        
        # 3. Classification stage [cite: 100, 503]
        self.flatten = nn.Flatten()
        # According to model summary, total parameters ~5.1M
        self.fc1 = nn.Linear(50 * 480, 128)
        self.dropout = nn.Dropout(p=0.2) # Monte Carlo Dropout [cite: 91, 507]
        self.fc2 = nn.Linear(128, 2) # Output UP and DOWN classes [cite: 119, 503]

    def forward(self, x):
        # x shape: (batch, 1, 400)
        x = self.initial_conv(x)
        
        x = self.inception1(x)
        x = self.inception2(x)
        
        # Prepare for LSTM (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)
        
        # LSTM + Attention [cite: 81, 79]
        x, _ = self.lstm(x)
        attn_output, _ = self.self_attention(x, x, x)
        
        # Fully connected layers
        x = self.flatten(attn_output)
        x = F.relu(self.fc1(x))
        
        # Note: To implement MC Dropout, set training=True during prediction [cite: 506]
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Return Logits for CrossEntropyLoss training in PyTorch
        return x

    def forward_tensor(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for a batch tensor."""
        return self.forward(tensor)

    def build_picks(self, raw_output: torch.Tensor, **kwargs) -> PickList:
        """
        Convert raw model output to picks with polarity labels.
        Assuming output index 1 is UP, 0 is DOWN.
        """
        picks = PickList()
        probs = torch.softmax(raw_output, dim=1)
        
        for i in range(probs.shape[0]):
            prob_up = probs[i, 1].item()
            polarity = PolarityLabel.UP if prob_up > 0.5 else PolarityLabel.DOWN
            
            picks.append(
                Pick(
                    trace_id="unknown",
                    time=None,
                    confidence=float(prob_up if prob_up > 0.5 else 1 - prob_up),
                    polarity=polarity,
                    extra={"logits": raw_output[i].tolist()}
                )
            )
        return picks

def rpnet(**kwargs):
    """Factory function for RPNet."""
    return RPNet(**kwargs)

