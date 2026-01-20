import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BasePolarityModel
from seispolarity.annotations import PickList, PolarityLabel, Pick

class InceptionModule(nn.Module):
    """
    基于论文图 1(b) 实现的 Inception 模块 [cite: 125]。
    包含不同尺度的卷积核（1x1, 3x3, 5x5）和最大池化。
    """
    def __init__(self, in_channels, c1, c2_in, c2_out, c3_in, c3_out, c4_out):
        super(InceptionModule, self).__init__()
        # 线路 1: 1x1 卷积
        self.branch1 = nn.Conv1d(in_channels, c1, kernel_size=1)
        
        # 线路 2: 1x1 卷积 + 3x3 卷积
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, c2_in, kernel_size=1),
            nn.Conv1d(c2_in, c2_out, kernel_size=3, padding=1)
        )
        
        # 线路 3: 1x1 卷积 + 5x5 卷积
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, c3_in, kernel_size=1),
            nn.Conv1d(c3_in, c3_out, kernel_size=5, padding=2)
        )
        
        # 线路 4: 3x3 最大池化 + 1x1 卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, c4_out, kernel_size=1)
        )

    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        f4 = self.branch4(x)
        # 沿通道维度拼接 [cite: 103]
        return torch.cat([f1, f2, f3, f4], dim=1)

class RPNet(BasePolarityModel, nn.Module):
    """
    RPNet 主模型架构 [cite: 76, 454]。
    输入形状: (batch_size, 1, 400) [cite: 78]。
    """
    def __init__(self, sample_rate=100.0, **kwargs):
        BasePolarityModel.__init__(self, name="RPNet", sample_rate=sample_rate, n_components=1, **kwargs)
        nn.Module.__init__(self)
        
        # 1. 特征提取阶段 (Feature Extraction) [cite: 98, 455]
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 两个 Inception 模块 [cite: 82, 83]
        self.inception1 = InceptionModule(64, 64, 64, 96, 16, 32, 32) # 输出通道: 64+96+32+32=224
        self.inception2 = InceptionModule(224, 128, 128, 192, 32, 96, 64) # 输出通道: 480
        
        # 2. P波聚焦阶段 (P-wave Focusing) [cite: 99, 457]
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=480, hidden_size=480, batch_first=True, bidirectional=False)
        self.self_attention = nn.MultiheadAttention(embed_dim=480, num_heads=8, batch_first=True)
        
        # 3. 分类阶段 (Classification) [cite: 100, 503]
        self.flatten = nn.Flatten()
        # 根据模型摘要，参数总量约为 5.1M
        self.fc1 = nn.Linear(50 * 480, 128)
        self.dropout = nn.Dropout(p=0.2) # Monte Carlo Dropout [cite: 91, 507]
        self.fc2 = nn.Linear(128, 2) # 输出 Up 和 Down 两类 [cite: 119, 503]

    def forward(self, x):
        # x shape: (batch, 1, 400)
        x = self.initial_conv(x)
        
        x = self.inception1(x)
        x = self.inception2(x)
        
        # 准备进入 LSTM (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)
        
        # LSTM + Attention [cite: 81, 79]
        x, _ = self.lstm(x)
        attn_output, _ = self.self_attention(x, x, x)
        
        # 全连接层
        x = self.flatten(attn_output)
        x = F.relu(self.fc1(x))
        
        # 注意：为了实现 MC Dropout，预测时需开启 training=True [cite: 506]
        x = self.dropout(x)
        x = self.fc2(x)
        
        # 返回 Logits 以便在 PyTorch 中使用 CrossEntropyLoss 训练
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

