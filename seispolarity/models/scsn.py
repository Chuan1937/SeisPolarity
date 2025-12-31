import torch
import torch.nn as nn
from .base import BasePolarityModel
from seispolarity.annotations import PickList, PolarityLabel, Pick

class SharedBackbone(nn.Module):
    """定义共享的骨干网络 (输入400点)。"""
    def __init__(self):
        super(SharedBackbone, self).__init__()
        self.sequential = nn.Sequential(
            # 输入 (N, 1, 400)
            nn.Conv1d(1, 32, kernel_size=21, padding='same'),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # -> (N, 32, 200)
            
            nn.Conv1d(32, 64, kernel_size=15, padding='same'),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # -> (N, 64, 100)

            nn.Conv1d(64, 128, kernel_size=11, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # -> (N, 128, 50)
            
            nn.Flatten(),
            
            # Flatten后的维度是 128 * 50 = 6400
            nn.Linear(6400, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
        )

    def forward(self, x):
        return self.sequential(x)

class SCSN(BasePolarityModel, nn.Module):
    """构建单任务的极性判断模型。"""
    def __init__(self, num_fm_classes=3, **kwargs):
        BasePolarityModel.__init__(self, name="SCSN", **kwargs)
        nn.Module.__init__(self)
        self.shared_backbone = SharedBackbone()
        self.fm_head = nn.Linear(512, num_fm_classes)
        
    def forward(self, x):
        # PyTorch Conv1d 需要 (N, C, L) 格式
        # BasePolarityModel.preprocess 已经提供了 (N, C, L)
        # 输入 x 的形状应该是 (N, 1, 400)
        
        f1 = self.shared_backbone(x)

        # 只保留极性判断的输出
        fm_output = self.fm_head(f1)
        return fm_output

    def forward_tensor(self, tensor: torch.Tensor, **kwargs):
        return self.forward(tensor)

    def build_picks(self, raw_output, **kwargs) -> PickList:
        # raw_output shape: (N, 3)
        # Assuming classes are 0: Up, 1: Down, 2: Unknown (or similar)
        # Need to verify class mapping from original code or paper.
        # Usually: 0: U, 1: D, 2: N? Or U, D, N?
        # For now, I'll assume argmax maps to PolarityLabel.
        
        # TODO: Verify class mapping.
        # Assuming: 0 -> UP, 1 -> DOWN, 2 -> UNKNOWN
        
        probs = torch.softmax(raw_output, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        picks = []
        # This model classifies a window, so it corresponds to one pick per input sample?
        # But BasePolarityModel.annotate takes a stream and preprocesses it into ONE tensor (1, C, T).
        # If the stream is long, SCSN expects 400 samples.
        # So SCSN is not suitable for continuous annotation via BasePolarityModel.annotate 
        # unless the stream IS the window.
        
        # If the input was a batch of windows, we would have N picks.
        # But BasePolarityModel.annotate creates (1, C, T).
        
        # If we use this for classification of existing picks, we might need a different interface.
        
        return [] # Placeholder
