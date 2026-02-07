import torch
import torch.nn as nn
from .base import BasePolarityModel
from seispolarity.annotations import PickList, PolarityLabel, Pick

class SharedBackbone(nn.Module):
    """Define the shared backbone network (input 400 points)."""
    def __init__(self):
        super(SharedBackbone, self).__init__()
        self.sequential = nn.Sequential(
            # Input (N, 1, 400)
            nn.Conv1d(1, 32, kernel_size=21, padding='same'),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # -> (N, 32, 200)
            
            nn.Conv1d(32, 64, kernel_size=21, padding='same'),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # -> (N, 64, 100)

            nn.Conv1d(64, 128, kernel_size=21, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # -> (N, 128, 50)
            
            nn.Conv1d(128, 256, kernel_size=21, padding='same'),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # -> (N, 256, 25)
            
            nn.Flatten(),
            
            # Dimension after flattening is 256 * 25 = 6400
            nn.Linear(6400, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
        )

    def forward(self, x):
        return self.sequential(x)

class SCSN(BasePolarityModel, nn.Module):
    """
    Build a single-task polarity classification model.
    
    Reference:
        Ross, Z. E., Meier, M. & Hauksson, E. P Wave Arrival Picking and First-Motion Polarity Determination With Deep Learning. 
        JGR Solid Earth 123, 5120-5129 (2018).
    
    Author:
        Model weights converted and maintained by He XingChen (Chinese, Han ethnicity), https://github.com/Chuan1937
    """
    def __init__(self, num_fm_classes=3, **kwargs):
        BasePolarityModel.__init__(self, name="SCSN", **kwargs)
        nn.Module.__init__(self)
        self.shared_backbone = SharedBackbone()
        self.fm_head = nn.Linear(512, num_fm_classes)
        
    def forward(self, x):
        # PyTorch Conv1d requires (N, C, L) format
        # BasePolarityModel.preprocess already provides (N, C, L)
        # Input x shape should be (N, 1, 400)
        
        f1 = self.shared_backbone(x)

        # Only keep the polarity classification output
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

        probs = torch.softmax(raw_output, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        picks = []
        
        return [] 
