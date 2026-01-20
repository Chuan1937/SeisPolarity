import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def norm(X):
    # X shape: (batch, seq_len)
    maxi = np.max(np.abs(X), axis=1, keepdims=True)
    # 防止除以零
    maxi[maxi == 0] = 1.0
    return X / maxi

class PolarCAP(nn.Module):
    def __init__(self, drop_rate=0.3):
        super(PolarCAP, self).__init__()
        
        # 1. Encoder 部分 (基于 Keras 的 enc 层)
        # 输入 shape: (batch, 1, 64)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=32, padding=16), # padding='same'
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, padding=0), # (batch, 32, 32)
            
            nn.Conv1d(32, 8, kernel_size=16, padding=8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2, padding=0)  # (batch, 8, 16) -> 这就是 'enc' 层
        )
        
        # 2. Decoder 分支 (用于信号重构 dec)
        self.decoder = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=16, padding=8),
            nn.Tanh(),
            nn.BatchNorm1d(8),
            nn.Upsample(scale_factor=2), # (batch, 8, 32)
            
            nn.Conv1d(8, 32, kernel_size=32, padding=16),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Upsample(scale_factor=2), # (batch, 32, 64)
            
            nn.Conv1d(32, 1, kernel_size=64, padding=32),
            nn.Tanh() # 输出重构波形 (batch, 1, 64)
        )
        
        # 3. Polarity 分类分支 (Dense 层 p)
        # enc 的输出 flatten 后的大小: 8 channels * 16 samples = 128
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 16, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: (batch, 1, 64)
        enc = self.encoder(x)
        
        # 任务 1: 重构波形
        # PyTorch 的 Conv1d 相同卷积可能导致长度偏移，这里手动裁切对齐到 64
        dec = self.decoder(enc)[:, :, :64]
        
        # 任务 2: 极性分类
        p = self.classifier(enc)
        
        return dec, p

    def predict(self, X_np):
        self.eval()
        with torch.no_grad():
            # 预处理
            X_norm = norm(X_np)
            X_tensor = torch.FloatTensor(X_norm).unsqueeze(1) # (batch, 1, 64)
            
            # 推理
            _, p_probs = self.forward(X_tensor)
            
            # 结果转换
            pol_pred = torch.argmax(p_probs, dim=1).cpu().numpy()
            pred_prob = torch.max(p_probs, dim=1).values.cpu().numpy()
            
            polarity_labels = ['Negative', 'Positive']
            predictions = [(polarity_labels[pol], prob) for pol, prob in zip(pol_pred, pred_prob)]
            
        return predictions

class PolarCAPLoss(nn.Module):
    """
    PolarCAP 模型的专用多任务损失函数。
    
    计算 MSE(dec_pred, inputs) + alpha * Huber(p_pred, labels)
    """
    # 论文使用的为200 - alpha
    def __init__(self, alpha=10.0, delta=0.5):
        super().__init__()
        self.alpha = alpha
        self.huber_criterion = nn.HuberLoss(delta=delta)

    def forward(self, outputs, targets, inputs=None):
        """
        Args:
            outputs: 模型输出 (dec_pred, p_pred)
            targets: 极性标签 (batch,)
            inputs: 原始输入波形 (batch, 1, 64)，用于计算重构损失
        """
        if inputs is None:
            raise ValueError("PolarCAPLoss requires 'inputs' (original waveforms) for reconstruction loss.")
        
        dec_pred, p_pred = outputs
        
        # 重构损失: MSE
        mse_loss = F.mse_loss(dec_pred, inputs)
        
        # 分类损失: Huber (Keras 实现中对 One-hot 标签使用了 Huber，这里 p_pred 是 (batch, 2))
        # 首先将 targets 转为 one-hot 以匹配 p_pred (batch, 2)
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        huber_loss = self.huber_criterion(p_pred, targets_one_hot)
        
        return mse_loss + self.alpha * huber_loss

# --- 损失函数定义 (保留原样以便向后兼容) ---
def get_polarcap_loss(dec_pred, dec_true, p_pred, p_true):
    mse_loss = F.mse_loss(dec_pred, dec_true)
    # Huber loss 在 PyTorch 中默认 delta=1.0，根据 Keras 代码设为 0.5
    huber_criterion = nn.HuberLoss(delta=0.5)
    huber_loss = huber_criterion(p_pred, p_true)
    
    # 调整权重为更合理的值
    total_loss = mse_loss + 10 * huber_loss
    return total_loss