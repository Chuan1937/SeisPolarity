import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def norm(X):
    """Normalize waveforms by their peak amplitude.

    Args:
        X: Input waveforms, shape (batch, seq_len)

    Returns:
        Normalized waveforms, same shape as input
    """
    maxi = np.max(np.abs(X), axis=1, keepdims=True)
    maxi[maxi == 0] = 1.0
    return X / maxi

class PolarCAP(nn.Module):
    """
    PolarCAP model for first-motion polarity classification.
    
    Reference:
        Chakraborty, M. et al. PolarCAP-A deep learning approach for first motion
        polarity classification of earthquake waveforms. Artificial Intelligence in
        Geosciences 3, 46-52 (2022).
    
    Author:
        Model weights converted and maintained by He XingChen (Chinese, Han ethnicity),
        https://github.com/Chuan1937
    """
    def __init__(self, drop_rate=0.3):
        super(PolarCAP, self).__init__()
        
        # 1. Encoder part (based on Keras enc layer)
        # Input shape: (batch, 1, 64)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=32, padding=16), # padding='same'
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, padding=0), # (batch, 32, 32)
            
            nn.Conv1d(32, 8, kernel_size=16, padding=8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2, padding=0)  # (batch, 8, 16) -> this is the 'enc' layer
        )
        
        # 2. Decoder branch (for signal reconstruction dec)
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
            nn.Tanh() # Output reconstructed waveform (batch, 1, 64)
        )
        
        # 3. Polarity classification branch (Dense layer p)
        # Flattened size of enc output: 8 channels * 16 samples = 128
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 16, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: (batch, 1, 64)
        enc = self.encoder(x)
        
        # Task 1: Reconstruct waveform
        # PyTorch Conv1d same padding may cause length offset, manually crop to align to 64
        dec = self.decoder(enc)[:, :, :64]
        
        # Task 2: Polarity classification
        p = self.classifier(enc)
        
        return dec, p

    def predict(self, X_np):
        """Predict polarity labels for input waveforms.

        Args:
            X_np: Input waveforms, shape (batch, seq_len)

        Returns:
            List of (polarity_label, probability) tuples, where polarity_label is
            either 'Negative' or 'Positive'
        """
        self.eval()
        with torch.no_grad():
            # Preprocessing
            X_norm = norm(X_np)
            X_tensor = torch.FloatTensor(X_norm).unsqueeze(1) # (batch, 1, 64)
            
            # Inference
            _, p_probs = self.forward(X_tensor)
            
            # Result conversion
            pol_pred = torch.argmax(p_probs, dim=1).cpu().numpy()
            pred_prob = torch.max(p_probs, dim=1).values.cpu().numpy()
            
            polarity_labels = ['Negative', 'Positive']
            predictions = [(polarity_labels[pol], prob) for pol, prob in zip(pol_pred, pred_prob)]
            
        return predictions

class PolarCAPLoss(nn.Module):
    """
    Multi-task loss function for PolarCAP model.
    
    Computes MSE(dec_pred, inputs) + alpha * Huber(p_pred, labels)
    """
    # The paper uses 200 - alpha
    def __init__(self, alpha=10.0, delta=0.5):
        super().__init__()
        self.alpha = alpha
        self.huber_criterion = nn.HuberLoss(delta=delta)

    def forward(self, outputs, targets, inputs=None):
        """
        Args:
            outputs: Model output (dec_pred, p_pred)
            targets: Polarity labels (batch,)
            inputs: Original input waveforms (batch, 1, 64), used for reconstruction loss
        """
        if inputs is None:
            raise ValueError(
                "PolarCAPLoss requires 'inputs' (original waveforms) for "
                "reconstruction loss."
            )
        
        dec_pred, p_pred = outputs
        
        # Reconstruction loss: MSE
        mse_loss = F.mse_loss(dec_pred, inputs)

        # Classification loss: Huber (Keras implementation uses Huber for One-hot
        # labels, p_pred is (batch, 2))
        # First convert targets to one-hot to match p_pred (batch, 2)
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        huber_loss = self.huber_criterion(p_pred, targets_one_hot)
        
        return mse_loss + self.alpha * huber_loss

# --- Loss function definition (kept for backward compatibility) ---
def get_polarcap_loss(dec_pred, dec_true, p_pred, p_true):
    """Compute PolarCAP loss for backward compatibility.

    Args:
        dec_pred: Predicted reconstruction output
        dec_true: True input waveforms for reconstruction
        p_pred: Predicted polarity probabilities
        p_true: True polarity labels (one-hot encoded)

    Returns:
        Total loss (MSE reconstruction loss + 10 * Huber classification loss)
    """
    mse_loss = F.mse_loss(dec_pred, dec_true)
    # Huber loss in PyTorch defaults to delta=1.0, set to 0.5 based on Keras code
    huber_criterion = nn.HuberLoss(delta=0.5)
    huber_loss = huber_criterion(p_pred, p_true)
    
    # Adjust weights to more reasonable values
    total_loss = mse_loss + 10 * huber_loss
    return total_loss