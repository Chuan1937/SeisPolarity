import torch
import torch.nn as nn
from .base import BasePolarityModel
from seispolarity.annotations import PickList

class PPNet(BasePolarityModel, nn.Module):
    """
    A PyTorch implementation of the `build_PP_model` from the provided Keras code.

    This model features a U-Net-like architecture with an LSTM and a Multi-Head 
    Self-Attention layer in the bottleneck. It produces two outputs:
    1. A segmentation-style mask (out1).
    2. A classification prediction (out2).
    
    Reference:
        Song, J., Zhu, W., Zi, J., Yang, H. & Chu, R. An Enhanced Focal Mechanism Catalog of Induced Earthquakes in Weiyuan, Sichuan, 
        from Dense Array Data and a Multitask Deep Learning Model. The Seismic Record 5, 175-184 (2025).
        Zhu, W., Tai, K. S., Mousavi, S. M., Bailis, P. & Beroza, G. C. An End-to-End Earthquake Detection Method for Joint Phase 
        Picking and Association using Deep Learning. JGR Solid Earth 127, e2021JB023283 (2022).
    
    Author:
        Model weights converted and maintained by He XingChen (Chinese, Han ethnicity), https://github.com/Chuan1937
    """
    def __init__(self, input_len=400, input_channels=1, num_classes=3, 
                 filter_sizes=[16, 32, 64], kernel_size=3, num_dense=128, lstm_units=None,
                 attention_heads=4, **kwargs):
        """
        Initializes the model layers.

        Args:
            input_len (int): The length of the input sequence.
            input_channels (int): The number of channels in the input sequence (e.g., 1 or 2).
            num_classes (int): The number of classes for the classification output.
            filter_sizes (list): A list of three integers for the number of filters in each Conv1D block.
            kernel_size (int): The kernel size for the convolutional layers.
            num_dense (int): The number of units in the dense layer for the classification head.
            lstm_units (int, optional): The number of units in the LSTM layer. If None, it defaults to 
                                       2 * filter_sizes[2].
            attention_heads (int): The number of heads for the multi-head self-attention mechanism.
        """
        BasePolarityModel.__init__(self, name="PPNet", **kwargs)
        nn.Module.__init__(self)

        # If lstm_units is not specified, calculate it as in the Keras model
        if lstm_units is None:
            lstm_units = filter_sizes[2] * 2

        # --- Encoder Path ---
        self.encoder_block1 = nn.Sequential(
            nn.Conv1d(input_channels, filter_sizes[0], kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(filter_sizes[0])
        )
        self.encoder_block2 = nn.Sequential(
            nn.Conv1d(filter_sizes[0], filter_sizes[1], kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(filter_sizes[1])
        )
        self.encoder_block3 = nn.Sequential(
            nn.Conv1d(filter_sizes[1], filter_sizes[2], kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(filter_sizes[2])
        )

        # --- Bottleneck ---
        self.lstm = nn.LSTM(input_size=filter_sizes[2], hidden_size=lstm_units, 
                            batch_first=True, bidirectional=False)
        
        # NOTE: Replacing `keras_self_attention` with PyTorch's standard MultiheadAttention.
        # This is the standard equivalent for self-attention.
        self.attention = nn.MultiheadAttention(embed_dim=lstm_units, num_heads=attention_heads, 
                                               batch_first=True)

        # --- Decoder Path (Segmentation Head) ---
        self.decoder_block1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(lstm_units, filter_sizes[2], kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(filter_sizes[2])
        )
        self.decoder_block2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(filter_sizes[2], filter_sizes[1], kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(filter_sizes[1])
        )
        self.decoder_block3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(filter_sizes[1], filter_sizes[0], kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(filter_sizes[0])
        )
        self.out1_conv = nn.Conv1d(filter_sizes[0], 1, kernel_size=kernel_size, padding='same')
        self.out1_activation = nn.Sigmoid()

        # --- Classifier Path (Classification Head) ---
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=lstm_units * (input_len // 8), out_features=num_dense)
        self.relu_dense = nn.ReLU()
        self.out2_dense = nn.Linear(in_features=num_dense, out_features=num_classes)
        self.out2_activation = nn.Softmax(dim=1)


    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, sequence_length).
        
        Returns:
            tuple: A tuple containing two output tensors:
                   - out1 (torch.Tensor): Segmentation output of shape (batch_size, sequence_length, 1).
                   - out2 (torch.Tensor): Classification output of shape (batch_size, num_classes).
        """
        # PyTorch Conv1D expects (batch, channels, length)
        # Input x is already (batch, channels, length) from BasePolarityModel

        # --- Encoder ---
        x = self.encoder_block1(x)  # Shape: (batch, 16, 200)
        x = self.encoder_block2(x)  # Shape: (batch, 32, 100)
        x = self.encoder_block3(x)  # Shape: (batch, 64, 50)

        # --- Bottleneck ---
        # Permute for LSTM: (batch, length, channels)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        
        # Self-Attention layer
        # Query, Key, and Value are all the same for self-attention
        at_x, attn_weights = self.attention(x, x, x)

        # --- Decoder Path ---
        # Permute for Conv1D: (batch, channels, length)
        x1 = at_x.permute(0, 2, 1)
        x1 = self.decoder_block1(x1)
        x1 = self.decoder_block2(x1)
        x1 = self.decoder_block3(x1)
        
        out1 = self.out1_conv(x1)
        # Permute back to (batch, length, 1) for consistency with Keras output
        out1 = out1.permute(0, 2, 1)
        out1 = self.out1_activation(out1)

        # --- Classifier Path ---
        # Flatten expects (batch, features), but at_x is (batch, length, features)
        # We need to flatten the last two dimensions
        x2 = self.flatten(at_x) 
        x2 = self.dense(x2)
        x2 = self.relu_dense(x2)
        out2 = self.out2_dense(x2)
        out2 = self.out2_activation(out2)

        return out1, out2

    def forward_tensor(self, tensor: torch.Tensor, **kwargs):
        return self.forward(tensor)

    def build_picks(self, raw_output, **kwargs) -> PickList:
        # raw_output is (out1, out2)
        # out1: segmentation
        # out2: classification
        return [] # Placeholder
