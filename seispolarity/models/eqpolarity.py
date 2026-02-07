import torch
import torch.nn as nn
import numpy as np
from .base import BasePolarityModel
from seispolarity.annotations import PickList

class MLP(nn.Module):
    """
    PyTorch version of feedforward neural network (MLP Block).
    Corresponds to the mlp function in Keras.
    """

    def __init__(self, in_features, hidden_units, dropout_rate):
        super().__init__()
        layers = []
        for units in hidden_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.GELU())  # Keras uses gelu
            layers.append(nn.Dropout(dropout_rate))
            in_features = units
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class StochasticDepth(nn.Module):
    """
    PyTorch version of stochastic depth layer.
    Corresponds to the StochasticDepth class in Keras.
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarization

        return x.div(keep_prob) * random_tensor


class CCTTokenizer(nn.Module):
    """
    PyTorch version of convolutional tokenizer.
    Corresponds to the CCTTokenizer1 class in Keras.
    """

    def __init__(self, kernel_size=4, stride=1, padding=1, pooling_kernel_size=3, num_conv_layers=2,
                 num_output_channels=None, projection_dim=200):
        super().__init__()

        if num_output_channels is None:
            num_output_channels = [projection_dim] * num_conv_layers

        layers = []
        in_channels = 1  # Initial number of channels is 1
        for i in range(num_conv_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=num_output_channels[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=2, padding=1))
            in_channels = num_output_channels[i]

        self.conv_model = nn.Sequential(*layers)

    def forward(self, x):
        # Input x shape: (batch, channels=1, length)
        # Already in the shape required by PyTorch Conv1d: (batch, channels, length)
        
        # Pass through convolutional model
        x = self.conv_model(x)

        # Convert back to shape required by Transformer: (batch, new_length, features)
        x = x.permute(0, 2, 1)
        return x


class TransformerBlock(nn.Module):
    """
    PyTorch version of single Transformer encoder module.
    """

    def __init__(self, projection_dim, num_heads, mlp_hidden_units, dropout_rate, stochastic_depth_rate):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(projection_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True  # Let input and output be (batch, seq, feature)
        )
        self.stochastic_depth1 = StochasticDepth(stochastic_depth_rate)

        self.layer_norm2 = nn.LayerNorm(projection_dim)
        self.mlp = MLP(
            in_features=projection_dim,
            hidden_units=mlp_hidden_units,
            dropout_rate=dropout_rate
        )
        self.stochastic_depth2 = StochasticDepth(stochastic_depth_rate)

    def forward(self, x):
        # Attention block
        x_norm1 = self.layer_norm1(x)
        attn_output, _ = self.attention(x_norm1, x_norm1, x_norm1)
        x = x + self.stochastic_depth1(attn_output)  # Residual connection

        # MLP block
        x_norm2 = self.layer_norm2(x)
        mlp_output = self.mlp(x_norm2)
        x = x + self.stochastic_depth2(mlp_output)  # Residual connection

        return x


class EQPolarityCCT(BasePolarityModel, nn.Module):
    """
    Complete PyTorch CCT model corresponding to construct_model.
    
    Reference:
        Chen, Y. et al. Deep learning for P-wave first-motion polarity determination and its application to focal mechanism inversion.
        IEEE Transactions on Geoscience and Remote Sensing (2024).
    
    Author:
        Model weights converted and maintained by He XingChen (Chinese, Han ethnicity), https://github.com/Chuan1937
    """

    def __init__(self,
                 input_length=200,
                 projection_dim=200,
                 num_heads=4,
                 transformer_layers=4,
                 mlp_hidden_units=None,
                 dropout_rate=0.2,
                 stochastic_depth_rate=0.1,
                 **kwargs):
        BasePolarityModel.__init__(self, name="EQPolarityCCT", **kwargs)
        nn.Module.__init__(self)

        if mlp_hidden_units is None:
            mlp_hidden_units = [projection_dim, projection_dim]

        # 1. Convolutional tokenizer
        self.tokenizer = CCTTokenizer(projection_dim=projection_dim)

        # 2. Transformer encoder stack
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, transformer_layers)]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(projection_dim, num_heads, mlp_hidden_units, dropout_rate, dpr[i])
            for i in range(transformer_layers)
        ])

        # 3. Output layer
        self.layer_norm_final = nn.LayerNorm(projection_dim)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)

        # Dynamically calculate final flattened dimension
        # Assuming input length goes through two stride=2 pooling, length becomes input_length / 4
        final_flatten_dim = (input_length // 4) * projection_dim
        self.output_layer = nn.Linear(final_flatten_dim, 1)

    def forward(self, x):
        # Input shape: (batch, 1, length)
        x = self.tokenizer(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.layer_norm_final(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.output_layer(x)

        # In PyTorch, it is recommended to use nn.BCEWithLogitsLoss, which includes sigmoid and is more stable.
        # Therefore, the model itself does not include the final sigmoid activation.
        return x

    def forward_tensor(self, tensor: torch.Tensor, **kwargs):
        return self.forward(tensor)

    def build_picks(self, raw_output, **kwargs) -> PickList:
        return [] # Placeholder
