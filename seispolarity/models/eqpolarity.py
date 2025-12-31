import torch
import torch.nn as nn
import numpy as np
from .base import BasePolarityModel
from seispolarity.annotations import PickList

class MLP(nn.Module):
    """
    PyTorch版本的前馈神经网络 (MLP Block)。
    对应 Keras 的 mlp 函数。
    """

    def __init__(self, in_features, hidden_units, dropout_rate):
        super().__init__()
        layers = []
        for units in hidden_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.GELU())  # Keras中使用的是gelu
            layers.append(nn.Dropout(dropout_rate))
            in_features = units
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class StochasticDepth(nn.Module):
    """
    PyTorch版本的随机深度层。
    对应 Keras 的 StochasticDepth 类。
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
        random_tensor.floor_()  # 二值化

        return x.div(keep_prob) * random_tensor


class CCTTokenizer(nn.Module):
    """
    PyTorch版本的卷积令牌化器。
    对应 Keras 的 CCTTokenizer1 类。
    """

    def __init__(self, kernel_size=4, stride=1, padding=1, pooling_kernel_size=3, num_conv_layers=2,
                 num_output_channels=None, projection_dim=200):
        super().__init__()

        if num_output_channels is None:
            num_output_channels = [projection_dim] * num_conv_layers

        layers = []
        in_channels = 1  # 初始通道数为1
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
        # 输入 x 的 shape: (batch, channels=1, length)
        # 已经是 PyTorch Conv1d 需要的 shape: (batch, channels, length)
        
        # 通过卷积模型
        x = self.conv_model(x)

        # 转换回 Transformer 需要的 shape: (batch, new_length, features)
        x = x.permute(0, 2, 1)
        return x


class TransformerBlock(nn.Module):
    """
    PyTorch版本的单个Transformer编码器模块。
    """

    def __init__(self, projection_dim, num_heads, mlp_hidden_units, dropout_rate, stochastic_depth_rate):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(projection_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True  # 让输入输出都是 (batch, seq, feature)
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
        # 注意力块
        x_norm1 = self.layer_norm1(x)
        attn_output, _ = self.attention(x_norm1, x_norm1, x_norm1)
        x = x + self.stochastic_depth1(attn_output)  # 残差连接

        # MLP块
        x_norm2 = self.layer_norm2(x)
        mlp_output = self.mlp(x_norm2)
        x = x + self.stochastic_depth2(mlp_output)  # 残差连接

        return x


class EQPolarityCCT(BasePolarityModel, nn.Module):
    """
    完整的、对应 construct_model 的 PyTorch CCT 模型。
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

        # 1. 卷积令牌化器
        self.tokenizer = CCTTokenizer(projection_dim=projection_dim)

        # 2. Transformer编码器堆叠
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, transformer_layers)]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(projection_dim, num_heads, mlp_hidden_units, dropout_rate, dpr[i])
            for i in range(transformer_layers)
        ])

        # 3. 输出层
        self.layer_norm_final = nn.LayerNorm(projection_dim)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)

        # 动态计算最终展平后的维度
        # 假设输入长度经过两次stride=2的pooling，长度变为 input_length / 4
        final_flatten_dim = (input_length // 4) * projection_dim
        self.output_layer = nn.Linear(final_flatten_dim, 1)

    def forward(self, x):
        # 输入 shape: (batch, 1, length)
        x = self.tokenizer(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.layer_norm_final(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.output_layer(x)

        # 在PyTorch中, 推荐使用 nn.BCEWithLogitsLoss, 它内置了sigmoid，更稳定。
        # 因此模型本身不包含最后的sigmoid激活。
        return x

    def forward_tensor(self, tensor: torch.Tensor, **kwargs):
        return self.forward(tensor)

    def build_picks(self, raw_output, **kwargs) -> PickList:
        return [] # Placeholder
