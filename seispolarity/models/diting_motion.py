import torch
import torch.nn as nn
from .base import BasePolarityModel
from seispolarity.annotations import PickList

class ParallelConvs(nn.Module):
    """
    一个包含四个并行1D卷积层的模块。
    - 两个，kernel_size = 3
    - 两个，kernel_size = 5
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_k3_1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_k3_2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_k5_1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv_k5_2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        c1 = self.relu(self.conv_k3_1(x))
        c2 = self.relu(self.conv_k3_2(x))
        c3 = self.relu(self.conv_k5_1(x))
        c4 = self.relu(self.conv_k5_2(x))
        return c1, c2, c3, c4

class GrowthBlock(nn.Module):
    """
    应用 ParallelConvs，将其输出与输入拼接，然后应用 Dropout。
    """
    def __init__(self, in_channels, growth_rate, dropout_rate=0.2):
        super().__init__()
        self.parallel_convs = ParallelConvs(in_channels, growth_rate)
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
        pc1, pc2, pc3, pc4 = self.parallel_convs(x)
        x_out = torch.cat([pc1, pc2, pc3, pc4, x], dim=1)
        return self.dropout(x_out)

class ProcessingChain(nn.Module):
    """
    修正后的处理链模块，准确反映 Keras 模型中的串行结构。
    结构: Conv -> [GrowthBlock -> Conv] * depth
    """
    def __init__(self, initial_in_channels, bottleneck_channels, growth_rate, depth, dropout_rate=0.2):
        super().__init__()
        self.chain = nn.ModuleList()
        
        # 初始卷积层
        self.chain.append(nn.Sequential(
            nn.Conv1d(initial_in_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.ReLU()
        ))

        # 核心的重复单元
        for _ in range(depth):
            self.chain.append(GrowthBlock(bottleneck_channels, growth_rate, dropout_rate))
            growth_out_channels = bottleneck_channels + 4 * growth_rate
            self.chain.append(nn.Sequential(
                nn.Conv1d(growth_out_channels, bottleneck_channels, kernel_size=3, padding=1),
                nn.ReLU()
            ))

    def forward(self, x):
        for layer in self.chain:
            x = layer(x)
        return x

class OutputStream(nn.Module):
    """
    用于生成最终预测的输出流模块。
    结构: GrowthBlock -> Conv1D -> Flatten
    """
    def __init__(self, in_channels, dropout_rate=0.2):
        super().__init__()
        # 在输出流中，growth_rate (增长率) 总是 2
        self.growth = GrowthBlock(in_channels, growth_rate=2, dropout_rate=dropout_rate)
        
        # GrowthBlock 的输出通道数 = in_channels + 4 * 2
        growth_out_channels = in_channels + 8
        
        # 最终的卷积层输出通道数总是 2
        self.conv = nn.Conv1d(growth_out_channels, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.growth(x)
        x = self.relu(self.conv(x))
        x = self.flatten(x)
        return x
    

class DitingMotion(BasePolarityModel, nn.Module):
    def __init__(self, input_channels=2, dropout_rate=0.2, **kwargs):
        BasePolarityModel.__init__(self, name="DitingMotion", **kwargs)
        nn.Module.__init__(self)
        
        # 初始模块
        self.initial_convs = ParallelConvs(input_channels, 8)
        self.initial_dropout = nn.Dropout(p=dropout_rate)

        # --- 主干网络路径 ---
        # Block 1: 一个简化的处理链
        self.growth1 = GrowthBlock(34, 8)
        self.conv1_out = nn.Conv1d(34 + 4*8, 8, kernel_size=3, padding=1)
        self.relu1_out = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Block 2
        self.growth2 = GrowthBlock(in_channels=10, growth_rate=8)
        # 这里的 ProcessingChain 深度为 1
        self.proc_chain2 = ProcessingChain(initial_in_channels=42, bottleneck_channels=8, growth_rate=8, depth=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Block 3
        self.growth3 = GrowthBlock(in_channels=18, growth_rate=8)
        # 这里的 ProcessingChain 深度为 2
        self.proc_chain3 = ProcessingChain(initial_in_channels=50, bottleneck_channels=8, growth_rate=8, depth=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Block 4
        self.growth4 = GrowthBlock(in_channels=26, growth_rate=8)
        # 这里的 ProcessingChain 深度为 2
        self.proc_chain4 = ProcessingChain(initial_in_channels=58, bottleneck_channels=8, growth_rate=8, depth=2)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        # Block 5 (无池化层)
        self.growth5 = GrowthBlock(in_channels=34, growth_rate=8)
        # 这里的 ProcessingChain 深度为 2
        self.proc_chain5 = ProcessingChain(initial_in_channels=66, bottleneck_channels=8, growth_rate=8, depth=2)
        
        # --- 并行输出流 ---
        # 主输出流
        self.output_stream_1 = OutputStream(in_channels=26) # 连接 block3_pool
        self.output_stream_2 = OutputStream(in_channels=34) # 连接 block4_pool
        self.output_stream_3 = OutputStream(in_channels=42) # 连接 concatenate_17

        # Clarity 输出流 (结构相同)
        self.output_stream_1_clarity = OutputStream(in_channels=26)
        self.output_stream_2_clarity = OutputStream(in_channels=34)
        self.output_stream_3_clarity = OutputStream(in_channels=42)
        
        # --- 全连接层 ---
        # 主输出对应的全连接层
        self.dense_s1 = nn.Sequential(nn.Linear(32, 8), nn.ReLU()) # 输入维度: 16*2=32
        self.dense_s2 = nn.Sequential(nn.Linear(16, 8), nn.ReLU()) # 输入维度: 8*2=16
        self.dense_s3 = nn.Sequential(nn.Linear(16, 8), nn.ReLU()) # 输入维度: 8*2=16
        self.dense_fuse = nn.Sequential(nn.Linear(24, 8), nn.ReLU()) # 输入维度: 8+8+8=24
        
        self.o3 = nn.Linear(8, 3)
        self.o4 = nn.Linear(8, 3)
        self.o5 = nn.Linear(8, 3)
        self.ofuse = nn.Linear(8, 3)

        # Clarity 输出对应的全连接层
        self.dense_s1_clarity = nn.Sequential(nn.Linear(32, 8), nn.ReLU())
        self.dense_s2_clarity = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        self.dense_s3_clarity = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        self.dense_fuse_clarity = nn.Sequential(nn.Linear(64, 8), nn.ReLU()) # 输入维度: 32+16+16=64
        
        self.o3_clarity = nn.Linear(8, 3)
        self.o4_clarity = nn.Linear(8, 3)
        self.o5_clarity = nn.Linear(8, 3)
        self.ofuse_clarity = nn.Linear(8, 3)

    def forward(self, x):
        # PyTorch Conv1D 期望形状: (N, channels, length)
        # BasePolarityModel 已经提供了 (N, C, L)

        # --- 主干网络前向传播 ---
        # 初始模块
        c1, c2, c3, c4 = self.initial_convs(x)
        x_cat = torch.cat([c1, c2, c3, c4, x], dim=1) # (N, 34, 128)
        x_drop = self.initial_dropout(x_cat)

        # Block 1
        # Block 1的结构与后续块不同，需要单独处理
        x_g1 = self.growth1(x_drop)
        x_p1 = self.relu1_out(self.conv1_out(x_g1))
        x_res1 = torch.cat([x, x_p1], dim=1) # (N, 2+8=10, 128)
        block1_pool = self.pool1(x_res1) # (N, 10, 64)

        # Block 2
        x_g2 = self.growth2(block1_pool) # (N, 42, 64)
        x_p2 = self.proc_chain2(x_g2)
        x_res2 = torch.cat([block1_pool, x_p2], dim=1) # (N, 10+8=18, 64)
        block2_pool = self.pool2(x_res2) # (N, 18, 32)

        # Block 3
        x_g3 = self.growth3(block2_pool) # (N, 50, 32)
        x_p3 = self.proc_chain3(x_g3)
        x_res3 = torch.cat([block2_pool, x_p3], dim=1) # (N, 18+8=26, 32)
        block3_pool = self.pool3(x_res3) # (N, 26, 16)
        
        # Block 4
        x_g4 = self.growth4(block3_pool) # (N, 58, 16)
        x_p4 = self.proc_chain4(x_g4)
        x_res4 = torch.cat([block3_pool, x_p4], dim=1) # (N, 26+8=34, 16)
        block4_pool = self.pool4(x_res4) # (N, 34, 8)

        # Block 5
        x_g5 = self.growth5(block4_pool) # (N, 66, 8)
        x_p5 = self.proc_chain5(x_g5)
        x_res5 = torch.cat([block4_pool, x_p5], dim=1) # (N, 34+8=42, 8)
        
        # --- 输出流前向传播 ---
        # 主输出流
        out_s1 = self.output_stream_1(block3_pool)
        out_s2 = self.output_stream_2(block4_pool)
        out_s3 = self.output_stream_3(x_res5)

        # Clarity 输出流
        out_s1_clarity = self.output_stream_1_clarity(block3_pool)
        out_s2_clarity = self.output_stream_2_clarity(block4_pool)
        out_s3_clarity = self.output_stream_3_clarity(x_res5)
        
        # --- 全连接层前向传播 ---
        # 主输出
        dense_s1_out = self.dense_s1(out_s1)
        dense_s2_out = self.dense_s2(out_s2)
        dense_s3_out = self.dense_s3(out_s3)
        
        dense_fuse_in = torch.cat([dense_s1_out, dense_s2_out, dense_s3_out], dim=1)
        dense_fuse_out = self.dense_fuse(dense_fuse_in)
        
        o3 = self.o3(dense_s1_out)
        o4 = self.o4(dense_s2_out)
        o5 = self.o5(dense_s3_out)
        ofuse = self.ofuse(dense_fuse_out)

        # Clarity 输出
        dense_s1_clarity_out = self.dense_s1_clarity(out_s1_clarity)
        dense_s2_clarity_out = self.dense_s2_clarity(out_s2_clarity)
        dense_s3_clarity_out = self.dense_s3_clarity(out_s3_clarity)
        
        dense_fuse_clarity_in = torch.cat([out_s1_clarity, out_s2_clarity, out_s3_clarity], dim=1)
        dense_fuse_clarity_out = self.dense_fuse_clarity(dense_fuse_clarity_in)
        
        o3_clarity = self.o3_clarity(dense_s1_clarity_out)
        o4_clarity = self.o4_clarity(dense_s2_clarity_out)
        o5_clarity = self.o5_clarity(dense_s3_clarity_out)
        ofuse_clarity = self.ofuse_clarity(dense_fuse_clarity_out)

        # 返回所有输出，或者根据需要返回特定的输出
        # 这里返回所有8个输出，与Keras模型一致
        return o3, o4, o5, ofuse, o3_clarity, o4_clarity, o5_clarity, ofuse_clarity

    def forward_tensor(self, tensor: torch.Tensor, **kwargs):
        return self.forward(tensor)

    def build_picks(self, raw_output, **kwargs) -> PickList:
        return [] # Placeholder
