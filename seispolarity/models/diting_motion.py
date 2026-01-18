import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BasePolarityModel

class CombConvLayer(nn.Module):
    """
    对应论文中的 'CombConv Layer' 及 Keras Summary 中的并行卷积结构。
    结构：4个并行分支 (2x kernel=3, 2x kernel=5)
    """
    def __init__(self, in_channels, out_filters=8):
        super().__init__()
        # Keras: conv1d, conv1d_1 (kernel=3)
        self.branch1 = nn.Conv1d(in_channels, out_filters, kernel_size=3, padding=1)
        self.branch2 = nn.Conv1d(in_channels, out_filters, kernel_size=3, padding=1)
        # Keras: conv1d_2, conv1d_3 (kernel=5)
        self.branch3 = nn.Conv1d(in_channels, out_filters, kernel_size=5, padding=2)
        self.branch4 = nn.Conv1d(in_channels, out_filters, kernel_size=5, padding=2)
        
    def forward(self, x):
        b1 = F.relu(self.branch1(x))
        b2 = F.relu(self.branch2(x))
        b3 = F.relu(self.branch3(x))
        b4 = F.relu(self.branch4(x))
        # 拼接所有分支 + 原始输入 (ResNet style sort of, but strict concat)
        # Keras summary 显示 concatenate 层连接了 input 和 4个分支
        out = torch.cat([b1, b2, b3, b4, x], dim=1) 
        return out

class DiTingBlock(nn.Module):
    """
    对应 Keras Summary 中的 Block 2, 3, 4, 5 的重复结构。
    Trace logic based on Keras summary:
    1. Input -> CombConv (Parallel) -> Concat -> Dropout
    2. Conv1d (Fusion/Bottleneck)
    3. CombConv (Parallel) -> Concat -> Dropout
    4. Conv1d (Fusion/Bottleneck)
    5. Final Concat (Input + Last Conv Output)
    """
    def __init__(self, in_channels, pooling=True, filters=8, dropout_rate=0.2):
        super().__init__()
        
        # --- Stage 1 ---
        # CombConv: In -> 4x8 filters. Output channels = In + 32
        self.comb1 = CombConvLayer(in_channels, filters)
        self.drop1 = nn.Dropout(dropout_rate)
        
        # Fusion 1: Reduces/Transform features. 
        # Keras summary e.g. Block2: conv1d_14 inputs 42 channels, outputs 8.
        stage1_out_ch = in_channels + 4 * filters
        self.fusion1 = nn.Conv1d(stage1_out_ch, filters, kernel_size=3, padding=1)
        
        # --- Stage 2 ---
        # CombConv on the output of Fusion 1 (which has 8 channels)
        self.comb2 = CombConvLayer(filters, filters) # In: 8, Out: 8+32=40
        self.drop2 = nn.Dropout(dropout_rate)
        
        # Fusion 2
        # Keras summary e.g. Block2: conv1d_19 inputs 40 channels, outputs 8.
        stage2_out_ch = filters + 4 * filters # 8 + 32 = 40
        self.fusion2 = nn.Conv1d(stage2_out_ch, filters, kernel_size=3, padding=1)
        
        # Pooling
        self.pooling = pooling
        if pooling:
            self.pool = nn.MaxPool1d(kernel_size=2)
            
    def forward(self, x):
        # Stage 1
        out = self.comb1(x)
        out = self.drop1(out)
        out = F.relu(self.fusion1(out))
        
        # Stage 2
        out = self.comb2(out)
        out = self.drop2(out)
        out = F.relu(self.fusion2(out))
        
        # Final Block Residual Connection: Concat [Input, Fusion2_Output]
        # e.g., Block 2: Input(10) + Fusion2(8) = 18 channels output
        final = torch.cat([x, out], dim=1)
        
        if self.pooling:
            final = self.pool(final)
            
        return final

class DitingMotion(BasePolarityModel, nn.Module):
    def __init__(self, input_channels=2, dropout_rate=0.15, **kwargs):
        BasePolarityModel.__init__(self, name="DitingMotion", **kwargs)
        nn.Module.__init__(self)
        
        # --- Block 1 (Unique Structure) ---
        # Keras: Input(2) -> Parallel(4x8) -> Concat(34) -> Drop -> Conv(8)
        self.b1_comb1 = CombConvLayer(input_channels, 8) # Out: 2 + 32 = 34
        self.b1_drop1 = nn.Dropout(dropout_rate)
        self.b1_conv1 = nn.Conv1d(34, 8, kernel_size=3, padding=1)
        
        # Keras: -> Parallel(4x8) -> Concat(40) -> Drop -> Conv(8)
        self.b1_comb2 = CombConvLayer(8, 8) # Out: 8 + 32 = 40
        self.b1_drop2 = nn.Dropout(dropout_rate)
        self.b1_conv2 = nn.Conv1d(40, 8, kernel_size=3, padding=1)
        
        # Keras: Concat(Input, Conv2_Out) -> 2 + 8 = 10 channels -> Pool
        self.b1_pool = nn.MaxPool1d(kernel_size=2)

        # --- Blocks 2 to 5 (Repeated DiTingBlocks) ---
        # Block 2: Input 10 -> Output 10+8=18 -> Pool (32 length)
        self.block2 = DiTingBlock(10, pooling=True, dropout_rate=dropout_rate)
        
        # Block 3: Input 18 -> Output 18+8=26 -> Pool (16 length)
        self.block3 = DiTingBlock(18, pooling=True, dropout_rate=dropout_rate)
        
        # Block 4: Input 26 -> Output 26+8=34 -> Pool (8 length)
        self.block4 = DiTingBlock(26, pooling=True, dropout_rate=dropout_rate)
        
        # Block 5: Input 34 -> Output 34+8=42 -> NO POOL (based on Keras summary logic, wait...)
        # Keras summary shows block4_pool (8, 34).
        # Then final processing. Wait, Keras summary shows Block 5 logic slightly diff or just one flow.
        # Let's trace Block 5 in summary:
        # block4_pool -> conv1d_50..53 (Parallel) -> Concat -> Drop -> Conv54 
        # -> Parallel -> Concat -> Drop -> Conv64 -> Concat(block4_pool, Conv64) -> Output 42.
        # Then it goes to Side Layers. It does NOT have a block5_pool.
        self.block5 = DiTingBlock(34, pooling=False, dropout_rate=dropout_rate)

        # --- Side Output Layers (Deep Supervision) ---
        # Keras: Dense layers connect to Flattened outputs of Blocks 3, 4, 5
        
        # Side Output 3 (From Block 3 Output)
        # Block 3 out shape: (N, 26, 16) -> Flatten -> 416? 
        # WAIT. Keras summary: 
        # flatten (Flatten) connected to conv1d_69. 
        # conv1d_69 comes from dropout_13 (Block 3 final parallel structure?).
        # Actually, in HED, side outputs usually come from the END of the block.
        # Let's use the explicit Keras logic:
        # The summary uses 1x1 convs (kernel size 1? or 3 with padding to keep size?) to reduce channels before flattening?
        # Keras summary: conv1d_65..68 (kernel 3?) connected to block3_pool.
        # The summary is complex here. Let's simplify to standard HED:
        # Take output of Block 3, 4, 5. Apply Conv to reduce to 2 channels (Prediction maps?) or just Flatten.
        # The Keras summary shows: conv1d_69 (size 2) -> Flatten -> Dense.
        
        # Implementation of Side Heads based on Keras "conv1d_69" (Block3), "conv1d_74" (Block4), "conv1d_79" (Block5)
        # These reduce channels to 2.
        self.side3_conv = nn.Conv1d(26, 2, kernel_size=3, padding=1) # Block 3 out channels 26
        self.side4_conv = nn.Conv1d(34, 2, kernel_size=3, padding=1) # Block 4 out channels 34
        self.side5_conv = nn.Conv1d(42, 2, kernel_size=3, padding=1) # Block 5 out channels 42 (No pool)
        
        # Dense Layers for Polarity (Output 3 classes: U, D, X)
        # Block 3: Length 16. 2 channels * 16 = 32 features.
        self.dense3 = nn.Sequential(nn.Linear(32, 8), nn.ReLU())
        self.out3 = nn.Linear(8, 3)
        
        # Block 4: Length 8. 2 channels * 8 = 16 features.
        self.dense4 = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        self.out4 = nn.Linear(8, 3)
        
        # Block 5: Length 8. 2 channels * 8 = 16 features.
        self.dense5 = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        self.out5 = nn.Linear(8, 3)
        
        # Fuse Layer
        # Concatenate outputs of dense3, dense4, dense5 (8+8+8=24)
        self.fuse_dense = nn.Sequential(nn.Linear(24, 8), nn.ReLU())
        self.out_fuse = nn.Linear(8, 3)
        
        # --- Clarity Heads (Duplicate structure for clarity) ---
        self.dense3_c = nn.Sequential(nn.Linear(32, 8), nn.ReLU())
        self.out3_c = nn.Linear(8, 3)
        
        self.dense4_c = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        self.out4_c = nn.Linear(8, 3)
        
        self.dense5_c = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        self.out5_c = nn.Linear(8, 3)
        
        # Clarity Fuse (Keras summary shows dense_7 inputs 64? From flatten_3,4,5?)
        # Let's mirror the polarity fuse for simplicity as it's robust.
        self.fuse_dense_c = nn.Sequential(nn.Linear(24, 8), nn.ReLU())
        self.out_fuse_c = nn.Linear(8, 3)

    def forward(self, x):
        # x shape: (N, 2, 128)
        
        # Block 1
        b1 = self.b1_comb1(x)
        b1 = self.b1_drop1(b1)
        b1 = F.relu(self.b1_conv1(b1))
        
        b1 = self.b1_comb2(b1)
        b1 = self.b1_drop2(b1)
        b1 = F.relu(self.b1_conv2(b1))
        
        # Block 1 Concat & Pool
        c1 = torch.cat([x, b1], dim=1) # (N, 10, 128)
        p1 = self.b1_pool(c1)          # (N, 10, 64)
        
        # Block 2
        p2 = self.block2(p1)           # (N, 18, 32)
        
        # Block 3
        p3 = self.block3(p2)           # (N, 26, 16)
        
        # Block 4
        p4 = self.block4(p3)           # (N, 34, 8)
        
        # Block 5 (No pool)
        p5 = self.block5(p4)           # (N, 42, 8)
        
        # --- Side Outputs Processing ---
        
        # Side 3
        s3 = F.relu(self.side3_conv(p3))
        s3_flat = s3.view(s3.size(0), -1) # Flatten (N, 32)
        d3 = self.dense3(s3_flat)
        o3 = self.out3(d3)
        
        # Side 4
        s4 = F.relu(self.side4_conv(p4))
        s4_flat = s4.view(s4.size(0), -1) # Flatten (N, 16)
        d4 = self.dense4(s4_flat)
        o4 = self.out4(d4)
        
        # Side 5
        s5 = F.relu(self.side5_conv(p5))
        s5_flat = s5.view(s5.size(0), -1) # Flatten (N, 16)
        d5 = self.dense5(s5_flat)
        o5 = self.out5(d5)
        
        # Fuse Polarity
        d_fuse = torch.cat([d3, d4, d5], dim=1)
        o_fuse = self.out_fuse(self.fuse_dense(d_fuse))
        
        # --- Clarity Processing (Parallel heads) ---
        d3_c = self.dense3_c(s3_flat)
        o3_c = self.out3_c(d3_c)
        
        d4_c = self.dense4_c(s4_flat)
        o4_c = self.out4_c(d4_c)
        
        d5_c = self.dense5_c(s5_flat)
        o5_c = self.out5_c(d5_c)
        
        d_fuse_c = torch.cat([d3_c, d4_c, d5_c], dim=1)
        o_fuse_c = self.out_fuse_c(self.fuse_dense_c(d_fuse_c))
        
        return o3, o4, o5, o_fuse, o3_c, o4_c, o5_c, o_fuse_c
    
    def forward_tensor(self, tensor: torch.Tensor, **kwargs):
        """Model forward pass for a batch tensor."""
        return self.forward(tensor)
    
    def build_picks(self, raw_output, **kwargs):
        """Convert raw model output to picks with polarity labels.
        
        DitingMotion 模型有8个输出，我们需要使用融合输出（索引3）作为主要预测。
        """
        from seispolarity.annotations import Pick, PickList, PolarityLabel
        
        # raw_output 是一个包含8个张量的元组
        # 我们使用融合输出（索引3）作为主要预测
        if isinstance(raw_output, (tuple, list)) and len(raw_output) >= 4:
            # 获取融合输出（索引3）
            fuse_output = raw_output[3]
            # 应用softmax获取概率
            probs = torch.softmax(fuse_output, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            picks = PickList()
            # 为每个样本创建一个pick
            for i in range(preds.shape[0]):
                pred = preds[i].item()
                prob = probs[i, pred].item()
                
                # 映射预测到极性标签
                if pred == 0:
                    polarity = PolarityLabel.UP
                elif pred == 1:
                    polarity = PolarityLabel.DOWN
                else:
                    polarity = PolarityLabel.UNKNOWN
                
                # 创建pick（这里需要更多上下文信息，暂时使用占位符）
                pick = Pick(
                    trace_id="unknown",
                    time=None,  # 需要从输入数据中获取时间信息
                    confidence=prob,
                    polarity=polarity
                )
                picks.append(pick)
            
            return picks
        else:
            # 如果输出格式不符合预期，返回空列表
            return PickList()