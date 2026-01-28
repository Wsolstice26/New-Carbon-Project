# models/network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ 修改1: 从 blocks 里删掉了 SimpleMambaBlock，只保留其他的
from .blocks import MultiScaleBlock3D, SFTLayer3D, EfficientContextBlock, FrequencyHardConstraint, MoEBlock

# ✅ 修改2: 导入官方 Mamba 库
from mamba_ssm import Mamba

# 保持 ConvBlock3D 不变
class ConvBlock3D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, 1, 1),
            nn.PReLU(),
            nn.Conv3d(out_c, out_c, 3, 1, 1)
        )
    def forward(self, x): return self.conv(x)

class DSTCarbonFormer(nn.Module):
    def __init__(self, aux_c=9, main_c=1, dim=64):
        super().__init__()
        
        # 1. 辅助流编码器
        self.aux_head = nn.Conv3d(aux_c, dim, 3, 1, 1)
        self.aux_multiscale = MultiScaleBlock3D(dim) 
        
        # 2. 主流编码器
        self.main_head = nn.Conv3d(main_c, dim, 3, 1, 1)
        
        # 3. 双流融合 (SFT Fusion)
        # Stage 1: 标准 SFT + ResBlock
        self.sft1 = SFTLayer3D(dim)
        self.res1 = ConvBlock3D(dim, dim)
        
        # 🔥 Stage 2: 升级为 SFT + MoE Block (容量更大)
        self.sft2 = SFTLayer3D(dim)
        self.moe_block = MoEBlock(dim, num_experts=3, top_k=1) # 使用 MoE 替代普通 ResBlock
        
        # 4. 全局上下文 -> 升级为 Mamba 增强
        self.global_context = EfficientContextBlock(dim)
        
        # ✅ 修改3: 使用官方 Mamba 初始化
        # 这里的 dim 对应输入通道数 (d_model)
        self.mamba_block = Mamba(
            d_model=dim, 
            d_state=16,  # 内部状态维度，官方默认16
            d_conv=4,    # 局部卷积宽度，官方默认4
            expand=2     # 扩展系数，官方默认2
        )
        
        # 5. 重建层
        self.tail = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv3d(dim, 1, 3, 1, 1)
        )
        
        # 6. 频率硬约束
        self.constraint = FrequencyHardConstraint(radius=10)

    def forward(self, aux, main):
        # Feature Extraction
        f_aux = self.aux_head(aux)
        f_aux = self.aux_multiscale(f_aux) 
        
        f_main = self.main_head(main)
        
        # Stage 1 Fusion
        f_main = self.sft1(f_main, f_aux)
        f_main = self.res1(f_main) + f_main
        
        # Stage 2 Fusion (MoE)
        f_main = self.sft2(f_main, f_aux)
        f_main = self.moe_block(f_main) # 🔥 经过 MoE
        
        # Global Context (Mamba)
        f_global = self.global_context(f_main)
        
        # ✅ 修改4: 数据形状适配 (关键步骤)
        # 官方 Mamba 需要 (Batch, Length, Dim) 的输入
        # 我们的 f_global 是 3D 图像格式 (Batch, Dim, T, H, W)
        # 所以必须把 T, H, W 展平 (Flatten) 才能喂进去
        
        B, C, T, H, W = f_global.shape
        
        # 1. 变形: (B, C, T, H, W) -> (B, C, T*H*W) -> (B, T*H*W, C)
        x_mamba = f_global.flatten(2).transpose(1, 2)
        
        # 2. 进官方 Mamba 跑一圈 (享受 CUDA 加速)
        x_mamba = self.mamba_block(x_mamba)
        
        # 3. 还原: (B, T*H*W, C) -> (B, C, T*H*W) -> (B, C, T, H, W)
        f_mamba = x_mamba.transpose(1, 2).view(B, C, T, H, W)
        
        # ---------------------------------------------
        
        f_final = f_main + f_mamba
        
        # Reconstruction
        residual = self.tail(f_final)
        pred = F.relu(main + residual)
        
        # Hard Constraint
        final_output = self.constraint(pred, main)
        
        return final_output