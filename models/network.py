import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入所有需要的模块，包括刚写的 FrequencyHardConstraint
from .blocks import MultiScaleBlock3D, SFTLayer3D, EfficientContextBlock, FrequencyHardConstraint

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
        self.sft1 = SFTLayer3D(dim)
        self.res1 = ConvBlock3D(dim, dim)
        self.sft2 = SFTLayer3D(dim)
        self.res2 = ConvBlock3D(dim, dim)
        
        # 4. 全局上下文 (高效版)
        self.global_context = EfficientContextBlock(dim)
        
        # 5. 重建层
        self.tail = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv3d(dim, 1, 3, 1, 1)
        )
        
        # 🔥 6. 新增：SEN2SR 频率硬约束层
        # radius=10 表示保留中心 10 个像素半径的低频信息不被修改
        self.constraint = FrequencyHardConstraint(radius=10)

    def forward(self, aux, main):
        # Feature Extraction
        f_aux = self.aux_head(aux)
        f_aux = self.aux_multiscale(f_aux) 
        
        f_main = self.main_head(main)
        
        # SFT Fusion
        f_main = self.sft1(f_main, f_aux)
        f_main = self.res1(f_main) + f_main
        f_main = self.sft2(f_main, f_aux)
        f_main = self.res2(f_main) + f_main
        
        # Global Context
        f_global = self.global_context(f_main)
        f_final = f_main + f_global
        
        # Reconstruction
        residual = self.tail(f_final)
        pred = F.relu(main + residual)
        
        # 🔥 7. 最后一步：应用硬约束
        # 强制 Pred 的低频部分必须和 Main 一样
        final_output = self.constraint(pred, main)
        
        return final_output