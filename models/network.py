# models/network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ 修改1: 移除从 blocks 导入 FrequencyHardConstraint，改为在本地定义以确保逻辑准确
# 保留其他模块的导入
from .blocks import MultiScaleBlock3D, SFTLayer3D, EfficientContextBlock, MoEBlock

# ✅ 修改2: 鲁棒性导入 Mamba
try:
    from mamba_ssm import Mamba
except ImportError:
    print("⚠️ [Network] 未找到 mamba_ssm，将使用 Identity 代替 (仅供调试)")
    Mamba = None

# ==========================================
# 🔥 新增: 本地定义的物理硬约束层
# ==========================================
class FrequencyHardConstraint(nn.Module):
    """
    物理硬约束层 (Physical Hard Constraint):
    强制 Prediction 的低频部分 (Low Frequency) 严格等于 Input 的低频部分。
    
    原理：
    Input 是 4km 的马赛克数据 (由 1km 降采样而来)，它丢失了高频细节，
    但在低频（宏观总量）上是物理守恒的。
    因此，我们强制 Output 在低频段与 Input 保持一致，只允许模型生成高频细节。
    """
    def __init__(self, radius=16):
        super().__init__()
        self.radius = radius

    def forward(self, pred, low_res_input):
        # 1. FFT 变换到频域 (Batch, C, T, H, W) -> (Batch, C, T, H, W) 复数
        pred_fft = torch.fft.fft2(pred)
        input_fft = torch.fft.fft2(low_res_input)
        
        # 2. 创建低频掩码 (Low Pass Mask)
        B, C, T, H, W = pred.shape
        cy, cx = H // 2, W // 2
        
        # 生成网格坐标
        y = torch.arange(H).to(pred.device)
        x = torch.arange(W).to(pred.device)
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        
        # 计算到中心的距离 (频谱搬移后中心是低频)
        # 注意：这里我们假设 H, W 是空间维度
        dist = torch.sqrt((y_grid - cy)**2 + (x_grid - cx)**2)
        
        # 生成 Mask (1 表示低频区域，0 表示高频区域)
        mask = (dist <= self.radius).float().view(1, 1, 1, H, W)
        
        # 3. 频谱搬移 (Shift) 让低频来到中心
        pred_fft_shifted = torch.fft.fftshift(pred_fft, dim=(-2, -1))
        input_fft_shifted = torch.fft.fftshift(input_fft, dim=(-2, -1))
        
        # 4. 🔥 核心操作: 替换低频
        # 用 Input 的低频 + Pred 的高频
        combined_fft_shifted = input_fft_shifted * mask + pred_fft_shifted * (1 - mask)
        
        # 5. 逆变换回空域
        combined_fft = torch.fft.ifftshift(combined_fft_shifted, dim=(-2, -1))
        output = torch.fft.ifft2(combined_fft).real
        
        return output

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

# ==========================================
# 🔥 主网络结构 DSTCarbonFormer
# ==========================================
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
        
        # Stage 2: SFT + MoE Block
        self.sft2 = SFTLayer3D(dim)
        self.moe_block = MoEBlock(dim, num_experts=3, top_k=1)
        
        # 4. 全局上下文 (Mamba)
        self.global_context = EfficientContextBlock(dim)
        
        # Mamba 初始化
        if Mamba is not None:
            self.mamba_block = Mamba(
                d_model=dim, 
                d_state=16, 
                d_conv=4,    
                expand=2     
            )
        else:
            self.mamba_block = nn.Identity()
        
        # 5. 重建层
        self.tail = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv3d(dim, 1, 3, 1, 1)
        )
        
        # 6. 频率硬约束
        # ✅ 修改3: 将 radius 设为 16，适配 160x160 的尺寸
        self.constraint = FrequencyHardConstraint(radius=16)

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
        f_main = self.moe_block(f_main)
        
        # Global Context
        f_global = self.global_context(f_main)
        
        # ✅ 数据形状适配 Mamba
        B, C, T, H, W = f_global.shape
        
        # (B, C, T, H, W) -> (B, L, C)
        x_mamba = f_global.flatten(2).transpose(1, 2)
        
        # Mamba Forward
        x_mamba = self.mamba_block(x_mamba)
        
        # 还原: (B, L, C) -> (B, C, T, H, W)
        f_mamba = x_mamba.transpose(1, 2).view(B, C, T, H, W)
        
        f_final = f_main + f_mamba
        
        # Reconstruction
        residual = self.tail(f_final)
        
        # 初始预测
        pred_raw = F.relu(main + residual)
        
        # ✅ 最后一步: 物理硬约束
        # 强制把 pred_raw 的低频部分替换为 main (马赛克输入) 的低频部分
        final_output = self.constraint(pred_raw, main)
        
        return final_output