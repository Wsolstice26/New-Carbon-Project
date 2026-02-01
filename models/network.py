import torch
import torch.nn as nn
import torch.nn.functional as F

# 从 blocks 导入已优化的模块
from .blocks import MultiScaleBlock3D, SFTLayer3D, EfficientContextBlock, MoEBlock

# 鲁棒性导入 Mamba
try:
    from mamba_ssm import Mamba
except ImportError:
    print("⚠️ [Network] 未找到 mamba_ssm，将使用 Identity 代替 (仅供调试)")
    Mamba = None

# ==========================================
# 🛠️ 辅助类: 深度可分离卷积 (性能救星)
# ==========================================
class DepthwiseSeparableConv3d(nn.Module):
    """
    将标准 Conv3d 拆分为 Depthwise + Pointwise，
    解决 AMD ROCm 上标准 3D 卷积反向传播极慢 (1.7s -> 0.1s) 的问题。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size, stride, padding, 
            groups=in_channels # 关键：分组数=通道数
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# ==========================================
# 🛡️ 物理硬约束层 (AMP 安全版)
# ==========================================
class FrequencyHardConstraint(nn.Module):
    def __init__(self, radius=16):
        super().__init__()
        self.radius = radius

    def forward(self, pred, low_res_input):
        # 🔥 关键修复: 关闭 AMP，强制 FP32
        # FFT 在 FP16 下极易溢出导致 NaN，必须保护
        with torch.amp.autocast('cuda', enabled=False):
            pred = pred.float()
            low_res_input = low_res_input.float()
            
            # 1. FFT 变换
            pred_fft = torch.fft.fft2(pred)
            input_fft = torch.fft.fft2(low_res_input)
            
            # 2. 创建 Mask (Lazy Creation to save memory)
            B, C, T, H, W = pred.shape
            cy, cx = H // 2, W // 2
            y = torch.arange(H, device=pred.device)
            x = torch.arange(W, device=pred.device)
            y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
            dist = torch.sqrt((y_grid - cy)**2 + (x_grid - cx)**2)
            mask = (dist <= self.radius).float().view(1, 1, 1, H, W)
            
            # 3. 频谱搬移与替换
            pred_fft_shifted = torch.fft.fftshift(pred_fft, dim=(-2, -1))
            input_fft_shifted = torch.fft.fftshift(input_fft, dim=(-2, -1))
            
            # 这里的逻辑是：低频取 input (物理守恒)，高频取 pred (细节生成)
            combined_fft_shifted = input_fft_shifted * mask + pred_fft_shifted * (1 - mask)
            
            # 4. 逆变换
            combined_fft = torch.fft.ifftshift(combined_fft_shifted, dim=(-2, -1))
            output = torch.fft.ifft2(combined_fft).real
            
            return output

# ==========================================
# 🧱 基础卷积块 (已替换为高效卷积)
# ==========================================
class ConvBlock3D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # 替换原有的 nn.Conv3d 为 DepthwiseSeparableConv3d
        self.conv = nn.Sequential(
            DepthwiseSeparableConv3d(in_c, out_c, 3, 1, 1),
            nn.PReLU(),
            DepthwiseSeparableConv3d(out_c, out_c, 3, 1, 1)
        )
    def forward(self, x): return self.conv(x)

# ==========================================
# 🔥 主网络结构 DSTCarbonFormer
# ==========================================
class DSTCarbonFormer(nn.Module):
    def __init__(self, aux_c=9, main_c=1, dim=64):
        super().__init__()
        
        # 1. 辅助流编码器 (Head 也换成高效卷积)
        self.aux_head = DepthwiseSeparableConv3d(aux_c, dim, 3, 1, 1)
        self.aux_multiscale = MultiScaleBlock3D(dim) 
        
        # 2. 主流编码器
        self.main_head = DepthwiseSeparableConv3d(main_c, dim, 3, 1, 1)
        
        # 3. 双流融合 (SFT Fusion)
        # Stage 1: 标准 SFT + ResBlock
        self.sft1 = SFTLayer3D(dim)
        self.res1 = ConvBlock3D(dim, dim)
        
        # Stage 2: SFT + MoE Block
        self.sft2 = SFTLayer3D(dim)
        self.moe_block = MoEBlock(dim, num_experts=3, top_k=1)
        
        # 4. 全局上下文 (Mamba)
        self.global_context = EfficientContextBlock(dim)
        
        # 🔥 Mamba 优化: 降采样比例 (Lightweight Strategy)
        self.down_scale = 4 
        # 降采样层 (120 -> 30)
        self.mamba_down = nn.AvgPool3d((1, self.down_scale, self.down_scale))
        
        if Mamba is not None:
            self.mamba_block = Mamba(
                d_model=dim, 
                d_state=16, 
                d_conv=4,    
                expand=2     
            )
        else:
            self.mamba_block = nn.Identity()
        
        # 5. 重建层 (Tail)
        self.tail = nn.Sequential(
            DepthwiseSeparableConv3d(dim, dim, 3, 1, 1),
            nn.PReLU(),
            DepthwiseSeparableConv3d(dim, 1, 3, 1, 1)
        )
        
        # 6. 频率硬约束
        self.constraint = FrequencyHardConstraint(radius=16)

    # 将 Mamba 逻辑剥离并禁止编译，防止 Dynamo 报错
    @torch.compiler.disable
    def _forward_mamba_safe(self, x):
        """
        x: [B, C, T, H_small, W_small]
        """
        B, C, T, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2) # (B, L, C)
        x_out = self.mamba_block(x_flat)
        return x_out.transpose(1, 2).view(B, C, T, H, W)

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
        
        # Global Context (Channel Attention)
        f_global = self.global_context(f_main)
        
        # ===========================
        # 🔥 Mamba 轻量化路径
        # ===========================
        # 1. 降采样 (B, C, T, 120, 120) -> (B, C, T, 30, 30)
        # 这一步让计算量减少 16 倍！
        f_small = self.mamba_down(f_global)
        
        # 2. 运行 Mamba (Eager Mode, Safe)
        f_mamba_small = self._forward_mamba_safe(f_small)
        
        # 3. 上采样回原尺寸 (使用三线性插值)
        f_mamba = F.interpolate(
            f_mamba_small, 
            size=f_global.shape[2:], # (T, H, W)
            mode='trilinear', 
            align_corners=False
        )
        
        # 残差连接
        f_final = f_main + f_mamba
        
        # Reconstruction
        residual = self.tail(f_final)
        
        # 初始预测
        pred_raw = F.relu(main + residual)
        
        # 物理硬约束
        final_output = self.constraint(pred_raw, main)
        
        return final_output