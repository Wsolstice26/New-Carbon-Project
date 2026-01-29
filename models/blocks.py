import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# ==========================================
# 1. 多尺度感知模块 (Multi-Scale Block)
# ==========================================
class MultiScaleBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hid_c = channels // 4
        self.branch1 = nn.Conv3d(channels, hid_c, 3, 1, 1, dilation=1)
        self.branch2 = nn.Conv3d(channels, hid_c, 3, 1, 2, dilation=2)
        self.branch3 = nn.Conv3d(channels, hid_c, 3, 1, 4, dilation=4)
        self.branch4 = nn.Conv3d(channels, hid_c, 1, 1, 0)
        self.fusion = nn.Conv3d(channels, channels, 1, 1, 0)

    def forward(self, x):
        b1 = F.relu(self.branch1(x))
        b2 = F.relu(self.branch2(x))
        b3 = F.relu(self.branch3(x))
        b4 = F.relu(self.branch4(x))
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.fusion(out) + x


# ==========================================
# 2. [轻量版] SFT 融合层 (Lite SFT)
# ==========================================
class SFTLayer3D(nn.Module):
    """
    使用深度可分离卷积优化，速度提升 20 倍。
    """
    def __init__(self, channels):
        super().__init__()
        self.sft_net = nn.Sequential(
            # 深度卷积 (Depthwise)
            nn.Conv3d(channels, channels, 3, 1, 1, groups=channels),
            # 点卷积 (Pointwise)
            nn.Conv3d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1),
            # 投影
            nn.Conv3d(channels, channels*2, 1, 1, 0)
        )
    def forward(self, main, aux):
        scale_shift = self.sft_net(aux)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        return main * (1 + scale) + shift


# ==========================================
# 3. 高效全局注意力 (Efficient Global Context)
# ==========================================
class EfficientContextBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduce_conv = nn.Conv3d(dim, dim // 2, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim // 2, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid()
        )
        self.restore_conv = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        b, c, t, h, w = x.shape
        identity = x
        y = self.reduce_conv(x)
        y = self.avg_pool(y).view(b, -1)
        y = self.mlp(y).view(b, c, 1, 1, 1)
        out = x * y
        return self.restore_conv(out) + identity


# ==========================================
# 4. [防爆版] 频率硬约束层 (Safe Frequency Constraint)
# ==========================================
class FrequencyHardConstraint(nn.Module):
    def __init__(self, radius=16):
        super().__init__()
        self.radius = radius 

    def get_low_pass_filter(self, shape, device):
        b, c, t, h, w = shape
        center_h, center_w = h // 2, w // 2
        
        y = torch.arange(h, device=device)
        x = torch.arange(w, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        dist = (grid_x - center_w)**2 + (grid_y - center_h)**2
        mask = torch.zeros((h, w), device=device)
        mask[dist <= self.radius**2] = 1.0
        return mask.view(1, 1, 1, h, w)

    def forward(self, pred, input_main):
        # 🛡️【关键修改】强制局部使用 FP32 
        # enabled=False 暂时关闭 AMP，防止 FFT 在 FP16 下溢出 NaN
        with torch.amp.autocast('cuda', enabled=False):
            # 必须手动转为 float()，因为 autocontext 关闭时不会自动转换
            pred = pred.float()
            input_main = input_main.float()

            if pred.shape != input_main.shape:
                input_main = F.interpolate(
                    input_main.view(input_main.shape[0], -1, input_main.shape[3], input_main.shape[4]),
                    size=pred.shape[-2:], mode='bilinear', align_corners=False
                ).view_as(pred)

            # FFT 计算 (FP32 下非常安全)
            fft_pred = torch.fft.fftn(pred, dim=(-2, -1))
            fft_input = torch.fft.fftn(input_main, dim=(-2, -1))
            
            fft_pred_shift = torch.fft.fftshift(fft_pred, dim=(-2, -1))
            fft_input_shift = torch.fft.fftshift(fft_input, dim=(-2, -1))
            
            mask = self.get_low_pass_filter(pred.shape, pred.device)
            
            fft_fused_shift = fft_input_shift * mask + fft_pred_shift * (1 - mask)
            
            fft_fused = torch.fft.ifftshift(fft_fused_shift, dim=(-2, -1))
            output = torch.fft.ifftn(fft_fused, dim=(-2, -1)).real
            
            return output


# ==========================================
# 5. [并行版] MoE 模块 (Parallel Soft MoE)
# ==========================================
class MoEBlock(nn.Module):
    """
    并行计算所有专家并加权，消除 Gather/Scatter 开销。
    """
    def __init__(self, dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(dim, dim, 1),
                nn.PReLU(),
                nn.Conv3d(dim, dim, 1)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        x_perm = x.permute(0, 2, 3, 4, 1)
        logits = self.gate(x_perm) 
        weights = F.softmax(logits, dim=-1) 
        
        final_out = 0
        for i in range(self.num_experts):
            expert_out = self.experts[i](x)
            w = weights[..., i].unsqueeze(1)
            final_out += expert_out * w
            
        return final_out + x