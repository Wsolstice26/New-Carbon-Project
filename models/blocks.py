import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# ==========================================
# 1. 多尺度感知模块 (Multi-Scale Block)
# 🚀 深度可分离卷积优化版 (AMD ROCm Friendly)
# ==========================================
class MultiScaleBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hid_c = channels // 4
        
        def dw_conv3d(in_c, out_c, k, s, p, d):
            return nn.Sequential(
                nn.Conv3d(in_c, in_c, k, s, p, dilation=d, groups=in_c),
                nn.Conv3d(in_c, out_c, 1, 1, 0)
            )

        self.branch1 = dw_conv3d(channels, hid_c, 3, 1, 1, 1)
        self.branch2 = dw_conv3d(channels, hid_c, 3, 1, 2, 2)
        self.branch3 = dw_conv3d(channels, hid_c, 3, 1, 4, 4)
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
    def __init__(self, channels):
        super().__init__()
        self.sft_net = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1, groups=channels),
            nn.Conv3d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1),
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
# 4. [备份] 频率硬约束层 (Frequency Constraint)
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
        with torch.amp.autocast('cuda', enabled=False):
            pred = pred.float()
            input_main = input_main.float()
            if pred.shape != input_main.shape:
                input_main = F.interpolate(
                    input_main.view(input_main.shape[0], -1, input_main.shape[3], input_main.shape[4]),
                    size=pred.shape[-2:], mode='bilinear', align_corners=False
                ).view_as(pred)
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
# 5. MoE 模块 
# ==========================================
class MoEBlock(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim

        # gate: Linear(C->E) 等价于 1x1x1 Conv(C->E)
        self.gate = nn.Conv3d(dim, num_experts, kernel_size=1, bias=True)

        self.experts_layer1 = nn.Conv3d(dim, dim * num_experts, kernel_size=1)
        self.act = nn.SiLU()
        self.experts_layer2 = nn.Conv3d(
            dim * num_experts,
            dim * num_experts,
            kernel_size=1,
            groups=num_experts
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        E = self.num_experts
        K = self.top_k

        # logits: [B, E, T, H, W]  （无 permute）
        logits = self.gate(x)

        if K < E:
            # topk over expert-dim
            topk_vals, topk_idx = torch.topk(logits, k=K, dim=1)  # [B,K,T,H,W]

            # 只在 topk 上做 softmax（等价于 masked -inf softmax）
            topk_w = F.softmax(topk_vals, dim=1).to(dtype=x.dtype)  # [B,K,T,H,W]

            # scatter 回完整 E 维 weights: [B,E,T,H,W]
            weights = torch.zeros_like(logits, dtype=x.dtype)       # 直接用 x.dtype，少一次 cast
            weights.scatter_(1, topk_idx, topk_w)
        else:
            weights = F.softmax(logits, dim=1).to(dtype=x.dtype)

        # experts 输出: [B, E*C, T,H,W] -> [B,E,C,T,H,W]
        expert_out = self.experts_layer2(self.act(self.experts_layer1(x)))
        expert_out = expert_out.view(B, E, C, T, H, W)

        # weights: [B,E,T,H,W] -> [B,E,1,T,H,W]
        weights = weights.unsqueeze(2)

        out = (expert_out * weights).sum(dim=1)  # [B,C,T,H,W]
        return out + x




# ==========================================
# 6. [修正] 物理约束层 (Water-Filling) - 支持动态 Scale
# ==========================================
class PhysicsConstraintLayer(nn.Module):
    """
    基于 Water-Filling (单纯形投影) 的物理约束层。
    保证：
    1) 非负性 (Non-negative)
    2) 均值/总量守恒 (Mean/Sum Consistency)  —— 在指定 block scale 上
    3) 数值稳定 (No NaN)

    关键修正：
    - 支持 forward 传入 scale，实现训练(全局/4km) 与推理(1km) 不同约束尺度。
    """
    def __init__(self, scale_factor=10, norm_const=11.0):
        super().__init__()
        self.default_scale = int(scale_factor)
        self.norm_const = float(norm_const)

    def water_filling_projection(self, pred_linear, input_down, scale: int):
        """
        pred_linear: [B, C, T, H, W] 线性域，非负
        input_down:  [B, C, T, H/s, W/s] 线性域的 block 均值
        scale:       block 尺度 s
        """
        B, C, T, H, W = pred_linear.shape
        s = int(scale)
        assert H % s == 0 and W % s == 0, f"H,W must be divisible by scale={s}, got {(H, W)}"
        n = s * s  # block 内像素数

        # 1) 目标总和 S：均值 * 像素数
        S = input_down * n  # [B,C,T,h_grid,w_grid]

        # 2) reshape 到 block 向量
        h_grid, w_grid = H // s, W // s
        p_blocks = (
            pred_linear.view(B, C, T, h_grid, s, w_grid, s)
                      .permute(0, 1, 2, 3, 5, 4, 6)
                      .reshape(B, C, T, h_grid, w_grid, n)
        )

        p_flat = p_blocks.reshape(-1, n)   # [Nblock, n]
        S_flat = S.reshape(-1, 1)          # [Nblock, 1]

        # 3) sort-based simplex projection (Duchi-style)
        u, _ = torch.sort(p_flat, dim=-1, descending=True)  # [Nblock, n]
        cssv = torch.cumsum(u, dim=-1)                      # [Nblock, n]
        k = torch.arange(1, n + 1, device=pred_linear.device).view(1, n)  # [1, n]
        t = (cssv - S_flat) / k                              # [Nblock, n]

        mask = u > t
        rho = mask.sum(dim=-1, keepdim=True).clamp(min=1)   # [Nblock, 1]
        theta = t.gather(dim=-1, index=rho - 1)             # [Nblock, 1]

        q_flat = torch.clamp(p_flat - theta, min=0.0)       # [Nblock, n]

        # 4) reshape 回原图
        q_blocks = q_flat.view(B, C, T, h_grid, w_grid, n)
        q_out = (
            q_blocks.view(B, C, T, h_grid, w_grid, s, s)
                   .permute(0, 1, 2, 3, 5, 4, 6)
                   .reshape(B, C, T, H, W)
        )
        return q_out

    def forward(self, pred_log_norm, input_mosaic_log_norm, scale=None):
        """
        pred_log_norm:         [B, C, T, H, W]   (log1p(x)/norm_const)
        input_mosaic_log_norm: [B, C, T, H, W]   同形状，作为守恒约束来源
        scale: int or None     block 尺度；None 则用 default_scale
        """
        s = int(scale) if scale is not None else self.default_scale

        # 1) 还原线性空间（非负）
        pred_linear = torch.expm1(pred_log_norm * self.norm_const).clamp(min=0)
        input_linear = torch.expm1(input_mosaic_log_norm * self.norm_const).clamp(min=0)

        # 2) 计算 block 均值（在 scale=s 的尺度上）
        input_down = F.avg_pool3d(
            input_linear,
            kernel_size=(1, s, s),
            stride=(1, s, s)
        )  # [B,C,T,H/s,W/s]

        # 3) Water-Filling 投影（在同一个尺度上守恒）
        corrected_linear = self.water_filling_projection(pred_linear, input_down, s)

        # 4) 回到 log 归一化空间
        corrected_log_norm = torch.log1p(corrected_linear) / self.norm_const
        return corrected_log_norm
