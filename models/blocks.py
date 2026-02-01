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
        
        # 🔥 定义深度可分离 3D 卷积 (Depthwise Separable Conv)
        # 作用：将计算量和显存占用降低 5-8 倍，绕过 AMD MIOpen 的性能黑洞
        def dw_conv3d(in_c, out_c, k, s, p, d):
            return nn.Sequential(
                # 1. Depthwise: 独立处理每个通道的空间信息 (groups=in_c)
                # 这步极快，且避开了标准 Conv3d 的优化缺陷
                nn.Conv3d(in_c, in_c, k, s, p, dilation=d, groups=in_c),
                # 2. Pointwise: 1x1 卷积融合通道信息 (本质是矩阵乘法，AMD 擅长)
                nn.Conv3d(in_c, out_c, 1, 1, 0)
            )

        # 使用优化后的 dw_conv3d 替换标准 nn.Conv3d
        self.branch1 = dw_conv3d(channels, hid_c, 3, 1, 1, 1)
        self.branch2 = dw_conv3d(channels, hid_c, 3, 1, 2, 2)
        self.branch3 = dw_conv3d(channels, hid_c, 3, 1, 4, 4)
        
        # Branch4 本身就是 1x1，不需要改
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
# 5. MoE 模块 
# ==========================================
class MoEBlock(nn.Module):
    """
    [优化版] 并行 MoE 模块:
    1. 向量化执行: 消除 Python 循环，使用分组卷积并行计算所有专家。
    2. Top-K 掩码: 真正生效 top_k 参数，强制稀疏路由学习。
    """
    def __init__(self, dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim
        
        # 门控网络 (Gating Network)
        self.gate = nn.Linear(dim, num_experts)
        
        # 专家网络 (Experts) - 向量化实现
        # -----------------------------------------------------------
        # 逻辑等价于: num_experts 个 [Conv(1x1) -> Act -> Conv(1x1)]
        # -----------------------------------------------------------
        
        # 第一层: 将输入投影到所有专家的中间空间
        # 输入: dim -> 输出: dim * num_experts
        self.experts_layer1 = nn.Conv3d(dim, dim * num_experts, kernel_size=1)
        
        # 激活函数: 推荐 SiLU 或 GELU，速度快且无参数依赖
        self.act = nn.SiLU() 
        
        # 第二层: 分组卷积 (Grouped Conv)
        # 这里的 groups=num_experts 极其关键，它确保了通道之间不串扰，
        # 相当于 N 个独立的卷积在并行运行。
        self.experts_layer2 = nn.Conv3d(
            dim * num_experts, 
            dim * num_experts, 
            kernel_size=1, 
            groups=num_experts # 每个组对应一个专家
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        
        # ===========================
        # 1. 计算路由权重 (Gating)
        # ===========================
        x_perm = x.permute(0, 2, 3, 4, 1) # [B, T, H, W, C]
        logits = self.gate(x_perm)        # [B, T, H, W, N]
        
        # --- Top-K 逻辑 ---
        if self.top_k < self.num_experts:
            # 找到 top_k 的值和索引 (保持梯度)
            topk_vals, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)
            
            # 创建掩码：初始化为负无穷
            mask = torch.full_like(logits, float('-inf'))
            
            # 将 top_k 位置填回原始数值
            # scatter_ 也就是把 topk_vals 放回 mask 的对应 topk_indices 位置
            mask.scatter_(-1, topk_indices, topk_vals)
            
            # 使用 mask 后的 logits (非 top_k 变为 -inf，Softmax 后为 0)
            logits = mask

        # 计算最终权重
        weights = F.softmax(logits, dim=-1) # [B, T, H, W, N]
        
        # ===========================
        # 2. 并行计算所有专家 (Vectorized Experts)
        # ===========================
        # Layer 1: [B, C, ...] -> [B, N*C, ...]
        expert_out = self.experts_layer1(x)
        expert_out = self.act(expert_out)
        
        # Layer 2 (Grouped): [B, N*C, ...] -> [B, N*C, ...]
        expert_out = self.experts_layer2(expert_out)
        
        # ===========================
        # 3. 加权融合 (Weighted Sum)
        # ===========================
        # 重塑形状: [B, N*C, T, H, W] -> [B, N, C, T, H, W]
        expert_out = expert_out.view(B, self.num_experts, C, T, H, W)
        
        # 调整权重形状以进行广播乘法
        # weights: [B, T, H, W, N] -> [B, N, 1, T, H, W]
        weights = weights.permute(0, 4, 1, 2, 3).unsqueeze(2)
        
        # 加权求和: Sum(Expert_i * Weight_i)
        # 这一步会自动把权重为 0 (非 Top-K) 的专家输出过滤掉
        final_out = torch.sum(expert_out * weights, dim=1)
        
        return final_out + x