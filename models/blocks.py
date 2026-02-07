import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. [新增] 基础工具：DropPath
#    用于深层网络的正则化，防止过拟合和梯度消失
# ==========================================
class DropPath(nn.Module):
    """
    Stochastic Depth (DropPath) for residual connections.
    Ref: https://github.com/rwightman/pytorch-image-models
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # handle broadcasting for different tensor shapes
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

# ==========================================
# 1. [新增] 轻量级时序卷积 (Temporal DWConv)
#    专门针对 T=3 设计，强制模型在空间操作前看前后帧
# ==========================================
class TemporalDWConv3d(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.t_conv = nn.Conv3d(
            dim, dim, 
            kernel_size=(kernel_size, 1, 1), # 只在 T 维度卷积 (3, 1, 1)
            stride=1, 
            padding=(kernel_size//2, 0, 0),  # 保持时间维度 T 不变
            groups=dim,                      # Depthwise 分组卷积，省参数
            bias=False
        )
    
    def forward(self, x):
        return self.t_conv(x)

# ==========================================
# 2. [新增] 门控融合层 (Gated Fusion)
#    替代 Cross-Attention，用于 Aux 和 Main 的高效融合
# ==========================================
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_conv = nn.Conv3d(dim * 2, 1, kernel_size=1, bias=True)
        self.aux_proj = nn.Conv3d(dim, dim, kernel_size=1, bias=False)
        
    def forward(self, x, aux):
        # x: Main Feature [B, C, T, H, W]
        # aux: Aux Feature [B, C, T, H, W]
        
        # 1. 计算门控系数 (0~1)
        cat_feat = torch.cat([x, aux], dim=1)
        gate = torch.sigmoid(self.gate_conv(cat_feat))
        
        # 2. 投影 Aux 特征
        aux_out = self.aux_proj(aux)
        
        # 3. 加权融合 (Residual)
        return x + gate * aux_out

# ==========================================
# 3. 多尺度感知模块 (MultiScale Block)
#    用于 Aux Head 提取路网等多尺度特征
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
# 4. SFT 融合层 (Lite SFT)
#    用于空间特征调制 (仿射变换)
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
# 5. 高效全局注意力 (Efficient Context)
#    轻量级 SE-Block，用于通道重校准
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
        # 全局平均池化 -> MLP -> 通道权重
        y = self.reduce_conv(x)
        y = self.avg_pool(y).view(b, -1)
        y = self.mlp(y).view(b, c, 1, 1, 1)
        out = x * y
        return self.restore_conv(out) + identity

# ==========================================
# 6. MoE 模块 (Mixture of Experts)
#    用于增强模型的非线性表达能力
# ==========================================
class MoEBlock(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim

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

        logits = self.gate(x)

        if K < E:
            topk_vals, topk_idx = torch.topk(logits, k=K, dim=1)
            topk_w = F.softmax(topk_vals, dim=1).to(dtype=x.dtype)
            weights = torch.zeros_like(logits, dtype=x.dtype)
            weights.scatter_(1, topk_idx, topk_w)
        else:
            weights = F.softmax(logits, dim=1).to(dtype=x.dtype)

        expert_out = self.experts_layer2(self.act(self.experts_layer1(x)))
        expert_out = expert_out.view(B, E, C, T, H, W)

        weights = weights.unsqueeze(2)
        out = (expert_out * weights).sum(dim=1)
        return out + x

from mamba_ssm import Mamba    
# ==========================================
# 8. [新增] 双向 Mamba 适配器 (Bi-Mamba)
#    对于图像/视频重建任务，必须同时感知上下文
# ==========================================
class BiMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # 正向 Mamba
        self.fwd_mamba = Mamba(
            d_model=dim, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        # 反向 Mamba
        self.bwd_mamba = Mamba(
            d_model=dim, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
    
    def forward(self, x):
        # x: [B, L, C]
        
        # 1. 正向扫描
        x_fwd = self.fwd_mamba(x)
        
        # 2. 反向扫描 (先翻转序列，处理完再翻转回来)
        # dim=1 是序列长度维度 L
        x_bwd = self.bwd_mamba(x.flip(dims=[1])).flip(dims=[1])
        
        # 3. 结果融合 (相加)
        return x_fwd + x_bwd