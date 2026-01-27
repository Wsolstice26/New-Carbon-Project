import torch
import torch.nn as nn
import torch.nn.functional as F

# --- A. 多尺度感知模块 (Multi-Scale Block) ---
# 保持不变
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

# --- B. SFT 融合层 ---
# 保持不变
class SFTLayer3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.sft_net = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(channels, channels*2, 3, 1, 1)
        )
    def forward(self, main, aux):
        scale_shift = self.sft_net(aux)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        return main * (1 + scale) + shift

# --- C. [修改版] 高效全局注意力 (Efficient Global Context) ---
# 替换掉了原来那个炸显存的 Transformer
# 作用：先把图变小(Pooling)再算注意力，算完再插值回去
class EfficientContextBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 1. 降维，减少计算量
        self.reduce_conv = nn.Conv3d(dim, dim // 2, 1)
        
        # 2. 全局池化 (把 128x128 变成 1x1 的点，获取全局信息)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # 3. 激励网络 (类似 SE-Block)
        self.mlp = nn.Sequential(
            nn.Linear(dim // 2, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid()
        )
        
        # 4. 恢复
        self.restore_conv = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        # x: [B, C, T, H, W]
        b, c, t, h, w = x.shape
        
        # 残差连接输入
        identity = x
        
        # 1. 降低通道
        y = self.reduce_conv(x) # [B, C/2, T, H, W]
        
        # 2. 全局平均池化 -> 变成一个向量
        y = self.avg_pool(y).view(b, -1) # [B, C/2]
        
        # 3. 计算全局权重
        y = self.mlp(y).view(b, c, 1, 1, 1) # [B, C, 1, 1, 1]
        
        # 4. 把权重乘回原图 (Excite)
        out = x * y
        
        return self.restore_conv(out) + identity

# ==========================================
# ➕ 新增：SEN2SR 核心 - 频率硬约束层
# ==========================================
import torch.fft

class FrequencyHardConstraint(nn.Module):
    """
    实现 SEN2SR 的核心逻辑：
    在频域中，强行把【输入的低频信息】和【模型的预测高频信息】拼接。
    保证：宏观数值（低频）绝不失真，只生成纹理（高频）。
    """
    def __init__(self, radius=16):
        super().__init__()
        self.radius = radius # 控制保留多少低频信息（半径越小，保留的低频越少）

    def get_low_pass_filter(self, shape, device):
        # 创建一个圆形的低通滤波器掩膜 (Mask)
        # 1 = 保留低频 (用输入的), 0 = 保留高频 (用预测的)
        b, c, t, h, w = shape
        center_h, center_w = h // 2, w // 2
        
        # 生成网格坐标
        y = torch.arange(h, device=device)
        x = torch.arange(w, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        # 计算距离中心的距离
        dist = (grid_x - center_w)**2 + (grid_y - center_h)**2
        
        # 生成 Mask
        mask = torch.zeros((h, w), device=device)
        mask[dist <= self.radius**2] = 1.0
        
        # 调整维度以匹配 [B, C, T, H, W]
        return mask.view(1, 1, 1, h, w)

    def forward(self, pred, input_main):
        """
        pred: 模型预测的高分辨率图像 (包含可能的错误低频)
        input_main: 原始的粗糙输入 (低频是绝对准确的)
        """
        # 1. 确保输入尺寸一致 (通常 input_main 已经是 128x128 的插值结果)
        if pred.shape != input_main.shape:
            input_main = F.interpolate(
                input_main.view(input_main.shape[0], -1, input_main.shape[3], input_main.shape[4]),
                size=pred.shape[-2:], mode='bilinear', align_corners=False
            ).view_as(pred)

        # 2. 转到频域 (FFT)
        # 只在空间维度 (H, W) 上做 FFT
        fft_pred = torch.fft.fftn(pred, dim=(-2, -1))
        fft_input = torch.fft.fftn(input_main, dim=(-2, -1))
        
        # 移频 (把低频移到图像中心)
        fft_pred_shift = torch.fft.fftshift(fft_pred, dim=(-2, -1))
        fft_input_shift = torch.fft.fftshift(fft_input, dim=(-2, -1))
        
        # 3. 获取滤波器掩膜
        mask = self.get_low_pass_filter(pred.shape, pred.device)
        
        # 4. 核心操作：融合
        # Mask区域(低频): 使用 input 的真实信息
        # 非Mask区域(高频): 使用 pred 的生成信息
        fft_fused_shift = fft_input_shift * mask + fft_pred_shift * (1 - mask)
        
        # 5. 逆变换回空域 (IFFT)
        fft_fused = torch.fft.ifftshift(fft_fused_shift, dim=(-2, -1))
        output = torch.fft.ifftn(fft_fused, dim=(-2, -1)).real
        
        return output
    

    # [追加到 models/blocks.py]

# ==========================================
# 🆕 新增模块 1: MoE (Mixture of Experts)
# ==========================================
class MoEBlock(nn.Module):
    """
    稀疏混合专家模块 (Sparse MoE)
    用途：替换普通卷积层，增加模型容量但保持低计算量。
    """
    def __init__(self, dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 门控网络 (Gating Network): 决定用哪个专家
        self.gate = nn.Linear(dim, num_experts)
        
        # 专家网络 (Experts): 这里用简单的 1x1 卷积代替 MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(dim, dim, 1),
                nn.PReLU(),
                nn.Conv3d(dim, dim, 1)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        # x: [B, C, T, H, W]
        b, c, t, h, w = x.shape
        
        # 1. 计算路由权重
        # 先把空间时间维度展平做 attention
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, c) # [N, C]
        logits = self.gate(x_flat) # [N, num_experts]
        
        # 2. 选出 Top-K 专家
        probs, indices = torch.topk(logits, self.top_k, dim=1)
        probs = F.softmax(probs, dim=1)
        
        # 3. 动态聚合
        # 为了代码简单且兼容性好，这里用循环实现（虽然慢一点点，但稳）
        out = torch.zeros_like(x_flat)
        
        for k in range(self.top_k):
            expert_idx = indices[:, k] # 当前第k个选择的专家索引
            prob = probs[:, k].unsqueeze(1) # 权重
            
            # 这里的 mask 稍微有点耗时，工程化通常会用 scatter
            # 但针对小 Batch 训练，直接遍历专家更直观
            for i in range(self.num_experts):
                # 找到所有选择了专家 i 的样本位置
                mask = (expert_idx == i)
                if mask.any():
                    # 提取这些样本
                    selected_x = x_flat[mask]
                    # 还原成 3D 形状送入卷积 [Batch_sub, C, 1, 1, 1] 模拟
                    # 注意：为了让 3D 卷积能处理，我们需要 reshape 回去
                    # 这里简化处理：直接用 Linear 模拟 1x1 卷积效果
                    # 真正的 MoE 卷积需要更复杂的 scatter/gather
                    # 下面是逻辑等效的简化版：
                    
                    # 重新构造 expert 的 forward
                    # 由于我们上面定义的是 Conv3d，这里为了对齐形状：
                    inp_sub = selected_x.view(-1, c, 1, 1, 1)
                    out_sub = self.experts[i](inp_sub).view(-1, c)
                    
                    # 累加结果
                    out[mask] += out_sub * prob[mask]
                    
        return out.view(b, t, h, w, c).permute(0, 4, 1, 2, 3) + x # 残差连接

# ==========================================
# 🆕 新增模块 2: Mamba-like Block (纯 PyTorch 版)
# ==========================================
class SimpleMambaBlock(nn.Module):
    """
    简化版 Mamba 块 (无需安装 mamba-ssm 库，适配 AMD)
    使用 门控卷积 + 深度可分离卷积 模拟 SSM 的选择性机制
    """
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.dim = dim
        
        # 1. 输入投影
        self.in_proj = nn.Conv3d(dim, dim * 2, 1)
        
        # 2. 深度卷积 (模拟 SSM 的长期记忆)
        self.conv = nn.Conv3d(dim, dim, 3, 1, 1, groups=dim)
        
        # 3. 状态投影 (模拟 SSM 的参数离散化)
        self.x_proj = nn.Linear(dim, d_state + dim * 2) 
        
        # 4. 门控机制
        self.act = nn.SiLU()
        
        # 5. 输出投影
        self.out_proj = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        # x: [B, C, T, H, W]
        residual = x
        b, c, t, h, w = x.shape
        
        # 1. 投影分为两支: x (信号) 和 z (门)
        xz = self.in_proj(x)
        x_signal, z_gate = torch.chunk(xz, 2, dim=1)
        
        # 2. 处理信号支 (Conv -> SiLU)
        x_signal = self.conv(x_signal)
        x_signal = self.act(x_signal)
        
        # 3. 简化版 SSM 操作 (用 门控注意力 模拟)
        # 真正的 Mamba 这里是 scan 操作，这里我们用 Global Avg 模拟状态选择
        x_flat = x_signal.mean(dim=[2,3,4]) # [B, C] 全局描述符
        params = self.x_proj(x_flat) # [B, d_state + 2*dim]
        
        # 动态调整特征 (Selective Scan 的平替)
        dt, B_state, C_state = torch.split(params, [c, c, 16], dim=1)
        dt = torch.sigmoid(dt).view(b, c, 1, 1, 1)
        
        y = x_signal * dt # 这种“软门控”模拟了选择性遗忘
        
        # 4. 门控融合
        y = y * self.act(z_gate)
        
        # 5. 输出
        return self.out_proj(y) + residual