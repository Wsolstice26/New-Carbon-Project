import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

# ==========================================
# 1. 基础组件 (适配 3D)
# ==========================================

# --- SSIM 辅助函数 (保持不变) ---
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window_3d(window_size, channel):
    # 创建 3D 高斯窗口
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # 扩展到 3D: [C, 1, 1, H, W] -> 用于对每一帧做 SSIM
    window = Variable(_2D_window.expand(channel, 1, 1, window_size, window_size).contiguous())
    return window

def _ssim_3d(img1, img2, window, window_size, channel, size_average=True):
    # img: [B, C, T, H, W]
    # 我们把 T 维度当作 Batch 的一部分来处理，或者逐帧计算
    # 这里为了简便，将 [B, C, T, H, W] -> [B*T, C, H, W] 计算 2D SSIM
    b, c, t, h, w = img1.shape
    img1_2d = img1.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
    img2_2d = img2.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
    
    # 动态创建 2D Window
    real_window = create_window(window_size, c).to(img1.device).type_as(img1)
    
    mu1 = F.conv2d(img1_2d, real_window, padding=window_size//2, groups=c)
    mu2 = F.conv2d(img2_2d, real_window, padding=window_size//2, groups=c)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1_2d*img1_2d, real_window, padding=window_size//2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2_2d*img2_2d, real_window, padding=window_size//2, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1_2d*img2_2d, real_window, padding=window_size//2, groups=c) - mu1_mu2
    
    C1 = 0.01**2; C2 = 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    if size_average: return ssim_map.mean()
    else: return ssim_map.mean(1).mean(1).mean(1)

# 复用原本的 create_window (2D版)
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

class SSIMLoss3D(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1

    def forward(self, img1, img2):
        img1 = torch.nan_to_num(img1, nan=0.0)
        img2 = torch.nan_to_num(img2, nan=0.0)
        
        return F.relu(1 - _ssim_3d(img1, img2, None, self.window_size, self.channel, self.size_average))

class TVLoss3D(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss3D, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        # x: [B, C, T, H, W]
        batch_size = x.size()[0]
        h_x = x.size()[3]
        w_x = x.size()[4]
        
        count_h = x[:, :, :, 1:, :].numel()
        count_w = x[:, :, :, :, 1:].numel()
        
        # 计算空间上的 TV (H方向 + W方向)
        h_tv = torch.pow((x[:, :, :, 1:, :] - x[:, :, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, :, 1:] - x[:, :, :, :, :w_x - 1]), 2).sum()
        
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

class AdaptiveCVLoss(nn.Module):
    """(保留原本逻辑，适配 3D 输入)"""
    def __init__(self, eps=1e-6, max_weight=50.0):
        super(AdaptiveCVLoss, self).__init__()
        self.eps = eps
        self.max_weight = max_weight 

    def forward(self, pred, target):
        pred = torch.nan_to_num(pred.float(), nan=0.0)
        target = torch.nan_to_num(target.float(), nan=0.0)

        target_flat = target.reshape(-1)
        mask_nonzero = target_flat > self.eps
        weight_map_flat = torch.ones_like(target_flat)

        n_total = target_flat.numel()
        n_nonzero = mask_nonzero.sum()

        if n_nonzero < 10:
            diff = pred - target
            return torch.sqrt(diff * diff + self.eps**2).mean()

        ratio = n_total / (n_nonzero + 1.0)
        w_macro = torch.log1p(ratio.detach())

        valid_values = target_flat[mask_nonzero]
        # 添加 clamp 防止方差过大
        cv = valid_values.std() / (valid_values.mean() + self.eps)
        alpha = torch.clamp(cv, min=0.1, max=10.0)
        w_micro = 1.0 + alpha * torch.log1p(valid_values)

        combined_weight = torch.clamp(w_macro * w_micro, max=self.max_weight)
        weight_map_flat[mask_nonzero] = combined_weight
        
        diff = pred - target
        basic_loss = torch.sqrt(diff * diff + self.eps**2)
        return (basic_loss * weight_map_flat.view_as(target).detach()).mean()

# ==========================================
# 🛑 [HybridLoss] 集成版
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self, consistency_scale=4):
        super(HybridLoss, self).__init__()
        # 1. 基础损失模块
        self.adaptive_loss = AdaptiveCVLoss() 
        self.ssim_loss = SSIMLoss3D() # 升级为 3D
        self.tv_loss = TVLoss3D()     # 升级为 3D
        
        # 2. 物理一致性参数 (4km -> 1km, scale=4)
        self.scale = consistency_scale
        
        # 3. 动态权重: [CV, SSIM, TV, Consistency]
        self.w_params = nn.Parameter(torch.zeros(4)) 

    def forward(self, pred, target, input_mosaic_low_res=None):
        """
        pred: [B, 1, T, 160, 160] (高清预测)
        target: [B, 1, T, 160, 160] (高清真值)
        input_mosaic_low_res: [B, 1, T, 160, 160] (4km马赛克输入)
          - 注意：虽然 input_mosaic 已经是 160x160，但它本质上是由 4x4 的格子组成的。
          - 我们的目标是：AvgPool(pred) 应该等于 AvgPool(input_mosaic)。
        """
        
        # A. 细节与结构损失
        l_cv = self.adaptive_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        l_tv = self.tv_loss(pred)
        
        # B. 🔥 物理一致性损失 (Physical Consistency)
        # 逻辑：把预测的高清图，重新聚合回 4km 网格，看看总碳排放对不对得上。
        # 使用 avg_pool3d 模拟物理聚合过程
        
        # 如果训练脚本传了 input_mosaic，就用它；否则用 target 降采样代替 (兼容旧代码)
        target_reference = input_mosaic_low_res if input_mosaic_low_res is not None else target
        
        # 下采样到 40x40 (模拟 4km 物理网格)
        pred_down = F.avg_pool3d(pred, kernel_size=(1, self.scale, self.scale), stride=(1, self.scale, self.scale))
        ref_down = F.avg_pool3d(target_reference, kernel_size=(1, self.scale, self.scale), stride=(1, self.scale, self.scale))
        
        l_consist = F.l1_loss(pred_down, ref_down)
        
        # C. 动态加权
        weights = torch.exp(self.w_params)
        weights = weights / weights.sum() * 4.0 # 归一化
        
        total_loss = (weights[0] * l_cv + 
                      weights[1] * l_ssim + 
                      weights[2] * l_tv + 
                      weights[3] * l_consist)
                      
        return total_loss