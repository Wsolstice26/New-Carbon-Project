import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

# ==========================================
# 1. 基础组件 (SSIM 辅助函数)
# ==========================================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    C1 = 0.01**2; C2 = 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average: return ssim_map.mean()
    else: return ssim_map.mean(1).mean(1).mean(1)

# ==========================================
# 2. 核心损失定义
# ==========================================

class TVLoss(nn.Module):
    """
    全变分损失：专门用于平滑图像，消除网格伪影和噪点。
    """
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class CharbonnierLoss(nn.Module):
    """
    鲁棒的 L1 Loss 变体，比标准 L1 收敛更稳。
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    
    def forward(self, x, y, weight_map=None):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if weight_map is not None:
            loss = loss * weight_map # 重点区域加权
        return torch.mean(loss)

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        # 适配 5D 输入 (B, C, T, H, W) -> (B*T, C, H, W)
        if img1.dim() == 5:
            b, c, t, h, w = img1.size()
            img1 = img1.view(b * t, c, h, w)
            img2 = img2.view(b * t, c, h, w)

        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda: window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
            
        ssim_val = _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return F.relu(1 - ssim_val)

# ==========================================
# 3. 自适应混合损失 (Auto-Weighted HybridLoss)
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        
        # 1. 定义具体的 Loss 计算模块
        self.pixel_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.tv_loss = TVLoss()
        
        # 2. 🔥 核心：定义可学习的参数 (Log Variance)
        # 初始化为 0.0，对应初始权重 = 0.5 (1 / (2*exp(0)))
        # 我们用 Parameter 包装它，优化器会自动更新它
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, pred, target, input_main=None):
        # 1. 权重地图 (针对高排放区的空间加权，保持不变)
        mask = (target > 1e-6).float()
        weight_map = 1.0 + mask * 9.0  # 背景=1.0, 城市=10.0
        
        # 2. 计算原始 Loss 值
        l_pix = self.pixel_loss(pred, target, weight_map=weight_map)
        l_ssim = self.ssim_loss(pred, target)
        
        if pred.dim() == 5:
            b, c, t, h, w = pred.size()
            l_tv = self.tv_loss(pred.view(b*t, c, h, w))
        else:
            l_tv = self.tv_loss(pred)
            
        # 3. 🔥 应用不确定性加权 (Kendall et al. CVPR 2018)
        # Loss = (1 / 2*sigma^2) * L + log(sigma)
        # 这里 sigma^2 = exp(log_var)
        
        # 项 1: Pixel Loss (数值)
        precision_pix = torch.exp(-self.log_vars[0])
        loss_pix = 0.5 * precision_pix * l_pix + 0.5 * self.log_vars[0]
        
        # 项 2: SSIM Loss (结构)
        precision_ssim = torch.exp(-self.log_vars[1])
        loss_ssim = 0.5 * precision_ssim * l_ssim + 0.5 * self.log_vars[1]
        
        # 项 3: TV Loss (平滑)
        precision_tv = torch.exp(-self.log_vars[2])
        loss_tv = 0.5 * precision_tv * l_tv + 0.5 * self.log_vars[2]
        
        # 4. 总和
        total_loss = loss_pix + loss_ssim + loss_tv
        
        return total_loss