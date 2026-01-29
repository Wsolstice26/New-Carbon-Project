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
# 2. 核心损失模块定义
# ==========================================

class TVLoss(nn.Module):
    """
    全变分损失 (Total Variation Loss)
    作用：专门用于平滑图像，消除超分辨率中常见的网格伪影和高频噪点。
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
        # 计算水平和垂直方向的梯度差异
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class BalancedCharbonnierLoss(nn.Module):
    """
    🔥 [核心升级] 平衡掩码 Charbonnier Loss
    作用：
    1. 使用 Charbonnier (L1变体) 保证数值回归的鲁棒性。
    2. 引入平衡机制：强制 '城市区域' 和 '背景区域' 对 Loss 的贡献各占 50%。
       这解决了背景 0 值过多导致梯度被稀释的问题。
    """
    def __init__(self, eps=1e-3):
        super(BalancedCharbonnierLoss, self).__init__()
        self.eps = eps
    
    def forward(self, x, y):
        # 1. 计算基础误差图
        diff_sq = (x - y)**2
        loss_map = torch.sqrt(diff_sq + self.eps * self.eps)
        
        # 2. 创建非零掩码 (阈值设为 1e-6)
        mask = (y > 1e-6).float()
        inv_mask = 1.0 - mask
        
        # 3. 分别计算城市和背景的平均 Loss
        # 加上 1e-8 是为了防止分母为 0 (例如全黑图片)
        loss_city = (loss_map * mask).sum() / (mask.sum() + 1e-8)
        loss_bg = (loss_map * inv_mask).sum() / (inv_mask.sum() + 1e-8)
        
        # 4. 强制 50/50 平衡
        # 无论背景面积多大，它只能贡献一半的 Loss
        return 0.5 * loss_city + 0.5 * loss_bg

class SSIMLoss(torch.nn.Module):
    """
    结构相似性损失 (SSIM)
    作用：保证重建结果在视觉结构（路网、纹理）上与真值一致。
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        # 适配 5D 输入 (B, C, T, H, W) -> 展平为 4D 进行卷积计算
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
# 3. 自适应混合损失 (正数权重版)
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        
        # 初始化三个核心 Loss
        self.pixel_loss = BalancedCharbonnierLoss() # 升级为平衡版
        self.ssim_loss = SSIMLoss()
        self.tv_loss = TVLoss()
        
        # 🔥 [关键修改] 定义 3 个可学习的权重参数
        # 初始化为 0.0，这意味着初始时刻 exp(0)=1，即三者权重相等
        self.w_params = nn.Parameter(torch.zeros(3))

        # 🔥 新增：定义放大倍数 (Scale Factor)
        # 建议设为 100 或 1000，让 Loss 回到 0.x ~ 1.x 的区间
        self.loss_scale = 1000.0

    def forward(self, pred, target, input_main=None):
        # 1. 计算各分项 Loss
        l_pix = self.pixel_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        
        # TV Loss 需要处理 5D 数据的 reshape
        if pred.dim() == 5:
            b, c, t, h, w = pred.size()
            l_tv = self.tv_loss(pred.view(b*t, c, h, w))
        else:
            l_tv = self.tv_loss(pred)
            
        # 2. 🔥 权重自适应计算 (Softmax 归一化思想)
        # 使用 exp 确保权重永远为正数，避免出现负数 Loss
        weights = torch.exp(self.w_params) 
        
        # 归一化：让权重之和恒等于 3.0
        # 这样既能保持量级稳定，又能让模型动态调整三者的比例
        weights = weights / weights.sum() * 3.0
        
        # 3. 加权求和
        # weights[0] -> Pixel Loss (数值精度)
        # weights[1] -> SSIM Loss (结构纹理)
        # weights[2] -> TV Loss (去网格化)
        total_loss = (weights[0] * l_pix + 
                      weights[1] * l_ssim + 
                      weights[2] * l_tv)
        
        return total_loss * self.loss_scale