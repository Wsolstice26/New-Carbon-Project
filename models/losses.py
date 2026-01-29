import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

# ==========================================
# 1. 基础组件 (SSIM 辅助函数 - 保持不变)
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
    全变分损失 (Total Variation Loss) - 保持不变
    作用：消除网格伪影，平滑图像。
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

class AdaptiveCVLoss(nn.Module):
    """
    🔥 [v1.8 核心升级] 自适应变异系数平衡损失
    
    取代了原来的 BalancedCharbonnierLoss。
    新特性：
    1. 显式 NaN 清洗：保护 9060 XT 不被脏数据搞崩。
    2. 自动宏观平衡：用 log(Ratio) 替代死板的 50/50，适应 33% 稀疏度。
    3. 自动微观平衡：利用 CV (变异系数) 自动识别“超级排放点”，并用 Log 函数加权。
    """
    def __init__(self, eps=1e-6, max_weight=50.0):
        super(AdaptiveCVLoss, self).__init__()
        self.eps = eps
        self.max_weight = max_weight 

    def forward(self, pred, target):
        # 1. 安全清洗 (Safety First) - 强制转 float32 统计
        pred = torch.nan_to_num(pred, nan=0.0)
        target = torch.nan_to_num(target, nan=0.0)

        # 2. 准备数据
        target_flat = target.view(-1).float()
        mask_nonzero = target_flat > self.eps
        weight_map_flat = torch.ones_like(target_flat) # 默认为 1.0 (背景)

        # 3. 统计数量
        n_total = target_flat.numel()
        n_nonzero = mask_nonzero.sum()

        # 如果全是背景，只算基础误差
        if n_nonzero < 10:
            diff = pred - target
            loss = torch.sqrt(diff * diff + self.eps**2)
            return loss.mean()

        # 4. [宏观] Step 1: 0 vs Non-0 平衡
        ratio = n_total / (n_nonzero + 1.0)
        w_macro = torch.log1p(ratio.detach())

        # 5. [微观] Step 2: 基于 CV 的高值加权
        valid_values = target_flat[mask_nonzero]
        mu = valid_values.mean()
        std = valid_values.std()
        
        # 计算变异系数 CV
        cv = std / (mu + self.eps)
        alpha = torch.clamp(cv, min=0.1, max=10.0) # 限制敏感度范围

        # 计算微观权重
        w_micro = 1.0 + alpha * torch.log1p(valid_values)

        # 6. 组合权重 & 截断
        combined_weight = w_macro * w_micro
        combined_weight = torch.clamp(combined_weight, max=self.max_weight)
        
        # 填回权重图
        weight_map_flat[mask_nonzero] = combined_weight
        weight_map = weight_map_flat.view_as(target)

        # 7. 计算最终加权 Loss
        diff = pred - target
        basic_loss = torch.sqrt(diff * diff + self.eps**2)
        
        # detach() 权重，只优化预测值
        final_loss = (basic_loss * weight_map.detach()).mean()

        return final_loss

class SSIMLoss(torch.nn.Module):
    """
    结构相似性损失 (SSIM) - 保持不变
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        # 显式处理 NaN，防止 SSIM 计算出错
        img1 = torch.nan_to_num(img1, nan=0.0)
        img2 = torch.nan_to_num(img2, nan=0.0)

        # 适配 5D 输入
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
# 3. 自适应混合损失 (Hybrid Wrapper)
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        
        # 🔥 [修改点] 将旧的 BalancedCharbonnierLoss 替换为 AdaptiveCVLoss
        self.pixel_loss = AdaptiveCVLoss(max_weight=50.0) 
        self.ssim_loss = SSIMLoss()
        self.tv_loss = TVLoss()
        
        # 可学习的权重参数 (初始化为 0 -> 权重 1:1:1)
        self.w_params = nn.Parameter(torch.zeros(3))

        # Loss 放大倍数，保持 100.0 不变，防止数值下溢
        self.loss_scale = 100.0

    def forward(self, pred, target, input_main=None):
        # 1. 计算各分项 Loss
        l_pix = self.pixel_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        
        # TV Loss 维度适配
        if pred.dim() == 5:
            b, c, t, h, w = pred.size()
            l_tv = self.tv_loss(pred.view(b*t, c, h, w))
        else:
            l_tv = self.tv_loss(pred)
            
        # 2. 权重自适应 (Softmax 归一化)
        weights = torch.exp(self.w_params)
        weights = weights / weights.sum() * 3.0
        
        # 3. 加权求和
        total_loss = (weights[0] * l_pix + 
                      weights[1] * l_ssim + 
                      weights[2] * l_tv)
        
        return total_loss * self.loss_scale