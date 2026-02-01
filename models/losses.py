import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 基础组件 (适配 3D)
# ==========================================

def gaussian(window_size, sigma):
    # 生成 1D 高斯分布
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

class SSIMLoss3D(nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(SSIMLoss3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        
        # 🚀 [优化] 初始化时创建 Window 并注册为 Buffer
        # 这样避免了每次 Forward 都在 CPU 创建 tensor 再传给 GPU
        window = self.create_window(window_size, channel)
        self.register_buffer('window', window)

    def create_window(self, window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        # 生成 2D 高斯核 [1, 1, 11, 11]
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        
        # 🔥 必须扩展为 4D: [C, 1, H, W] 才能被 F.conv2d 接受
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim_3d(self, img1, img2):
        # img: [B, C, T, H, W] -> reshape -> [B*T, C, H, W]
        b, c, t, h, w = img1.shape
        img1_2d = img1.reshape(-1, c, h, w)
        img2_2d = img2.reshape(-1, c, h, w)
        
        # 自动获取 Buffer 中的 window
        window = self.window
        if window.type_as(img1) != img1.type():
            window = window.type_as(img1)

        # 🎨 [优化] 使用反射填充 (Reflection Padding) 代替默认补零
        padding = self.window_size // 2
        
        def conv_valid(input, window):
            # 先 pad 再 conv，减少边界效应
            padded = F.pad(input, (padding, padding, padding, padding), mode='reflect')
            return F.conv2d(padded, window, padding=0, groups=c)

        mu1 = conv_valid(img1_2d, window)
        mu2 = conv_valid(img2_2d, window)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = conv_valid(img1_2d * img1_2d, window) - mu1_sq
        sigma2_sq = conv_valid(img2_2d * img2_2d, window) - mu2_sq
        sigma12 = conv_valid(img1_2d * img2_2d, window) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        # 鲁棒性处理：NaN 替换
        if torch.isnan(img1).any() or torch.isnan(img2).any():
            img1 = torch.nan_to_num(img1, nan=0.0)
            img2 = torch.nan_to_num(img2, nan=0.0)
            
        return torch.clamp(1.0 - self._ssim_3d(img1, img2), min=0.0, max=1.0)


class TVLoss3D(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss3D, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        # x: [B, C, T, H, W]
        # 🎨 [优化] 升级为 L1 TV Loss (各向异性)，更好地保留尖锐边缘
        h_tv = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).sum()
        w_tv = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).sum()
        
        count = x.numel()
        return self.tv_loss_weight * (h_tv + w_tv) / count


class AdaptiveCVLoss(nn.Module):
    """
    自适应变异系数损失
    🚀 优化：实现了 Batch 维度独立计算，防止 Batch 内样本相互干扰
    """
    def __init__(self, eps=1e-6, max_weight=10.0):
        super(AdaptiveCVLoss, self).__init__()
        self.eps = eps
        self.max_weight = max_weight 

    def forward(self, pred, target):
        pred = torch.nan_to_num(pred.float(), nan=0.0)
        target = torch.nan_to_num(target.float(), nan=0.0)

        # 1. 基础 Loss (Charbonnier Loss: 平滑 L1)
        diff = pred - target
        basic_loss = torch.sqrt(diff**2 + self.eps**2) # [B, C, T, H, W]

        # 2. 计算动态权重 (向量化，无 CPU sync)
        with torch.no_grad():
            mask_nonzero = (target > self.eps).float()
            
            # 统计每个样本的非零像素个数 [B, 1, 1, 1, 1]
            n_nonzero = mask_nonzero.sum(dim=(1, 2, 3, 4), keepdim=True)
            n_total = float(target.shape[1] * target.shape[2] * target.shape[3] * target.shape[4])
            
            # (A) 宏观权重
            ratio = n_total / (n_nonzero + 1.0)
            w_macro = torch.log1p(ratio)

            # (B) 微观权重
            target_masked = target * mask_nonzero
            mean_val = target_masked.sum(dim=(1, 2, 3, 4), keepdim=True) / (n_nonzero + self.eps)
            
            var_val = (target_masked - mean_val)**2 * mask_nonzero
            std_val = torch.sqrt(var_val.sum(dim=(1, 2, 3, 4), keepdim=True) / (n_nonzero + self.eps))
            
            cv = std_val / (mean_val + self.eps)
            alpha = torch.clamp(cv, min=0.1, max=10.0)
            w_micro = 1.0 + alpha * torch.log1p(target)

            combined_weight = torch.clamp(w_macro * w_micro, max=self.max_weight)
            
            # 只有当非零像素足够多时才启用 CV 权重
            valid_sample_mask = (n_nonzero > 10).float()
            final_weight_map = combined_weight * valid_sample_mask + 1.0 * (1.0 - valid_sample_mask)

        # 3. 加权 Loss
        weighted_loss = basic_loss * final_weight_map
        return weighted_loss.mean()


# ==========================================
# 🛑 [HybridLoss] 集成版 (Softmax 权重)
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self, consistency_scale=4):
        super(HybridLoss, self).__init__()
        # 1. 基础损失模块
        self.adaptive_loss = AdaptiveCVLoss() 
        self.ssim_loss = SSIMLoss3D()
        self.tv_loss = TVLoss3D(tv_loss_weight=1.0)
        
        # 2. 物理一致性参数
        self.scale = consistency_scale
        
        # 3. 动态权重 (可学习的权重参数)
        # 🔥 修改: 改回 w_params，初始为 0
        self.w_params = nn.Parameter(torch.zeros(4)) 

    def forward(self, pred, target, input_mosaic_low_res=None):
        """
        pred: [B, 1, T, 160, 160]
        target: [B, 1, T, 160, 160]
        input_mosaic_low_res: [B, 1, T, 160, 160]
        """
        
        # A. 细节与结构损失
        l_cv = self.adaptive_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        l_tv = self.tv_loss(pred)
        
        # B. 物理一致性损失
        target_reference = input_mosaic_low_res if input_mosaic_low_res is not None else target
        
        # 物理降采样 (AvgPool)
        pred_down = F.avg_pool3d(pred, kernel_size=(1, self.scale, self.scale), stride=(1, self.scale, self.scale))
        ref_down = F.avg_pool3d(target_reference, kernel_size=(1, self.scale, self.scale), stride=(1, self.scale, self.scale))
        
        l_consist = F.l1_loss(pred_down, ref_down)
        
        # C. 自动加权 (Softmax Weighted)
        # 🔥 修改: 使用 Softmax 确保权重为正且和为1
        weights = torch.softmax(self.w_params, dim=0)
        
        # 放大权重，让初始值接近 1.0 (否则初始梯度太小)
        weights = weights * 4.0
        
        # 加权求和 (每一项都肯定是正数)
        loss = (weights[0] * l_cv + 
                weights[1] * l_ssim + 
                weights[2] * l_tv + 
                weights[3] * l_consist)
        
        return loss