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
# 2. 损失函数类定义
# ==========================================

# 🔥 新增：TV Loss (全变分损失) - 专门解决"边界不平滑"
class TVLoss(nn.Module):
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

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    
    # 修改：支持传入 weight_map 进行加权
    def forward(self, x, y, weight_map=None):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if weight_map is not None:
            loss = loss * weight_map # 🔥 核心：对重点区域加权
        return torch.mean(loss)

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(1, 1, 1, 1)
        if torch.cuda.is_available(): self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kw//2, kh//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        if x.dim() == 5:
            b, c, t, h, w = x.size()
            x = x.view(b * t, c, h, w)
            y = y.view(b * t, c, h, w)
        return self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
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

class ConsistencyLoss(nn.Module):
    def __init__(self, norm_factor=11.0):
        super().__init__()
        self.loss = nn.L1Loss()
        self.norm_factor = norm_factor 
        
    def forward(self, pred_high_res, input_low_res):
        pred_real = torch.expm1(pred_high_res * self.norm_factor)
        input_real = torch.expm1(input_low_res * self.norm_factor)
        
        target_h, target_w = input_low_res.shape[-2:]
        t_dim = input_low_res.shape[2]
        pred_down_real = F.adaptive_avg_pool3d(pred_real, output_size=(t_dim, target_h, target_w))
        
        pred_down_log = torch.log1p(pred_down_real) / self.norm_factor
        input_log = torch.log1p(input_real) / self.norm_factor 
        
        return self.loss(pred_down_log, input_log)

# ==========================================
# 3. 混合损失函数 (HybridLoss) - 核心修改版
# ==========================================

class HybridLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.2, gamma=0.2, delta=1.0, eta=0.1):
        super(HybridLoss, self).__init__()
        # 权重配置 (根据您的需求进行了微调)
        self.alpha = alpha  # Pixel Loss (带加权)
        self.beta = beta    # SSIM Loss (调大，增强结构)
        self.gamma = gamma  # Edge Loss (调大，增强边界)
        self.delta = delta  # Consistency Loss (物理守恒)
        self.eta = eta      # 🔥 新增: TV Loss (平滑度)
        
        self.pixel_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgeLoss()
        self.cons_loss = ConsistencyLoss(norm_factor=11.0)
        self.tv_loss = TVLoss() # 初始化 TV Loss
        
    def forward(self, pred, target, input_main):
        # 1. 构建权重地图 (Weight Map)
        # 逻辑：如果 target > 0 (有碳排放)，权重设为 10.0；否则设为 1.0
        # 注意：target 已经被 Log+归一化了，所以 0 依然是 0，有值的地方是小数
        # 我们设置一个极小的阈值 1e-6 来判定非零区域
        
        mask = (target > 1e-6).float()
        # 权重公式：Background=1.0, Emission=10.0
        weight_map = 1.0 + mask * 9.0 
        
        # 2. 计算各项损失
        l_pix = self.pixel_loss(pred, target, weight_map=weight_map) # 传入权重
        l_ssim = self.ssim_loss(pred, target)
        l_edge = self.edge_loss(pred, target)
        l_cons = self.cons_loss(pred, input_main)
        
        # 处理 TV Loss (需先将 5D 转 4D: B*T, C, H, W)
        if pred.dim() == 5:
            b, c, t, h, w = pred.size()
            l_tv = self.tv_loss(pred.view(b*t, c, h, w))
        else:
            l_tv = self.tv_loss(pred)
            
        # 3. 总和
        total_loss = (self.alpha * l_pix + 
                      self.beta * l_ssim + 
                      self.gamma * l_edge + 
                      self.delta * l_cons + 
                      self.eta * l_tv)
        
        return total_loss