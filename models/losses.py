# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Loss A: Weighted Linear L1 Loss (Pixel-wise Sum Reduction)
# ============================================================
class WeightedL1Loss(nn.Module):
    """
    [线性一致性损失 - 像素级求和版] 
    直接优化 |pred - gt|。
    
    关键变化：
    使用 Sum Reduction (除以 BatchSize) 替代 Mean Reduction。
    这意味着：模型不再通过"平均"来稀释误差，而是必须面对每一个像素的绝对误差总和。
    这对于稀疏的高排放点（High Values）至关重要。
    """
    def __init__(self, use_charbonnier: bool = False, eps: float = 1e-6):
        super().__init__()
        self.use_charbonnier = bool(use_charbonnier)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, nz_ratio: torch.Tensor, cv: torch.Tensor) -> torch.Tensor:
        # pred/gt: [B,1,T,12,12] (物理数值)
        
        # 1. 计算线性差值
        diff = pred - gt
        
        if self.use_charbonnier:
            # Charbonnier Loss (L1 的平滑版)
            loss_map = torch.sqrt(diff * diff + self.eps * self.eps)
        else:
            # 标准 L1
            loss_map = diff.abs()
            
        # 2. 动态权重计算 (保持原逻辑，这对于处理长尾分布很重要)
        mask_nz = gt > 0
        weights = torch.ones_like(gt)
        
        if mask_nz.any():
            # A. 全局权重: log(nz_ratio)
            if isinstance(nz_ratio, torch.Tensor) and nz_ratio.ndim == 1:
                w_global = torch.log(nz_ratio.view(-1, 1, 1, 1, 1) + 1e-6)
                w_global = w_global.expand_as(gt)
            else:
                w_global = torch.log(nz_ratio + 1e-6)
            
            # B. 局部权重: (1 + log1p(GT)) * CV
            if isinstance(cv, torch.Tensor) and cv.ndim == 1:
                cv_val = cv.view(-1, 1, 1, 1, 1).expand_as(gt)
            else:
                cv_val = cv

            w_local = (1.0 + torch.log1p(gt)) * cv_val
            
            # 截断权重防止梯度爆炸 (1.0 ~ 20.0)
            w_final_nz = (w_global * w_local).clamp(min=1.0, max=20.0) 
            weights[mask_nz] = w_final_nz[mask_nz]

        # 🚀【核心修改】Mean -> Sum
        # 计算每个样本的总误差，然后对 Batch 取平均。
        # 这样既保留了 Sum 的强梯度特性，又防止 Batch Size 变化影响 Loss 规模。
        return (loss_map * weights.detach()).sum() / pred.size(0)


# ============================================================
# Loss B: Sparsity prior (Sum Reduction)
# ============================================================
class SparsityLoss(nn.Module):
    """
    [稀疏损失] 约束 100m 细节，防止底噪。
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        # 🚀【核心修改】Mean -> Sum
        # 保持与主 Loss 量级一致
        return pred.abs().sum() / pred.size(0)


# ============================================================
# Loss C: Block Entropy Loss
# ============================================================
class BlockEntropyLoss(nn.Module):
    """
    [熵损失] 约束 100m 纹理，防止方块效应。
    (注意：在 Sum 模式下，Entropy 的数值相对较小，需要在 Config 里调大权重或者直接忽略)
    """
    def __init__(
        self,
        scale: int = 10,
        mode: str = "max",
        target_entropy: float = 1.5,
        eps: float = 1e-8,
        soft_valid_k: float = 20.0,
    ):
        super().__init__()
        self.scale = int(scale)
        self.mode = str(mode)
        self.target_entropy = float(target_entropy)
        self.eps = float(eps)
        self.soft_valid_k = float(soft_valid_k)

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        x = pred.clamp(min=0.0)
        B, C, T, H, W = x.shape
        s = self.scale
        
        if H % s != 0 or W % s != 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        h_grid, w_grid = H // s, W // s
        n = s * s

        blocks = (
            x.view(B, C, T, h_grid, s, w_grid, s)
             .permute(0, 1, 2, 3, 5, 4, 6)
             .reshape(B, C, T, h_grid, w_grid, n)
        )

        block_sum = blocks.sum(dim=-1, keepdim=True)
        p = blocks / (block_sum + self.eps)
        entropy = -(p * torch.log(p + self.eps)).sum(dim=-1)

        soft_valid = torch.sigmoid(self.soft_valid_k * (block_sum.squeeze(-1) - self.eps)).to(entropy.dtype)
        denom = soft_valid.sum().clamp(min=1.0)
        entropy_mean = (entropy * soft_valid).sum() / denom

        if self.mode == "max":
            return -entropy_mean
        return torch.abs(entropy_mean - self.target_entropy)


# ============================================================
# Criterion: HybridLoss (Targeting High R2)
# ============================================================
class HybridLoss(nn.Module):
    def __init__(
        self,
        consistency_scale: int = 10,
        w_sparse: float = 1e-3,
        w_ent: float = 1e-3,
        ent_mode: str = "max",          
        target_entropy: float = 1.5,    
        use_charbonnier_A: bool = False,
    ):
        super().__init__()
        self.w_sparse = float(w_sparse)
        self.w_ent = float(w_ent)

        # 1. 主 Loss (Linear Sum)
        self.loss_A = WeightedL1Loss(use_charbonnier=use_charbonnier_A)
        
        # 2. 辅助 Loss
        self.loss_B = SparsityLoss()
        self.loss_C = BlockEntropyLoss(
            scale=consistency_scale, 
            mode=ent_mode, 
            target_entropy=target_entropy
        )

    def forward(
        self,
        pred: torch.Tensor,               
        target: torch.Tensor,             
        pred_100m: torch.Tensor = None,   
        nz_ratio_win: torch.Tensor = None,
        cv_log_win: torch.Tensor = None
    ) -> torch.Tensor:
        
        # 1. 主 Loss (Sum Reduction)
        lA = self.loss_A(pred, target, nz_ratio=nz_ratio_win, cv=cv_log_win)

        # 2. 辅助 Loss
        p_for_prior = pred_100m if pred_100m is not None else pred

        lB = torch.zeros((), device=pred.device)
        if self.w_sparse > 0:
            lB = self.loss_B(p_for_prior)
            
        lC = torch.zeros((), device=pred.device)
        if self.w_ent > 0:
            lC = self.loss_C(p_for_prior)

        # 直接求和
        total = lA + (self.w_sparse * lB) + (self.w_ent * lC)

        self.last_losses = {
            "lA_linear_sum": lA.detach(),
            "lB_sparse": lB.detach(),
            "lC_ent": lC.detach(),
        }

        return total