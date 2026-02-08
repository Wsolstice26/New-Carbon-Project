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
    保持不变
    """
    def __init__(self, use_charbonnier: bool = False, eps: float = 1e-6):
        super().__init__()
        self.use_charbonnier = bool(use_charbonnier)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, nz_ratio: torch.Tensor, cv: torch.Tensor) -> torch.Tensor:
        diff = pred - gt
        
        if self.use_charbonnier:
            loss_map = torch.sqrt(diff * diff + self.eps * self.eps)
        else:
            loss_map = diff.abs()
            
        mask_nz = gt > 0
        weights = torch.ones_like(gt)
        
        if mask_nz.any():
            if isinstance(nz_ratio, torch.Tensor) and nz_ratio.ndim == 1:
                w_global = torch.log(nz_ratio.view(-1, 1, 1, 1, 1) + 1e-6)
                w_global = w_global.expand_as(gt)
            else:
                w_global = torch.log(nz_ratio + 1e-6)
            
            if isinstance(cv, torch.Tensor) and cv.ndim == 1:
                cv_val = cv.view(-1, 1, 1, 1, 1).expand_as(gt)
            else:
                cv_val = cv

            w_local = (1.0 + torch.log1p(gt)) * cv_val
            w_final_nz = (w_global * w_local).clamp(min=1.0, max=20.0) 
            weights[mask_nz] = w_final_nz[mask_nz]

        return (loss_map * weights.detach()).sum() / pred.size(0)


# ============================================================
# Loss B: Sparsity prior (Sum Reduction)
# ============================================================
class SparsityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        return pred.abs().sum() / pred.size(0)


# ============================================================
# Loss C: Block Entropy Loss
# ============================================================
class BlockEntropyLoss(nn.Module):
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
# Criterion: HybridLoss (Auto-Weighted Multi-Task Version)
# ============================================================
class HybridLoss(nn.Module):
    def __init__(
        self,
        consistency_scale: int = 10,
        # 虽然是自动加权，但我们仍用这几个参数作为开关（0表示关闭）
        w_sparse: float = 1e-3, 
        w_ent: float = 1e-3,
        w_mse: float = 1.0,     # 🚀 [新增开关] 只要 >0 就启用 MSE
        ent_mode: str = "max",          
        target_entropy: float = 1.5,    
        use_charbonnier_A: bool = False,
    ):
        super().__init__()
        
        # 记录开关状态
        self.enable_mse = w_mse > 0
        self.enable_sparse = w_sparse > 0
        self.enable_ent = w_ent > 0

        # 1. 定义子 Loss
        self.loss_A = WeightedL1Loss(use_charbonnier=use_charbonnier_A)
        self.loss_B = SparsityLoss()
        self.loss_C = BlockEntropyLoss(
            scale=consistency_scale, 
            mode=ent_mode, 
            target_entropy=target_entropy
        )

        # 🚀 [新增] 定义可学习的权重参数 (log variances)
        # 对应: [L1, MSE, Sparse, Entropy]
        # 初始化为 0.0，意味着初始权重系数为 exp(-0) = 1.0
        self.log_vars = nn.Parameter(torch.zeros(4))

    def forward(
        self,
        pred: torch.Tensor,               
        target: torch.Tensor,             
        pred_100m: torch.Tensor = None,   
        nz_ratio_win: torch.Tensor = None,
        cv_log_win: torch.Tensor = None
    ) -> torch.Tensor:
        
        # --- 1. 计算原始 Loss (Raw Values) ---
        
        # L1 (必须计算)
        l1 = self.loss_A(pred, target, nz_ratio=nz_ratio_win, cv=cv_log_win)

        # MSE (手动 Sum Reduction)
        if self.enable_mse:
            diff = pred - target
            # 关键：这里用 Sum / Batch，为了和 L1 量级匹配
            l_mse = (diff * diff).sum() / pred.size(0)
        else:
            l_mse = torch.tensor(0.0, device=pred.device)

        # 辅助 Loss
        p_for_prior = pred_100m if pred_100m is not None else pred
        
        l_sparse = self.loss_B(p_for_prior) if self.enable_sparse else torch.tensor(0.0, device=pred.device)
        l_ent = self.loss_C(p_for_prior) if self.enable_ent else torch.tensor(0.0, device=pred.device)

        # --- 2. 自动加权 (Auto-Weighting) ---
        # 公式: Loss = 0.5 * (Raw_Loss * exp(-s) + s)
        # s = log_var. 模型会自动权衡：如果 loss 很难降，就调大 s (降低权重)。
        
        total_loss = 0
        
        # A. L1 (总是启用, Index 0)
        precision_l1 = torch.exp(-self.log_vars[0])
        total_loss += 0.5 * (precision_l1 * l1 + self.log_vars[0])
        
        # B. MSE (Index 1)
        if self.enable_mse:
            precision_mse = torch.exp(-self.log_vars[1])
            total_loss += 0.5 * (precision_mse * l_mse + self.log_vars[1])
            
        # C. Sparse (Index 2)
        if self.enable_sparse:
            precision_sp = torch.exp(-self.log_vars[2])
            total_loss += 0.5 * (precision_sp * l_sparse + self.log_vars[2])
            
        # D. Entropy (Index 3)
        if self.enable_ent:
            precision_ent = torch.exp(-self.log_vars[3])
            total_loss += 0.5 * (precision_ent * l_ent + self.log_vars[3])

        # --- 3. 记录日志 ---
        with torch.no_grad():
            weights = torch.exp(-self.log_vars) # 打印实际权重供观察
        
        self.last_losses = {
            "L1_raw": l1.detach(),
            "MSE_raw": l_mse.detach(),
            "Sparse_raw": l_sparse.detach(),
            "Ent_raw": l_ent.detach(),
            # 观察这些权重，看看模型到底听谁的
            "w_L1": weights[0].detach(),
            "w_MSE": weights[1].detach(),
            "w_Sparse": weights[2].detach(),
            "w_Ent": weights[3].detach(),
        }

        return total_loss