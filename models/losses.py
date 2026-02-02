import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# 0. Weakly-supervised Weighted Pixel Loss  (NEW)
# ============================================================
class WeightedPixelLoss(nn.Module):
    """
    Weakly-supervised pixel reconstruction loss.

    - Macro: global log(Nz / Nnz) boost for non-zero pixels
    - Micro: w_micro = 1 + alpha * log1p(y)
    - Normalized: sum(w * loss) / sum(w)   (avoid dilution)
    """
    def __init__(
        self,
        norm_factor: float,
        global_nz_ratio: float,
        global_cv_log: float,
        eps: float = 1e-6,
        max_weight: float = 50.0,
    ):
        super().__init__()
        self.norm_factor = float(norm_factor)
        self.eps = eps
        self.max_weight = max_weight

        # ===== Macro: non-zero global log amplifier =====
        self.w_macro_nz = math.log1p(global_nz_ratio)

        # ===== Micro: global CV_log -> alpha =====
        self.alpha = float(global_cv_log)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred / target: log1p(x) / norm_factor domain
        """

        # ----- linear-domain emission (for weighting only) -----
        y_lin = torch.expm1(target * self.norm_factor).clamp(min=0.0)

        mask_nz = (y_lin > 0).to(pred.dtype)
        mask_z  = 1.0 - mask_nz

        # ----- base reconstruction loss (Charbonnier in log-domain) -----
        diff = pred - target
        base_loss = torch.sqrt(diff * diff + self.eps**2)

        # ----- micro weight (non-zero only) -----
        w_micro = 1.0 + self.alpha * torch.log1p(y_lin)

        # ----- macro + micro combination -----
        w = torch.ones_like(y_lin, dtype=pred.dtype)
        w = w + mask_nz * self.w_macro_nz
        w = w * (mask_z + mask_nz * w_micro)
        w = torch.clamp(w, max=self.max_weight)

        # ----- normalized weighted loss -----
        return (w.detach() * base_loss).sum() / (w.sum() + self.eps)


# ============================================================
# 1. Edge-Aware TV Loss (unchanged)
# ============================================================
class EdgeAwareTVLoss(nn.Module):
    """
    pred: [B, 1, T, H, W]
    aux:  [B, C, T, H, W]
    """
    def __init__(self, alpha: float = 3.0, beta: float = 0.1):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    @staticmethod
    def _get_gradient(x: torch.Tensor) -> torch.Tensor:
        h_grad = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        h_grad = F.pad(h_grad, (0, 0, 0, 1, 0, 0), mode="replicate")

        w_grad = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
        w_grad = F.pad(w_grad, (0, 1, 0, 0, 0, 0), mode="replicate")

        return h_grad + w_grad

    def forward(self, pred: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        pred_safe = pred.clamp(min=0)

        aux_guide = (aux[:, 0:1, ...] + aux[:, 6:7, ...]) / 2.0
        aux_grad = self._get_gradient(aux_guide)

        scale = aux_grad.mean().clamp(min=1e-8)
        aux_grad_norm = aux_grad / scale

        weight = torch.exp(-self.alpha * aux_grad_norm)
        pred_grad = self._get_gradient(pred_safe)

        return (weight * pred_grad).mean() + self.beta * pred_grad.mean()


# ============================================================
# 2. Distribution Loss (unchanged)
# ============================================================
class DistributionLoss(nn.Module):
    def __init__(self, target_entropy: float = 1.5, scale: int = 10, eps: float = 1e-8):
        super().__init__()
        self.target_entropy = float(target_entropy)
        self.scale = int(scale)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        pred_safe = pred.clamp(min=0)
        B, C, T, H, W = pred_safe.shape

        s = self.scale
        if (H % s) != 0 or (W % s) != 0:
            raise ValueError(f"H,W must be divisible by scale={s}, got H,W={(H, W)}")

        h_grid, w_grid = H // s, W // s
        n = s * s

        blocks = (
            pred_safe.view(B, C, T, h_grid, s, w_grid, s)
                     .permute(0, 1, 2, 3, 5, 4, 6)
                     .reshape(B, C, T, h_grid, w_grid, n)
        )

        block_sum = blocks.sum(dim=-1, keepdim=True)
        valid = (block_sum.squeeze(-1) > self.eps)

        p = blocks / (block_sum + self.eps)
        entropy_block = -(p * torch.log(p + self.eps)).sum(dim=-1)

        if valid.any():
            entropy_mean = entropy_block[valid].mean()
        else:
            entropy_mean = torch.zeros((), device=pred.device, dtype=pred.dtype)

        return torch.abs(entropy_mean - self.target_entropy)


# ============================================================
# 3. Temporal Loss (unchanged)
# ============================================================
class TemporalLoss(nn.Module):
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        pred_safe = pred.clamp(min=0)
        diff = torch.abs(pred_safe[:, :, 1:, ...] - pred_safe[:, :, :-1, ...])
        return diff.mean()


# ============================================================
# 4. Hybrid Loss (UPDATED: + Pixel Loss)
# ============================================================
class HybridLoss(nn.Module):
    """
    Loss components:
      0) Pixel reconstruction (weak supervision, weighted)
      1) Pre-Consistency (linear-domain, 1km)
      2) Edge-aware TV
      3) Entropy
      4) Temporal
    """
    def __init__(
        self,
        consistency_scale: int = 10,
        norm_factor: float = 11.0,
        global_nz_ratio: float = 100.0,
        global_cv_log: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.scale = int(consistency_scale)
        self.norm_factor = float(norm_factor)
        self.eps = float(eps)

        # ----- NEW pixel loss -----
        self.pixel_loss = WeightedPixelLoss(
            norm_factor=norm_factor,
            global_nz_ratio=global_nz_ratio,
            global_cv_log=global_cv_log,
        )

        self.edge_tv_loss = EdgeAwareTVLoss(alpha=3.0)
        self.dist_loss = DistributionLoss(target_entropy=1.5, scale=self.scale)
        self.temp_loss = TemporalLoss()

        # log_vars: [Pixel, PreCons, EdgeTV, Entropy, Temporal]
        self.log_vars = nn.Parameter(torch.zeros(5))

    def _to_linear(self, x_log_norm: torch.Tensor) -> torch.Tensor:
        x_lin = torch.expm1(x_log_norm * self.norm_factor)
        return x_lin.clamp(min=0.0)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux: torch.Tensor,
        pred_raw=None
    ) -> torch.Tensor:
        main_pred = pred_raw if pred_raw is not None else pred

        # (0) Pixel reconstruction (weak supervision)
        l_pix = self.pixel_loss(main_pred, target)

        # (1) Pre-Consistency @ linear domain
        pred_lin = self._to_linear(main_pred)
        target_lin = self._to_linear(target)

        pred_down = F.avg_pool3d(
            pred_lin,
            kernel_size=(1, self.scale, self.scale),
            stride=(1, self.scale, self.scale)
        )
        target_down = F.avg_pool3d(
            target_lin,
            kernel_size=(1, self.scale, self.scale),
            stride=(1, self.scale, self.scale)
        )
        l_pre = F.l1_loss(pred_down, target_down)

        # (2) Edge-aware TV
        l_edge = self.edge_tv_loss(main_pred, aux)

        # (3) Entropy
        l_ent = self.dist_loss(main_pred)

        # (4) Temporal
        l_temp = self.temp_loss(main_pred)

        losses = [l_pix, l_pre, l_edge, l_ent, l_temp]

        total = 0.0
        for i, li in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * li + self.log_vars[i]

        return total
