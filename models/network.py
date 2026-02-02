import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    MultiScaleBlock3D,
    SFTLayer3D,
    EfficientContextBlock,
    MoEBlock,
    PhysicsConstraintLayer,
)


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.dw = nn.Conv3d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv3d(in_ch, out_ch, 1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class DSTCarbonFormer(nn.Module):
    def __init__(self, aux_c=9, main_c=1, dim=48, norm_const=11.0):
        super().__init__()

        # Aux branch
        self.aux_head = nn.Sequential(
            nn.Conv3d(aux_c, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(dim, dim, 3, padding=1),
        )
        self.aux_multiscale = MultiScaleBlock3D(dim)

        # Main branch
        self.main_head = nn.Sequential(
            nn.Conv3d(main_c, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(dim, dim, 3, padding=1),
        )

        self.sft1 = SFTLayer3D(dim)
        self.res1 = nn.Sequential(
            DepthwiseSeparableConv3d(dim, dim),
            DepthwiseSeparableConv3d(dim, dim),
        )

        self.sft2 = SFTLayer3D(dim)
        self.moe_block = MoEBlock(dim)

        self.global_context = EfficientContextBlock(dim)

        # Mamba-like down/up (保持原结构)
        self.mamba_down = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.mamba = nn.Sequential(
            nn.Conv3d(dim, dim, 1),
            nn.GELU(),
            nn.Conv3d(dim, dim, 1),
        )

        self.tail = nn.Sequential(
            nn.Conv3d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(dim, main_c, 1),
        )

        # Physics constraint
        self.constraint_layer = PhysicsConstraintLayer(scale_factor=10, norm_const=norm_const)

        # 可选：开启 channels_last_3d（不改数学，只改内存格式；ROCm 上常能更快）
        self.use_channels_last_3d = os.environ.get("USE_CHANNELS_LAST_3D", "0") == "1"
        if self.use_channels_last_3d:
            self.to(memory_format=torch.channels_last_3d)

    def _forward_mamba_safe(self, x):
        # 你原来就是一个小网络，这里保持接口
        return self.mamba(x)

    @staticmethod
    def _upsample_hw_per_t_bilinear(x_small: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        """
        等价替换：
        - 原来：F.interpolate(mode="trilinear") 但 T 维没变化
        - 现在：逐 T 做 2D bilinear，上采样 H/W
        x_small: [B, C, T, Hs, Ws]
        return:  [B, C, T, Ht, Wt]
        """
        B, C, T, Hs, Ws = x_small.shape
        Ht, Wt = out_hw

        # [B,C,T,Hs,Ws] -> [B,T,C,Hs,Ws] -> [B*T, C, Hs, Ws]
        x2d = x_small.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, Hs, Ws)

        # 2D bilinear upsample
        x2d_up = F.interpolate(x2d, size=(Ht, Wt), mode="bilinear", align_corners=False)

        # [B*T,C,Ht,Wt] -> [B,T,C,Ht,Wt] -> [B,C,T,Ht,Wt]
        x_up = x2d_up.view(B, T, C, Ht, Wt).permute(0, 2, 1, 3, 4).contiguous()
        return x_up

    def forward(self, aux, main, constraint_scale=None):
        """
        aux:  [B, 9, T, H, W]
        main: [B, 1, T, H, W]   (log1p(x)/norm_const)
        constraint_scale:
            - train: 120 (global conservation)
            - infer: 10  (1km conservation)
            - None: 跳过 constraint 计算（训练/benchmark 如果 loss 只用 pred_raw，可用 None）
        returns:
            out_constrained, pred_raw
        """

        if self.use_channels_last_3d:
            aux = aux.contiguous(memory_format=torch.channels_last_3d)
            main = main.contiguous(memory_format=torch.channels_last_3d)

        f_aux = self.aux_head(aux)
        f_aux = self.aux_multiscale(f_aux)

        f_main = self.main_head(main)

        f_main = self.sft1(f_main, f_aux)
        f_main = self.res1(f_main) + f_main

        f_main = self.sft2(f_main, f_aux)
        f_main = self.moe_block(f_main)

        f_global = self.global_context(f_main)

        f_small = self.mamba_down(f_global)
        f_mamba_small = self._forward_mamba_safe(f_small)

        # ✅ 替换掉 trilinear（原：mode="trilinear"）
        # 因为 T 不变，只缩放 H/W，所以逐 T 做 2D bilinear 与原逻辑等价，但 ROCm backward 通常更快
        f_mamba = self._upsample_hw_per_t_bilinear(
            f_mamba_small,
            out_hw=(f_global.shape[-2], f_global.shape[-1])
        )

        f_final = f_main + f_mamba
        residual = self.tail(f_final)

        # 约束前（log_norm 域）：保持非负，保证对应线性域非负
        pred_raw = (main + residual).clamp(min=0)

        # ✅ constraint_scale=None 时跳过 constraint 计算（训练/benchmark 常用）
        if constraint_scale is None:
            out_constrained = pred_raw
        else:
            out_constrained = self.constraint_layer(pred_raw, main, scale=constraint_scale)

        return out_constrained, pred_raw
