# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    MultiScaleBlock3D,
    SFTLayer3D,
    EfficientContextBlock,
    MoEBlock,
    DropPath,           
    TemporalDWConv3d,   
    GatedFusion,
    BiMambaBlock        
)

class DepthwiseSeparableConv3d(nn.Module):
    """深度可分离卷积，标准结构"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.dw = nn.Conv3d(
            in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=False
        )
        self.pw = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))

class EnhancedResBlock(nn.Module):
    """
    [New] 增强型残差块:
    1. Temporal DWConv (先看时间)
    2. Spatial DWConv (再看空间)
    3. DropPath (随机深度正则化)
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.temporal_conv = TemporalDWConv3d(dim) # 时序混合
        self.spatial_conv = nn.Sequential(         # 空间混合
            DepthwiseSeparableConv3d(dim, dim),
            DepthwiseSeparableConv3d(dim, dim)
        )
        # 如果 drop_path > 0 则应用，否则是 Identity
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # 强制模型先融合前后帧信息
        t_feat = self.temporal_conv(x)
        # 提取空间特征
        s_feat = self.spatial_conv(t_feat)
        # 残差连接 + DropPath
        return x + self.drop_path(s_feat)

class DSTCarbonFormer(nn.Module):
    def __init__(self, aux_c=9, main_c=1, 
                 dim=96,               
                 norm_const=11.0, 
                 num_mamba_layers=2,   # [Config] 建议改为2，因为现在有3个Stage
                 num_res_blocks=4,     
                 drop_path_rate=0.1    
                 ):
        super().__init__()
        self.norm_const = float(norm_const)
        self.dim = dim

        # ===========================
        # 1. Heads (特征编码) - 保持不变
        # ===========================
        self.aux_head = nn.Sequential(
            nn.Conv3d(aux_c, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(dim, dim, 3, padding=1),
        )
        self.aux_multiscale = MultiScaleBlock3D(dim)
        
        self.main_head = nn.Sequential(
            nn.Conv3d(main_c, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(dim, dim, 3, padding=1),
        )

        # ===========================
        # 2. Fusion - 保持不变
        # ===========================
        self.sft1 = SFTLayer3D(dim)
        self.gated_fusion = GatedFusion(dim) 

        # ===========================
        # 3. Deep Body (深层特征体) - 微调 RIR
        # ===========================
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_res_blocks)]
        
        self.res_blocks = nn.ModuleList([
            EnhancedResBlock(dim, drop_path=dpr[i])
            for i in range(num_res_blocks)
        ])
        
        # [New] RIR (Residual in Residual) 收尾卷积
        # 用于在进入 Mamba 前整理特征
        self.body_tail = nn.Conv3d(dim, dim, 3, 1, 1)

        self.sft2 = SFTLayer3D(dim)
        self.moe_block = MoEBlock(dim)
        self.global_context = EfficientContextBlock(dim)

        # ===========================
        # 4. Hierarchical Bi-Mamba Bottleneck (3-Level U-Net) - 核心重构
        # ===========================
        # 结构: 120 -> 60 -> 30 -> 15(Global) -> 30 -> 60 -> 120
        
        mamba_dim = 64 

        # --- Level 1 Down: 120 -> 60 ---
        self.down1 = nn.Sequential(
            nn.Conv3d(dim, mamba_dim, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            nn.GroupNorm(8, mamba_dim),
            nn.GELU()
        )
        self.mamba_stage1 = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(mamba_dim), 
                BiMambaBlock(dim=mamba_dim, d_state=16, d_conv=4, expand=2)
            ) for _ in range(num_mamba_layers)
        ])

        # --- Level 2 Down: 60 -> 30 ---
        self.down2 = nn.Sequential(
            nn.Conv3d(mamba_dim, mamba_dim, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            nn.GroupNorm(8, mamba_dim),
            nn.GELU()
        )
        self.mamba_stage2 = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(mamba_dim), 
                BiMambaBlock(dim=mamba_dim, d_state=16, d_conv=4, expand=2)
            ) for _ in range(num_mamba_layers)
        ])

        # --- Level 3 Down: 30 -> 15 (上帝视角) ---
        self.down3 = nn.Sequential(
            nn.Conv3d(mamba_dim, mamba_dim, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            nn.GroupNorm(8, mamba_dim),
            nn.GELU()
        )
        self.mamba_stage3 = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(mamba_dim), 
                BiMambaBlock(dim=mamba_dim, d_state=16, d_conv=4, expand=2)
            ) for _ in range(num_mamba_layers)
        ])

        # --- Level 3 Up: 15 -> 30 (可学习上采样) ---
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(mamba_dim, mamba_dim, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),
            nn.GroupNorm(8, mamba_dim),
            nn.GELU()
        )
        # Fusion Level 2 (在 30x30 尺度融合)
        self.fusion_lvl2 = nn.Sequential(
            nn.Conv3d(mamba_dim * 2, mamba_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, mamba_dim),
            nn.GELU()
        )

        # --- Level 2 Up: 30 -> 60 ---
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(mamba_dim, mamba_dim, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),
            nn.GroupNorm(8, mamba_dim),
            nn.GELU()
        )
        # Fusion Level 1 (在 60x60 尺度融合)
        self.fusion_lvl1 = nn.Sequential(
            nn.Conv3d(mamba_dim * 2, mamba_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, mamba_dim),
            nn.GELU()
        )

        # --- Level 1 Up: 60 -> 120 ---
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(mamba_dim, dim, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),
            # 最后一次上采样不加激活，保持特征线性，方便与 Body 特征融合
        )

        # U-Net Body Skip Fusion (120x120)
        self.skip_fusion = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU()
        )

        # ===========================
        # 5. Tail (解码与输出)
        # ===========================
        self.tail = nn.Sequential(
            nn.Conv3d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(dim, main_c, 1),
        )
        
        # 环境设置
        self.use_channels_last_3d = os.environ.get("USE_CHANNELS_LAST_3D", "0") == "1"
        if self.use_channels_last_3d:
            self.to(memory_format=torch.channels_last_3d)

        self.aux_thr = float(os.environ.get("AUX_PRIOR_THR", "1e-6"))
        self._init_weights_logic()

    def _init_weights_logic(self):
        # [修改] 残差学习模式初始化
        # 最后一层初始化为 0，使得初始状态下 Residual=0，Output=Input(Base)
        last_layer = self.tail[-1]
        if isinstance(last_layer, nn.Conv3d):
            nn.init.constant_(last_layer.weight, 0.0)
            nn.init.constant_(last_layer.bias, 0.0)

    def _forward_mamba_stage(self, x, layers):
        """
        Helper function to handle flatten -> mamba -> unflatten
        x: [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape
        x_flat = x.view(B, C, -1).transpose(1, 2).contiguous() # [B, L, C]
        
        for layer in layers:
            # 隐式层级残差连接 (x + layer(x)) 
            # 这保证了每层 Mamba 都在学习增量
            x_flat = x_flat + layer(x_flat)
            
        out = x_flat.transpose(1, 2).view(B, C, T, H, W).contiguous()
        return out

    def _build_allow_mask(self, aux, main):
        if aux.shape[1] >= 7:
            aux_prior = (aux[:, 0:1, ...] + aux[:, 6:7, ...]) * 0.5
        else:
            aux_prior = aux[:, 0:1, ...]
        return torch.clamp((main > 0).float() + (aux_prior > self.aux_thr).float(), 0.0, 1.0)

    def forward(self, aux, main):
        if self.use_channels_last_3d:
            aux = aux.to(memory_format=torch.channels_last_3d)
            main = main.to(memory_format=torch.channels_last_3d)

        main_norm = main / self.norm_const
        
        # ===========================
        # Stage 1: Encoding
        # ===========================
        aux_feat = self.aux_head(aux)
        aux_feat = self.aux_multiscale(aux_feat)
        main_feat = self.main_head(main_norm)

        # 🚀 [Global Residual] 保存浅层基底特征 (120x120)
        shallow_feat = main_feat

        # ===========================
        # Stage 2: Fusion
        # ===========================
        x = self.sft1(main_feat, aux_feat)
        x = self.gated_fusion(x, aux_feat)

        # ===========================
        # Stage 3: Deep Body (RIR)
        # ===========================
        for block in self.res_blocks:
            x = block(x) 
        
        # [New] RIR (Residual in Residual) 连接
        # Body Output = Conv(Blocks(x)) + Shallow Input
        x = self.body_tail(x) + shallow_feat
        
        # 保存高清特征用于最外层 U-Net Skip
        x_high_res_skip = x

        # ===========================
        # Stage 4: Hierarchical Mamba Bottleneck (3-Level U-Net)
        # ===========================
        
        # --- 1. Level 1 Down: 120 -> 60 ---
        x_60_raw = self.down1(x)
        # Mamba Stage 1 (Capture Medium Freq)
        x_60 = self._forward_mamba_stage(x_60_raw, self.mamba_stage1)
        
        # --- 2. Level 2 Down: 60 -> 30 ---
        x_30_raw = self.down2(x_60)
        # Mamba Stage 2 (Capture Low Freq)
        x_30 = self._forward_mamba_stage(x_30_raw, self.mamba_stage2)
        
        # --- 3. Level 3 Down: 30 -> 15 ---
        x_15_raw = self.down3(x_30)
        # Mamba Stage 3 (Global Context / God's View)
        x_15 = self._forward_mamba_stage(x_15_raw, self.mamba_stage3)
        
        # --- 4. Level 3 Up: 15 -> 30 ---
        x_30_up = self.up3(x_15)
        # Fusion at 30x30 (Skip Connection from x_30)
        x_30_fused = self.fusion_lvl2(torch.cat([x_30_up, x_30], dim=1))
        
        # --- 5. Level 2 Up: 30 -> 60 ---
        x_60_up = self.up2(x_30_fused)
        # Fusion at 60x60 (Skip Connection from x_60)
        x_60_fused = self.fusion_lvl1(torch.cat([x_60_up, x_60], dim=1))
        
        # --- 6. Level 1 Up: 60 -> 120 ---
        x_120_out = self.up1(x_60_fused)

        # --- 7. Final Fusion with Body Features ---
        x_cat = torch.cat([x_high_res_skip, x_120_out], dim=1)
        x = self.skip_fusion(x_cat)

        # ===========================
        # Stage 5: Decoding
        # ===========================
        x = self.moe_block(x)          
        x = self.global_context(x)     
        x = self.sft2(x, aux_feat)     
        
        # ===========================
        # Stage 6: Output & Global Residual
        # ===========================
        # 1. 计算高频残差图 (Residual Map)
        # 此时 x 的数值可正可负
        residual = self.tail(x)
        
        # 2. 全局残差相加 (Additive)
        # 最终预测 = 平滑基底 + 高频残差
        pred = main_norm + residual
        
        # 3. 物理约束
        allow_mask = self._build_allow_mask(aux, main_norm)
        pred = pred * allow_mask * self.norm_const
        
        # 保证非负
        pred = F.relu(pred)
        
        return pred, pred