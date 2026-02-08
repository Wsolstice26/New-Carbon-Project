# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    MultiScaleBlock3D,
    SFTLayer3D,
    # EfficientContextBlock, # 已移除，用更强的 DualChannelAttention 替代
    DualChannelAttention,    # 导入新写的双路注意力
    MoEBlock,
    DropPath,           
    TemporalDWConv3d,   
    GatedFusion,
    BiMambaBlock        
)

# ==============================================================================
# 1. 核心模块：趋势编码器 (MambaTrendEncoder)
#    保持不变，这是模型的"眼睛"
# ==============================================================================

class MambaTrendEncoder(nn.Module):
    def __init__(self, in_c, dim, num_layers=3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Conv3d(in_c, dim, 1), nn.GELU())
        self.max_t, self.max_h, self.max_w = 12, 120, 120
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, self.max_t, self.max_h, self.max_w))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 保持 3 层 Mamba 以确保对低分趋势的深刻理解
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                BiMambaBlock(dim=dim, d_state=32, d_conv=4, expand=2)
            ) for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_feat = self.input_proj(x)
        if T <= self.max_t and H <= self.max_h and W <= self.max_w:
             pos_bias = F.interpolate(
                self.pos_embed[:, :, :T, :self.max_h, :self.max_w], 
                size=(T, H, W), mode='trilinear', align_corners=False
            )
             x_feat = x_feat + pos_bias
        
        # 使用 reshape + transpose 防止 view 报错
        x_flat = x_feat.reshape(B, -1, T*H*W).transpose(1, 2) 
        for layer in self.layers:
            x_flat = x_flat + layer(x_flat)
        x_flat = self.norm_final(x_flat)
        x_mamba = self.out_proj(x_flat)
        x_out = x_mamba.transpose(1, 2).reshape(B, -1, T, H, W)
        return x_out


# ==============================================================================
# 2. 基础组件
# ==============================================================================

class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.dw = nn.Conv3d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.pw(self.dw(x)))

class EnhancedResBlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.temporal_conv = TemporalDWConv3d(dim) 
        self.spatial_conv = nn.Sequential(DepthwiseSeparableConv3d(dim, dim), DepthwiseSeparableConv3d(dim, dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        return x + self.drop_path(self.spatial_conv(self.temporal_conv(x)))


# ==============================================================================
# 3. DSTCarbonFormer (轻量化 + 双头注意力版)
# ==============================================================================

class DSTCarbonFormer(nn.Module):
    def __init__(self, aux_c=9, main_c=1, 
                 dim=96,               
                 norm_const=11.0, 
                 num_mamba_layers=2,   # 这里的参数仅用于兼容性，内部我们会强制轻量化
                 num_res_blocks=4,     
                 drop_path_rate=0.1    
                 ):
        super().__init__()
        self.norm_const = float(norm_const)
        self.dim = dim

        # --- Stage 1: Encoding ---
        self.aux_head = nn.Sequential(nn.Conv3d(aux_c, dim, 3, padding=1), nn.GELU(), nn.Conv3d(dim, dim, 3, padding=1))
        self.aux_multiscale = MultiScaleBlock3D(dim)
        
        # Encoder: 保持强力 (3层)
        self.main_encoder = MambaTrendEncoder(main_c, dim, num_layers=3)
        self.main_align = nn.Sequential(nn.Conv3d(dim, dim, 3, 1, 1), nn.GELU())

        # --- Stage 2: Fusion ---
        self.sft1 = SFTLayer3D(dim)
        self.gated_fusion = GatedFusion(dim) 
        
        # 🚀 [双头注意力 1] 融合后的第一次清洗
        # 此时 Aux 和 Main 刚混合，用 Dual-Attention 剔除杂质
        self.fusion_attention = DualChannelAttention(dim, ratio=8)

        # --- Stage 3: Deep Body ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_res_blocks)]
        self.res_blocks = nn.ModuleList([EnhancedResBlock(dim, drop_path=dpr[i]) for i in range(num_res_blocks)])
        self.body_tail = nn.Conv3d(dim, dim, 3, 1, 1)

        # --- Stage 4: Light-weight Mamba U-Net (2-Level Only) ---
        # 瘦身策略：
        # 1. 通道数减半 (mamba_dim = dim // 2 = 48)
        # 2. 只有 2 级 (去掉 15x15 层，减少下采样次数)
        # 3. 每级只有 1 层 Mamba (Stack=1)
        mamba_dim = dim // 2  # 96 -> 48
        
        # Level 1 Down (120 -> 60)
        self.down1 = nn.Sequential(nn.Conv3d(dim, mamba_dim, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)), nn.GroupNorm(8, mamba_dim), nn.GELU())
        self.mamba_stage1 = nn.Sequential(nn.LayerNorm(mamba_dim), BiMambaBlock(dim=mamba_dim, d_state=16, d_conv=4, expand=2)) # 只用1层

        # Level 2 Down (60 -> 30) - 到底了
        self.down2 = nn.Sequential(nn.Conv3d(mamba_dim, mamba_dim, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)), nn.GroupNorm(8, mamba_dim), nn.GELU())
        self.mamba_stage2 = nn.Sequential(nn.LayerNorm(mamba_dim), BiMambaBlock(dim=mamba_dim, d_state=16, d_conv=4, expand=2)) # 只用1层

        # Level 2 Up (30 -> 60)
        self.up2 = nn.Sequential(nn.ConvTranspose3d(mamba_dim, mamba_dim, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)), nn.GroupNorm(8, mamba_dim), nn.GELU())
        self.fusion_lvl1 = nn.Sequential(nn.Conv3d(mamba_dim * 2, mamba_dim, kernel_size=1, bias=False), nn.GroupNorm(8, mamba_dim), nn.GELU())

        # Level 1 Up (60 -> 120)
        self.up1 = nn.Sequential(nn.ConvTranspose3d(mamba_dim, dim, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))) # Output channel回到 dim
        self.skip_fusion = nn.Sequential(nn.Conv3d(dim * 2, dim, kernel_size=1, bias=False), nn.GroupNorm(8, dim), nn.GELU())

        # --- Stage 5: Decoding ---
        self.sft2 = SFTLayer3D(dim)
        self.moe_block = MoEBlock(dim) # MoE 保持
        
        # 🚀 [双头注意力 2] 最终全局校准
        # 替换掉了原来的 EfficientContextBlock
        self.final_attention = DualChannelAttention(dim, ratio=8)

        # --- Stage 6: Tail ---
        self.tail = nn.Sequential(nn.Conv3d(dim, dim, 3, padding=1), nn.GELU(), nn.Conv3d(dim, main_c, 1))
        
        self.use_channels_last_3d = os.environ.get("USE_CHANNELS_LAST_3D", "0") == "1"
        if self.use_channels_last_3d:
            self.to(memory_format=torch.channels_last_3d)

        self.aux_thr = float(os.environ.get("AUX_PRIOR_THR", "1e-6"))
        self.res_scale = nn.Parameter(torch.ones(1) * 1.0)
        self._init_weights_logic()

    def _init_weights_logic(self):
        last_layer = self.tail[-1]
        if isinstance(last_layer, nn.Conv3d):
            nn.init.constant_(last_layer.weight, 0.0)
            nn.init.constant_(last_layer.bias, 0.0)

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
        
        # 1. Encoding
        aux_feat = self.aux_head(aux)
        aux_feat = self.aux_multiscale(aux_feat)
        
        main_feat_low = self.main_encoder(main_norm)
        
        # 使用 reshape + interpolate 防止 view 报错
        if main_feat_low.shape[-2:] != (120, 120):
            B, C, T, H_low, W_low = main_feat_low.shape
            main_feat_up = F.interpolate(
                main_feat_low.reshape(B, C*T, H_low, W_low), 
                size=(120, 120), mode='nearest'
            ).view(B, C, T, 120, 120)
        else:
            main_feat_up = main_feat_low
        main_feat = self.main_align(main_feat_up)
        
        # Main Skip 准备
        if main_norm.shape[-2:] != (120, 120):
             B, C, T, H_in, W_in = main_norm.shape
             main_skip = F.interpolate(
                main_norm.reshape(B, C*T, H_in, W_in), 
                size=(120, 120), mode='nearest'
            ).view(B, C, T, 120, 120)
        else:
            main_skip = main_norm

        shallow_feat = main_feat

        # 2. Fusion & Attention
        x = self.sft1(main_feat, aux_feat)
        x = self.gated_fusion(x, aux_feat)
        x = self.fusion_attention(x) # 🚀 清洗 1

        # 3. Deep Body
        for block in self.res_blocks:
            x = block(x) 
        x = self.body_tail(x) + shallow_feat
        x_high_res_skip = x

        # 4. Light-weight Mamba U-Net (2-Level)
        # Down 1
        x_60_raw = self.down1(x)
        B, C, T, H, W = x_60_raw.shape
        x_flat = x_60_raw.reshape(B, C, -1).transpose(1, 2)
        x_flat = x_flat + self.mamba_stage1(x_flat)
        x_60 = x_flat.transpose(1, 2).reshape(B, C, T, H, W)
        
        # Down 2 (Bottom)
        x_30_raw = self.down2(x_60)
        B, C, T, H, W = x_30_raw.shape
        x_flat = x_30_raw.reshape(B, C, -1).transpose(1, 2)
        x_flat = x_flat + self.mamba_stage2(x_flat)
        x_30 = x_flat.transpose(1, 2).reshape(B, C, T, H, W)
        
        # Up 2
        x_30_up = self.up2(x_30) # 直接上采样 x_30
        x_60_fused = self.fusion_lvl1(torch.cat([x_30_up, x_60], dim=1)) # Skip x_60
        
        # Up 1
        x_120_out = self.up1(x_60_fused)
        x_cat = torch.cat([x_high_res_skip, x_120_out], dim=1) # Skip High-Res
        x = self.skip_fusion(x_cat)

        # 5. Output
        x = self.moe_block(x)          
        x = self.sft2(x, aux_feat)     
        x = self.final_attention(x) # 🚀 清洗 2

        residual = self.tail(x)
        correction = self.res_scale * torch.tanh(residual) 
        pred = main_skip * (1.0 + correction)
        
        allow_mask = self._build_allow_mask(aux, main_skip)
        pred = pred * allow_mask * self.norm_const
        
        return F.relu(pred), F.relu(pred)