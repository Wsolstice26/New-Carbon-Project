# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    MultiScaleBlock3D,
    SFTLayer3D,
    # EfficientContextBlock, # [移除] 用更强的 DualChannelAttention 替代
    DualChannelAttention,    # [新增] 导入双路注意力
    MoEBlock,
    DropPath,           
    TemporalDWConv3d,   
    GatedFusion,
    BiMambaBlock        
)

# ==============================================================================
# 1. 核心模块：趋势编码器 (MambaTrendEncoder)
#    [修改] 引入渐进式特征提取 (Progressive Stem)
# ==============================================================================

class MambaTrendEncoder(nn.Module):
    """
    [精细化调整版] Mamba 趋势编码器
    改进点：
    1. Input Projection: 改为渐进式 3x3 卷积 (1->24->48->96)，增强空间感知
    2. Positional Embedding: 3D Learnable
    3. Mamba Stack: Deep Bi-Mamba
    """
    def __init__(self, in_c, dim, num_layers=3):
        super().__init__()
        
        # ⬇️⬇️⬇️ [修改 1] 渐进式升维 Stem ⬇️⬇️⬇️
        # 从 1 -> 96 过于剧烈，且 1x1 卷积看不到邻域
        # 改为：1 -> 24 -> 48 -> 96，配合 3x3 卷积提取局部纹理
        mid_c1 = dim // 4  # 24
        mid_c2 = dim // 2  # 48
        
        self.input_proj = nn.Sequential(
            # Step 1: 1 -> 24 (感知局部 3x3 邻域)
            nn.Conv3d(in_c, mid_c1, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_c1), # 使用 InstanceNorm 保持物理量级的独立性
            nn.GELU(),
            
            # Step 2: 24 -> 48 (加深理解)
            nn.Conv3d(mid_c1, mid_c2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_c2),
            nn.GELU(),
            
            # Step 3: 48 -> 96 (最终映射)
            nn.Conv3d(mid_c2, dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        # ⬆️⬆️⬆️ [修改结束] ⬆️⬆️⬆️

        self.max_t, self.max_h, self.max_w = 12, 120, 120
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, self.max_t, self.max_h, self.max_w))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
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
        
        x_flat = x_feat.view(B, -1, T*H*W).transpose(1, 2) 
        for layer in self.layers:
            x_flat = x_flat + layer(x_flat)
        x_flat = self.norm_final(x_flat)
        x_mamba = self.out_proj(x_flat)
        x_out = x_mamba.transpose(1, 2).view(B, -1, T, H, W)
        return x_out


# ==============================================================================
# 2. 原有基础组件 (保持原样)
# ==============================================================================

class DepthwiseSeparableConv3d(nn.Module):
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
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.temporal_conv = TemporalDWConv3d(dim) 
        self.spatial_conv = nn.Sequential(         
            DepthwiseSeparableConv3d(dim, dim),
            DepthwiseSeparableConv3d(dim, dim)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        t_feat = self.temporal_conv(x)
        s_feat = self.spatial_conv(t_feat)
        return x + self.drop_path(s_feat)

# ==============================================================================
# 3. DSTCarbonFormer 主模型 (含 DualAttention + Progressive Aux Stem)
# ==============================================================================

class DSTCarbonFormer(nn.Module):
    def __init__(self, aux_c=9, main_c=1, 
                 dim=96,               
                 norm_const=11.0, 
                 num_mamba_layers=2,   
                 num_res_blocks=4,     
                 drop_path_rate=0.1    
                 ):
        super().__init__()
        self.norm_const = float(norm_const)
        self.dim = dim

        # ===========================
        # 1. Heads (特征编码)
        # ===========================
        
        # ⬇️⬇️⬇️ [修改 2] Aux Head 渐进式升维 ⬇️⬇️⬇️
        # 9 -> 48 -> 96
        self.aux_head = nn.Sequential(
            nn.Conv3d(aux_c, dim // 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(dim // 2),
            nn.GELU(),
            
            nn.Conv3d(dim // 2, dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(dim),
            nn.GELU(),
            
            nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        )
        # ⬆️⬆️⬆️ [修改结束] ⬆️⬆️⬆️
        
        self.aux_multiscale = MultiScaleBlock3D(dim)
        
        self.main_encoder = MambaTrendEncoder(main_c, dim)
        self.main_align = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1),
            nn.GELU()
        )

        # ===========================
        # 2. Fusion
        # ===========================
        self.sft1 = SFTLayer3D(dim)
        self.gated_fusion = GatedFusion(dim) 
        
        # 🚀 [新增 1] 融合后的第一次清洗 (Dual-Attention)
        self.fusion_attention = DualChannelAttention(dim, ratio=8)

        # ===========================
        # 3. Deep Body
        # ===========================
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_res_blocks)]
        
        self.res_blocks = nn.ModuleList([
            EnhancedResBlock(dim, drop_path=dpr[i])
            for i in range(num_res_blocks)
        ])
        
        self.body_tail = nn.Conv3d(dim, dim, 3, 1, 1)

        self.sft2 = SFTLayer3D(dim)
        self.moe_block = MoEBlock(dim)
        
        # 🚀 [新增 2] 最终全局校准 (Dual-Attention)
        self.final_attention = DualChannelAttention(dim, ratio=8)

        # ===========================
        # 4. Hierarchical Bi-Mamba Bottleneck (保持原样 3 层结构)
        # ===========================
        mamba_dim = 64 

        # Level 1 Down
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

        # Level 2 Down
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

        # Level 3 Down
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

        # Level 3 Up
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(mamba_dim, mamba_dim, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),
            nn.GroupNorm(8, mamba_dim),
            nn.GELU()
        )
        self.fusion_lvl2 = nn.Sequential(
            nn.Conv3d(mamba_dim * 2, mamba_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, mamba_dim),
            nn.GELU()
        )

        # Level 2 Up
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(mamba_dim, mamba_dim, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),
            nn.GroupNorm(8, mamba_dim),
            nn.GELU()
        )
        self.fusion_lvl1 = nn.Sequential(
            nn.Conv3d(mamba_dim * 2, mamba_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, mamba_dim),
            nn.GELU()
        )

        # Level 1 Up
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(mamba_dim, dim, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),
        )

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

    def _forward_mamba_stage(self, x, layers):
        B, C, T, H, W = x.shape
        x_flat = x.view(B, C, -1).transpose(1, 2).contiguous() 
        for layer in layers:
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
        
        main_feat_low = self.main_encoder(main_norm)
        
        if main_feat_low.shape[-2:] != (120, 120):
            B, C, T, H_low, W_low = main_feat_low.shape
            main_feat_up = F.interpolate(
                main_feat_low.reshape(B, C*T, H_low, W_low), 
                size=(120, 120), 
                mode='nearest'
            ).view(B, C, T, 120, 120)
        else:
            main_feat_up = main_feat_low
            
        main_feat = self.main_align(main_feat_up)
        
        if main_norm.shape[-2:] != (120, 120):
             B, C, T, H_in, W_in = main_norm.shape
             main_skip = F.interpolate(
                main_norm.reshape(B, C*T, H_in, W_in), 
                size=(120, 120), 
                mode='nearest'
            ).view(B, C, T, 120, 120)
        else:
            main_skip = main_norm

        shallow_feat = main_feat

        # ===========================
        # Stage 2: Fusion
        # ===========================
        x = self.sft1(main_feat, aux_feat)
        x = self.gated_fusion(x, aux_feat)
        
        # 🚀 [调用 1] 融合后的第一次清洗
        x = self.fusion_attention(x)

        # ===========================
        # Stage 3: Deep Body
        # ===========================
        for block in self.res_blocks:
            x = block(x) 
        
        x = self.body_tail(x) + shallow_feat
        x_high_res_skip = x

        # ===========================
        # Stage 4: Hierarchical Mamba Bottleneck
        # ===========================
        # Level 1
        x_60_raw = self.down1(x)
        x_60 = self._forward_mamba_stage(x_60_raw, self.mamba_stage1)
        # Level 2
        x_30_raw = self.down2(x_60)
        x_30 = self._forward_mamba_stage(x_30_raw, self.mamba_stage2)
        # Level 3
        x_15_raw = self.down3(x_30)
        x_15 = self._forward_mamba_stage(x_15_raw, self.mamba_stage3)
        # Up 3
        x_30_up = self.up3(x_15)
        x_30_fused = self.fusion_lvl2(torch.cat([x_30_up, x_30], dim=1))
        # Up 2
        x_60_up = self.up2(x_30_fused)
        x_60_fused = self.fusion_lvl1(torch.cat([x_60_up, x_60], dim=1))
        # Up 1
        x_120_out = self.up1(x_60_fused)

        x_cat = torch.cat([x_high_res_skip, x_120_out], dim=1)
        x = self.skip_fusion(x_cat)

        # ===========================
        # Stage 5: Decoding
        # ===========================
        x = self.moe_block(x)          
        # x = self.global_context(x) # [移除]
        x = self.sft2(x, aux_feat)     
        
        # 🚀 [调用 2] 最终校准
        x = self.final_attention(x)

        # ===========================
        # Stage 6: Output & Global Residual
        # ===========================
        residual = self.tail(x)
        correction = self.res_scale * torch.tanh(residual) 
        pred = main_skip * (1.0 + correction)
        
        allow_mask = self._build_allow_mask(aux, main_skip)
        pred = pred * allow_mask * self.norm_const
        
        return F.relu(pred), F.relu(pred)