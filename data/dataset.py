import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import os
import json

# 全局归一化参数
NORM_ROAD = 11.0 
NORM_NTL_LOG = 6.0   
NORM_MAIN_LOG = 11.0

class DualStreamDataset(Dataset):
    def __init__(self, data_dir, split_config_path, mode='train', time_window=3):
        # 强制指向实际数据文件夹
        self.data_dir = "/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_120"
        self.window = time_window
        self.mode = mode
        
        # 1. 加载索引配置
        with open(split_config_path, 'r') as f:
            config = json.load(f)
        
        if mode == 'train':
            self.indices = config['train_indices']
        elif mode == 'val':
            self.indices = config['val_indices']
        else:
            self.indices = config['test_indices']
            
        self.all_years = range(2014, 2024)
        
        # 辅助流归一化因子
        self.aux_factors = torch.tensor([1.0]*9).float().view(9, 1, 1, 1)

        # 2. 构建样本索引列表
        self.samples = []
        for idx in self.indices:
            for i in range(len(self.all_years) - self.window + 1):
                years = list(self.all_years[i : i+self.window])
                self.samples.append({'patch_idx': idx, 'years': years})
        
        # 3. 预加载数据到内存 (32GB RAM 模式)
        print(f"🚀 [{mode}] 正在加载切片数据 (Path: {self.data_dir})...")
        self.cache_X = {} 
        self.cache_Y = {} 
        
        for y in self.all_years:
            x_path = os.path.join(self.data_dir, f"X_{y}.npy")
            y_path = os.path.join(self.data_dir, f"Y_{y}.npy")
            
            if os.path.exists(x_path):
                # 🔥 [关键] 使用 .copy() 确保内存独立且连续，防止多线程段错误
                self.cache_X[y] = np.load(x_path).copy()
                self.cache_Y[y] = np.load(y_path).copy()
        print(f"✅ [{mode}] 数据加载完成，总样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        p_idx = item['patch_idx']
        years = item['years']
        
        feat_stack = []
        coarse_stack = []
        
        for y in years:
            feat_stack.append(self.cache_X[y][p_idx])
            coarse_stack.append(self.cache_Y[y][p_idx])
        
        # 转换为 Tensor [C, T, H, W]
        feat_tensor = torch.from_numpy(np.stack(feat_stack, axis=1)).float()
        target_tensor = torch.from_numpy(np.stack(coarse_stack, axis=1)).float()

        # ============================================================
        # 🎨【核心修改】全局平均先验 (Global Average Prior)
        # ============================================================
        # 1. 直接计算时间轴上每一帧的均值 [1, T, 1, 1]
        # 使用 keepdim=True 方便后续广播，无需使用 expand_as 手动复制内存
        # 这一步代替了原来的 AvgPool -> Nearest Interpolate，实现了“白纸”输入
        global_mean = torch.mean(target_tensor, dim=(2, 3), keepdim=True)
        
        # 2. 自动广播 (Broadcasting) 形成没有任何位置信息的平滑输入
        # 此时 input_flat 的每个像素都等于该年份的均值
        input_flat = global_mean + torch.zeros_like(target_tensor)

        # 3. 归一化
        feat_norm = feat_tensor / self.aux_factors
        feat_norm[0] = feat_tensor[0] / NORM_ROAD
        # 增加 clamp(min=0) 提升 ROCm 环境下的数值安全性
        feat_norm[6] = torch.log1p(feat_tensor[6].clamp(min=0)) / NORM_NTL_LOG
        
        input_norm = torch.log1p(input_flat.clamp(min=0)) / NORM_MAIN_LOG
        target_norm = torch.log1p(target_tensor.clamp(min=0)) / NORM_MAIN_LOG
        
        return feat_norm, input_norm, target_norm