import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json

# ==========================================
# 📏 定义归一化参数 (Log 模式)
# ==========================================
# Band 1 (道路): 最大 10.2 -> 除以 11.0
NORM_ROAD = 11.0 

# Band 6 (夜光): ln(281)≈5.6 -> 除以 6.0
NORM_NTL_LOG = 6.0   

# 🔥 Main (碳排放): Log 变换
# max ≈ 34480 -> ln(34480+1) ≈ 10.45
# 我们除以 11.0，把它压缩到 0 ~ 0.95
NORM_MAIN_LOG = 11.0

class DualStreamDataset(Dataset):
    def __init__(self, data_dir, split_config_path, mode='train', time_window=3):
        self.data_dir = data_dir
        self.window = time_window
        
        with open(split_config_path, 'r') as f:
            config = json.load(f)
        
        if mode == 'train':
            self.indices = config['train_indices']
        elif mode == 'val':
            self.indices = config['val_indices']
        else:
            self.indices = config['test_indices']
            
        self.all_years = range(2014, 2024)
        
        # 将 Aux 归一化参数转为 Tensor (除 Band 1,6 外保持 1.0)
        # Band 0~8
        factors = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
        
        # 🔥🔥🔥 核心修复：增加一个维度以匹配 [C, T, H, W]
        # 变成 [9, 1, 1, 1]
        self.aux_factors = torch.tensor(factors).float().view(9, 1, 1, 1)

        self.samples = []
        for idx in self.indices:
            for i in range(len(self.all_years) - self.window + 1):
                years = list(self.all_years[i : i+self.window])
                self.samples.append({'patch_idx': idx, 'years': years})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        p_idx = item['patch_idx']
        years = item['years']
        
        feat_stack = []
        coarse_stack = []
        
        for y in years:
            x_path = os.path.join(self.data_dir, f"X_{y}.npy")
            y_path = os.path.join(self.data_dir, f"Y_{y}.npy")
            try:
                x_data = np.load(x_path, mmap_mode='r')[p_idx]
                y_data = np.load(y_path, mmap_mode='r')[p_idx]
            except Exception:
                x_data = np.zeros((9, 128, 128), dtype=np.float32)
                y_data = np.zeros((1, 128, 128), dtype=np.float32)
            feat_stack.append(x_data)
            coarse_stack.append(y_data)
        
        # 堆叠后形状: [9, 3, 128, 128]
        feat_tensor = torch.from_numpy(np.stack(feat_stack, axis=1)).float()
        coarse_tensor = torch.from_numpy(np.stack(coarse_stack, axis=1)).float()
        
        # 清洗 NaN
        feat_tensor = torch.nan_to_num(feat_tensor, nan=0.0)
        coarse_tensor = torch.nan_to_num(coarse_tensor, nan=0.0)
        
        # --- Aux 处理 ---
        # 1. 通用归一化 (大部分是除以1)
        # 此时 self.aux_factors 是 [9, 1, 1, 1]，可以完美广播到 [9, 3, 128, 128]
        feat_norm = feat_tensor / self.aux_factors
        
        # 2. Band 1 (道路) 单独处理
        feat_norm[1] = feat_tensor[1] / NORM_ROAD
        # 3. Band 6 (夜光) Log 处理
        feat_norm[6] = torch.log1p(feat_tensor[6]) / NORM_NTL_LOG
        
        # --- 🔥 Main (碳排放) Log 处理 ---
        # 核心修改：使用 log1p 压缩动态范围，避免小数值丢失
        coarse_norm = torch.log1p(coarse_tensor) / NORM_MAIN_LOG
        
        return feat_norm, coarse_norm, coarse_norm