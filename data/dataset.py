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
        # 🔥 [修正] 强制指向实际有数据的文件夹
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
        
        # 辅助流归一化因子 (7个通道)
        self.aux_factors = torch.tensor([1.0]*9).float().view(9, 1, 1, 1)

        # 2. 构建样本索引列表
        self.samples = []
        for idx in self.indices:
            for i in range(len(self.all_years) - self.window + 1):
                years = list(self.all_years[i : i+self.window])
                self.samples.append({'patch_idx': idx, 'years': years})
        
        # 3. 预加载数据到内存
        print(f"🚀 [{mode}] 正在加载切片数据 (Path: {self.data_dir})...")
        self.cache_X = {} 
        self.cache_Y = {} 
        
        for y in self.all_years:
            x_path = os.path.join(self.data_dir, f"X_{y}.npy")
            y_path = os.path.join(self.data_dir, f"Y_{y}.npy")
            
            if os.path.exists(x_path):
                self.cache_X[y] = np.load(x_path) 
                self.cache_Y[y] = np.load(y_path)
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
            # 读取数据 (实际尺寸是 160x160)
            x_data = self.cache_X[y][p_idx] # (7, 160, 160)
            y_data = self.cache_Y[y][p_idx] # (1, 160, 160)
            
            feat_stack.append(x_data)
            coarse_stack.append(y_data)
        
        # 转 Tensor
        feat_tensor = torch.from_numpy(np.stack(feat_stack, axis=1)).float()
        target_tensor = torch.from_numpy(np.stack(coarse_stack, axis=1)).float()

        # ============================================================
        # 🔥 [降质逻辑] 1km -> 4km -> 1km
        # ============================================================
        # 160 / 4 = 40 (4km grid)
        scale = 4  
        
        # A. 聚合 (AvgPool)
        down_avg = F.avg_pool3d(target_tensor.unsqueeze(0), 
                                kernel_size=(1, scale, scale), 
                                stride=(1, scale, scale)).squeeze(0)
        
        # B. 回填 (Nearest)
        input_mosaic = F.interpolate(down_avg.unsqueeze(0), 
                                     scale_factor=(1, scale, scale), 
                                     mode='nearest').squeeze(0)

        # 归一化
        feat_norm = feat_tensor / self.aux_factors
        # 0通道(Road) / 11, 6通道(NTL) Log/6
        feat_norm[0] = feat_tensor[0] / NORM_ROAD
        feat_norm[6] = torch.log1p(feat_tensor[6]) / NORM_NTL_LOG
        
        input_norm = torch.log1p(input_mosaic) / NORM_MAIN_LOG
        target_norm = torch.log1p(target_tensor) / NORM_MAIN_LOG
        
        return feat_norm, input_norm, target_norm