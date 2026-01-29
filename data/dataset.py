import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json

# ==========================================
# ⚙️ 全局归一化参数
# ==========================================
NORM_ROAD = 11.0 
NORM_NTL_LOG = 6.0   
NORM_MAIN_LOG = 11.0

class DualStreamDataset(Dataset):
    def __init__(self, data_dir, split_config_path, mode='train', time_window=3):
        self.data_dir = data_dir
        self.window = time_window
        
        # 加载索引配置
        with open(split_config_path, 'r') as f:
            config = json.load(f)
        
        if mode == 'train':
            self.indices = config['train_indices']
        elif mode == 'val':
            self.indices = config['val_indices']
        else:
            self.indices = config['test_indices']
            
        self.all_years = range(2014, 2024)
        
        # 辅助流归一化因子 (9个通道)
        factors = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
        self.aux_factors = torch.tensor(factors).float().view(9, 1, 1, 1)

        # 构建样本列表
        self.samples = []
        for idx in self.indices:
            # 滑动窗口：例如 [2014, 2015, 2016], [2015, 2016, 2017] ...
            for i in range(len(self.all_years) - self.window + 1):
                years = list(self.all_years[i : i+self.window])
                self.samples.append({'patch_idx': idx, 'years': years})
        
        # ==========================================
        # 🔥 暴力提速：全量预加载 (RAM Mode)
        # ==========================================
        print(f"🚀 [{mode}] 正在将数据加载到内存 (解决 IO 瓶颈)...")
        self.cache_X = {} 
        self.cache_Y = {} 
        
        try:
            for y in self.all_years:
                x_path = os.path.join(self.data_dir, f"X_{y}.npy")
                y_path = os.path.join(self.data_dir, f"Y_{y}.npy")
                
                if os.path.exists(x_path) and os.path.exists(y_path):
                    # 直接 load 到内存，大幅提升训练速度
                    self.cache_X[y] = np.load(x_path) 
                    self.cache_Y[y] = np.load(y_path)
                else:
                    print(f"⚠️ 缺数据: {y}")
            print(f"✅ [{mode}] 加载完成！当前内存占用较高，但速度最快。")
        except MemoryError:
            print(f"❌ 内存不足！如果不幸爆内存，建议在 __init__ 中改回 mmap_mode='r'")
            raise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        p_idx = item['patch_idx']
        years = item['years']
        
        feat_stack = []
        coarse_stack = []
        
        for y in years:
            # 1. 获取原始数据 (Numpy)
            if y in self.cache_X:
                x_data = self.cache_X[y][p_idx] # Shape: (9, 128, 128)
                y_data = self.cache_Y[y][p_idx] # Shape: (1, 128, 128)
            else:
                # 缺失年份补零
                x_data = np.zeros((9, 128, 128), dtype=np.float32)
                y_data = np.zeros((1, 128, 128), dtype=np.float32)
            
            # ==========================================
            # 🛡️ 【数据防毒面具】强制清洗 NaN 和 Inf
            # ==========================================
            # 在转 Tensor 之前就清洗，效率更高
            if np.isnan(x_data).any() or np.isinf(x_data).any():
                x_data = np.nan_to_num(x_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            if np.isnan(y_data).any() or np.isinf(y_data).any():
                y_data = np.nan_to_num(y_data, nan=0.0, posinf=0.0, neginf=0.0)
            # ==========================================
            
            feat_stack.append(x_data)
            coarse_stack.append(y_data)
        
        # 2. 堆叠时间维度 -> Tensor
        # Result Shape: [Channel, Time, H, W]
        feat_tensor = torch.from_numpy(np.stack(feat_stack, axis=1)).float()
        coarse_tensor = torch.from_numpy(np.stack(coarse_stack, axis=1)).float()
        
        # 3. 再次兜底检查 (防止 stack 过程中产生未知错误，虽然概率极低)
        feat_tensor = torch.nan_to_num(feat_tensor, nan=0.0)
        coarse_tensor = torch.nan_to_num(coarse_tensor, nan=0.0)
        
        # 4. 归一化 (Normalization)
        feat_norm = feat_tensor / self.aux_factors
        # 道路路网归一化
        feat_norm[1] = feat_tensor[1] / NORM_ROAD
        # 夜光遥感对数归一化
        feat_norm[6] = torch.log1p(feat_tensor[6]) / NORM_NTL_LOG
        # 主目标对数归一化
        coarse_norm = torch.log1p(coarse_tensor) / NORM_MAIN_LOG
        
        return feat_norm, coarse_norm, coarse_norm