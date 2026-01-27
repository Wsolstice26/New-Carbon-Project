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
        
        # 🔥 核心修复：增加一个维度以匹配 [C, T, H, W] -> [9, 1, 1, 1]
        self.aux_factors = torch.tensor(factors).float().view(9, 1, 1, 1)

        self.samples = []
        for idx in self.indices:
            for i in range(len(self.all_years) - self.window + 1):
                years = list(self.all_years[i : i+self.window])
                self.samples.append({'patch_idx': idx, 'years': years})
        
        # 🔥🔥🔥 优化点 1：初始化文件句柄缓存字典
        self.file_cache = {}

    def __len__(self):
        return len(self.samples)

    # 🔥🔥🔥 优化点 2：定义带缓存的读取函数
    def _load_npy(self, path):
        if path not in self.file_cache:
            # 只有第一次读取时打开文件，之后永久复用这个句柄
            # mmap_mode='r' 表示只建立映射，不全读进内存，省内存
            self.file_cache[path] = np.load(path, mmap_mode='r')
        return self.file_cache[path]

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
                # 🔥🔥🔥 优化点 3：使用 _load_npy 获取句柄
                # 这步操作耗时接近 0，不再频繁打开/关闭文件
                x_all = self._load_npy(x_path)
                y_all = self._load_npy(y_path)
                
                # 🔥🔥🔥 优化点 4：显式拷贝数据到内存
                # 从 mmap 中切片读取，并转为 numpy array
                # 这一步是真正发生 IO 的地方，但因为文件已经打开，速度极快
                x_data = np.array(x_all[p_idx]) 
                y_data = np.array(y_all[p_idx])
                
            except Exception:
                # 遇到坏数据给个全0，防止训练中断
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
        # 1. 通用归一化
        feat_norm = feat_tensor / self.aux_factors
        
        # 2. Band 1 (道路) 单独处理
        feat_norm[1] = feat_tensor[1] / NORM_ROAD
        # 3. Band 6 (夜光) Log 处理
        feat_norm[6] = torch.log1p(feat_tensor[6]) / NORM_NTL_LOG
        
        # --- Main (碳排放) Log 处理 ---
        coarse_norm = torch.log1p(coarse_tensor) / NORM_MAIN_LOG
        
        return feat_norm, coarse_norm, coarse_norm