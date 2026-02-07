# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# ============================================================
# ⚡️ 极速优化版 Dataset (Train/Val 统一 Uniform 策略)
# ============================================================

NORM_ROAD = 11.0       
NORM_NTL_LOG = 6.0     
CO2_NORM_FACTOR = 1000.0 

_YEAR_STATS_CACHE = {}

def _compute_year_emission_stats_linear(y_linear_np: np.ndarray):
    # 确保输入没有 NaN，虽然在外层已经处理过，这里做二次保险
    arr = np.nan_to_num(y_linear_np.astype(np.float64, copy=False), nan=0.0)
    arr[arr < 0] = 0.0
    mask_nz = arr > 0
    nnz = int(np.count_nonzero(mask_nz))
    total = int(arr.size)
    nz = total - nnz
    if nnz <= 0: return float(1e6), float(0.0)
    logv = np.log1p(arr[mask_nz])
    cv_log = float(logv.std()) / (float(logv.mean()) + 1e-12)
    nz_ratio = float(nz / (nnz + 1e-12))
    return nz_ratio, float(cv_log)

class DualStreamDataset(Dataset):
    def __init__(self, data_dir, split_config_path, mode="train", time_window=3):
        # 建议动态使用传入的 data_dir，或者保持你 hardcode 的路径
        self.data_dir = "/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_120"
        self.window = int(time_window)
        self.mode = mode

        with open(split_config_path, "r") as f:
            config = json.load(f)

        if mode == "train":
            self.indices = config["train_indices"]
        elif mode == "val":
            self.indices = config["val_indices"]
        else:
            self.indices = config["test_indices"]

        self.train_indices_for_stats = config["train_indices"]
        self.all_years = list(range(2014, 2024))
        self.aux_factors = torch.tensor([1.0] * 9).float().view(9, 1, 1, 1)

        print(f"🚀 [{mode}] 初始化索引...")
        self.samples = []
        for idx in self.indices:
            for i in range(len(self.all_years) - self.window + 1):
                years = self.all_years[i: i + self.window]
                self.samples.append({"patch_idx": idx, "years": years})

        print(f"⏳ [{mode}] 正在加载并预计算 (自动清洗 NaN -> 0)...")
        
        self.cache_X = {}          
        self.cache_gt_1km = {}     
        self.cache_main_init = {}  

        scale_factor = 10     
        area_ratio = float(scale_factor * scale_factor) # 100.0
        temp_stats_Y = {}

        for y in self.all_years:
            x_path = os.path.join(self.data_dir, f"X_{y}.npy")
            y_path = os.path.join(self.data_dir, f"Y_{y}.npy")
            
            if os.path.exists(x_path) and os.path.exists(y_path):
                # 🚀 [核心修改] 加载 X 并在内存中将 NaN 替换为 0
                raw_x = np.load(x_path)
                self.cache_X[y] = np.nan_to_num(raw_x, nan=0.0)
                
                # 🚀 [核心修改] 加载 Y 并在内存中将 NaN 替换为 0
                raw_y = np.load(y_path)
                y_data = np.nan_to_num(raw_y, nan=0.0)
                temp_stats_Y[y] = y_data

                with torch.no_grad():
                    y_tensor = torch.from_numpy(y_data).float()
                    # GT 1km
                    gt_1km = (F.avg_pool2d(y_tensor, kernel_size=scale_factor, stride=scale_factor) * area_ratio) / CO2_NORM_FACTOR
                    self.cache_gt_1km[y] = gt_1km.numpy()

                    # ====================================================
                    # 🚀 [策略] 统一策略: Train 和 Val 都使用 Uniform Main
                    # ====================================================
                    patch_mean = gt_1km.mean(dim=(-2, -1), keepdim=True)
                    main_init = F.interpolate(patch_mean, size=(120, 120), mode="nearest") / area_ratio

                    self.cache_main_init[y] = main_init.numpy()
        
        print(f"✅ [{mode}] 预处理完成 (已处理 NaN，已缩放 /{CO2_NORM_FACTOR})")
        print("🔥 [Strategy] Main Init 已设为全图均值 (Uniform)，强迫模型学习 Aux 结构。")

        cache_key = (self.data_dir, split_config_path, "train_stats_only", "scheme2_linear")
        if cache_key in _YEAR_STATS_CACHE:
            self.year_stats = _YEAR_STATS_CACHE[cache_key]
        else:
            print("📊 [stats] 计算统计量...")
            year_stats = {}
            train_idx = np.array(self.train_indices_for_stats, dtype=np.int64)
            for y in self.all_years:
                if y in temp_stats_Y:
                    raw_Y_train = temp_stats_Y[y][train_idx]
                    with torch.no_grad():
                        t_raw = torch.from_numpy(raw_Y_train).float()
                        t_gt_1km = (F.avg_pool2d(t_raw, kernel_size=scale_factor, stride=scale_factor) * area_ratio) / CO2_NORM_FACTOR
                        t_100m = F.interpolate(t_gt_1km, size=(120, 120), mode="nearest") / area_ratio
                        nz_ratio, cv_log = _compute_year_emission_stats_linear(t_100m.numpy())
                        year_stats[y] = (nz_ratio, cv_log)
            _YEAR_STATS_CACHE[cache_key] = year_stats
            self.year_stats = year_stats
        
        del temp_stats_Y

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        p_idx = item["patch_idx"]
        years = item["years"]

        feat = np.stack([self.cache_X[y][p_idx] for y in years], axis=1)          
        main_init = np.stack([self.cache_main_init[y][p_idx] for y in years], axis=1) 
        gt_1km = np.stack([self.cache_gt_1km[y][p_idx] for y in years], axis=1)       

        feat_t = torch.from_numpy(feat).float()
        main_init_t = torch.from_numpy(main_init).float()
        gt_1km_t = torch.from_numpy(gt_1km).float()

        feat_t = feat_t / self.aux_factors
        feat_t[0] = feat_t[0] / NORM_ROAD
        feat_t[6] = torch.log1p(feat_t[6].clamp(min=0)) / NORM_NTL_LOG

        nz_ratios = [self.year_stats[int(y)][0] for y in years]
        cv_logs = [self.year_stats[int(y)][1] for y in years]

        return feat_t, main_init_t, gt_1km_t, torch.tensor(np.mean(nz_ratios), dtype=torch.float32), torch.tensor(np.mean(cv_logs), dtype=torch.float32)