import os
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

# ==========================================
# ⚙️ 配置区域 (160x160 + 坐标增强版)
# ==========================================
INPUT_DIR = "/home/wdc/Carbon-Emission-Super-Resolution/data/Raw_TIF_Input"
OUTPUT_DATA_DIR = "/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_120"
PATCH_SIZE = 120    
STRIDE = 80         
VALID_THRESHOLD = 0.001 

# 你的TIF情况: [0-6]是特征, [7]是标签
# 我们要生成 [8, 9] 作为坐标
TARGET_BAND_INDEX = 7 

def process_and_slice():
    print(f"🚀 [坐标增强版] 开始处理 (Feature+Coord=9波段)...")
    if not os.path.exists(OUTPUT_DATA_DIR): os.makedirs(OUTPUT_DATA_DIR)
    
    tif_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".tif")])
    
    # ---------------------------------------------------------
    # 阶段 1: 扫描全局有效坐标 (保持不变)
    # ---------------------------------------------------------
    print("\n🔍 阶段 1: 扫描全局有效坐标...")
    global_valid_coords = set()
    first_tif = os.path.join(INPUT_DIR, tif_files[0])
    
    # 读取第一张图来确定 H, W，并预生成坐标网格
    with rasterio.open(first_tif) as src:
        H, W = src.height, src.width
        
    print(f"ℹ️ 地图尺寸: H={H}, W={W}")
    rows = range(0, H - PATCH_SIZE + 1, STRIDE)
    cols = range(0, W - PATCH_SIZE + 1, STRIDE)

    # 扫描
    for f in tif_files:
        path = os.path.join(INPUT_DIR, f)
        with rasterio.open(path) as src:
            img = src.read()
            img = np.nan_to_num(img, nan=0.0)
            for r in rows:
                for c in cols:
                    # 检查 Target(7) 或 Road(0)
                    if np.max(img[TARGET_BAND_INDEX, r:r+PATCH_SIZE, c:c+PATCH_SIZE]) > VALID_THRESHOLD or \
                       np.max(img[0, r:r+PATCH_SIZE, c:c+PATCH_SIZE]) > VALID_THRESHOLD:
                        global_valid_coords.add((r, c))
    
    sorted_coords = sorted(list(global_valid_coords))
    print(f"✅ 有效位置: {len(sorted_coords)}")

    # ---------------------------------------------------------
    # 阶段 2: 生成坐标并切片
    # ---------------------------------------------------------
    print("\n✂️  阶段 2: 生成坐标波段 + 切片保存...")
    
    # 🔥 [核心逻辑] 预生成全局坐标网格 (0~1归一化)
    # Y轴坐标 (0 at top, 1 at bottom)
    y_grid = np.linspace(0, 1, H).astype(np.float32)
    y_map = np.tile(y_grid[:, None], (1, W)) # (H, W)
    
    # X轴坐标 (0 at left, 1 at right)
    x_grid = np.linspace(0, 1, W).astype(np.float32)
    x_map = np.tile(x_grid[None, :], (H, 1)) # (H, W)
    
    # 扩展维度以便拼接: (H, W) -> (1, H, W)
    coord_channels = np.stack([y_map, x_map], axis=0) # Shape: (2, H, W)
    print(f"🌐 全局坐标网格已生成: {coord_channels.shape}")

    metadata_list = []

    for f in tif_files:
        year_match = re.search(r'(\d{4})', f)
        year = int(year_match.group(1)) if year_match else 0
        
        with rasterio.open(os.path.join(INPUT_DIR, f)) as src:
            img = src.read() # Shape: (8, H, W)
            img = np.nan_to_num(img, nan=0.0)
            transform = src.transform
        
        # 🔥 拼接: 原始8波段 + 2坐标波段 = 10波段
        # img[:7] (7特征) + coord (2特征) + img[7] (1标签)
        # 但为了方便，我们把 Feature 拼在一起
        
        # 1. 提取原始特征 (7层)
        raw_feats = img[:TARGET_BAND_INDEX] # (7, H, W)
        
        # 2. 提取标签 (1层)
        target_map = img[TARGET_BAND_INDEX:TARGET_BAND_INDEX+1] # (1, H, W)
        
        # 3. 组合新的 Aux (7+2=9层)
        # 顺序: [Feat 0-6, Global_Y, Global_X]
        combined_aux_map = np.concatenate([raw_feats, coord_channels], axis=0) # (9, H, W)
        
        patches_x = []
        patches_y = []
        
        for idx, (r, c) in enumerate(tqdm(sorted_coords, desc=f"   Processing {year}")):
            # 切片 Aux (9层)
            p_x = combined_aux_map[:, r:r+PATCH_SIZE, c:c+PATCH_SIZE]
            
            # 切片 Target (1层)
            p_y = target_map[:, r:r+PATCH_SIZE, c:c+PATCH_SIZE]
            
            patches_x.append(p_x)
            patches_y.append(p_y)
            
            if idx == 0 and year == 2014:
                lon, lat = transform * (c + PATCH_SIZE//2, r + PATCH_SIZE//2)
                metadata_list.append({'patch_index': idx, 'lon': lon, 'lat': lat})

        # 保存
        np.save(os.path.join(OUTPUT_DATA_DIR, f"X_{year}.npy"), np.array(patches_x))
        np.save(os.path.join(OUTPUT_DATA_DIR, f"Y_{year}.npy"), np.array(patches_y))
        
        if year == 2014:
             print(f"   🔎 最终保存形状 X: {np.array(patches_x).shape} (应为 N,9,160,160)")

    print(f"\n🎉 坐标波段已加入！现在 Aux 有 9 个通道了。")

if __name__ == "__main__":
    process_and_slice()