import os
import glob
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
AUX_DIR = "/home/wdc/Carbon-Emission-Super-Resolution/data/Raw_TIF_Input"
GT_DIR = "/home/wdc/Carbon-Emission-Super-Resolution/data/GT_Aligned_1km"
OUTPUT_DIR = "/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_120"
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = range(2014, 2024)
SCALE_FACTOR = 10           # 1km / 100m
AREA_RATIO = SCALE_FACTOR ** 2 
PATCH_SIZE_1KM = 12         
PATCH_SIZE_100M = 120       
STRIDE_1KM = 8              # 800m 步长

# 🔥【新逻辑】波段定义
AUX_BANDS = list(range(7))  # 只取前7个波段 (0-6)

# ================= 🛠️ 辅助函数：生成坐标通道 =================
def make_coord_channels(H, W):
    """
    生成归一化的坐标网格 [2, H, W]
    Channel 0: Y 坐标 (0~1)
    Channel 1: X 坐标 (0~1)
    """
    y_grid = np.linspace(0, 1, H, dtype=np.float32)
    x_grid = np.linspace(0, 1, W, dtype=np.float32)
    
    # 广播成矩阵
    y_map = np.tile(y_grid[:, None], (1, W))
    x_map = np.tile(x_grid[None, :], (H, 1))
    
    return np.stack([y_map, x_map], axis=0)

# ================= 🚀 第一阶段：求【并集】坐标 =================
# (保持不变：逻辑最稳，涵盖所有有排放的区域)
def get_union_coordinates():
    print("🌍 [Phase 1] 正在遍历所有年份，寻找所有出现过排放的坐标...")
    valid_coords_set = set()
    
    for year in YEARS:
        gt_files = glob.glob(os.path.join(GT_DIR, f"*{year}*.tif"))
        if not gt_files: continue
        gt_path = gt_files[0]

        with rasterio.open(gt_path) as src:
            H, W = src.height, src.width
            data = src.read(1)
            data = np.nan_to_num(data, nan=0.0)
            data[data < 0] = 0.0
            
            for r in range(0, H - PATCH_SIZE_1KM + 1, STRIDE_1KM):
                for c in range(0, W - PATCH_SIZE_1KM + 1, STRIDE_1KM):
                    patch = data[r : r+PATCH_SIZE_1KM, c : c+PATCH_SIZE_1KM]
                    if patch.sum() > 0: # 只要有排放就保留
                        valid_coords_set.add((r, c))
    
    sorted_coords = sorted(list(valid_coords_set))
    print(f"✅ [Phase 1] 完成！全时段共发现 {len(sorted_coords)} 个有效排放区域。")
    return sorted_coords

# ================= 🚀 第二阶段：生成数据 (融合波段修复逻辑) =================

def process_year_with_coords(year, valid_coords):
    # 找文件
    gt_files = glob.glob(os.path.join(GT_DIR, f"*{year}*.tif"))
    if not gt_files: return 0
    gt_path = gt_files[0]

    aux_files = glob.glob(os.path.join(AUX_DIR, f"*{year}*.tif"))
    if not aux_files: aux_files = glob.glob(os.path.join(AUX_DIR, "*MultiBand*.tif"))
    if not aux_files: return 0
    aux_path = aux_files[0]

    print(f"🔄 [Phase 2] 正在生成 {year} 年数据 (波段修复 + 严格对齐)...")

    # 读取全图
    with rasterio.open(gt_path) as src_gt, rasterio.open(aux_path) as src_aux:
        gt_data_full = src_gt.read(1)
        gt_data_full = np.nan_to_num(gt_data_full, nan=0.0)
        gt_data_full[gt_data_full < 0] = 0.0

        # 1. 地理对齐读取 Aux
        gt_bounds = src_gt.bounds
        window = from_bounds(gt_bounds.left, gt_bounds.bottom, gt_bounds.right, gt_bounds.top, transform=src_aux.transform)
        
        # 读取 Aux (此时可能包含多余波段)
        aux_raw = src_aux.read(window=window, boundless=True, fill_value=0) # [C_raw, H_100, W_100]

        # 2. 尺寸强制对齐
        target_h, target_w = gt_data_full.shape[0] * SCALE_FACTOR, gt_data_full.shape[1] * SCALE_FACTOR
        c_raw, h, w = aux_raw.shape
        
        # 创建画布
        aux_aligned = np.zeros((c_raw, target_h, target_w), dtype=np.float32)
        min_h, min_w = min(h, target_h), min(w, target_w)
        aux_aligned[:, :min_h, :min_w] = aux_raw[:, :min_h, :min_w]
        
        # 🔥【关键修改】3. 波段筛选与重组
        # A. 只取前 7 个波段
        aux_feats = aux_aligned[AUX_BANDS, :, :] # [7, H, W]
        
        # B. 生成坐标通道 [2, H, W]
        coord_ch = make_coord_channels(target_h, target_w)
        
        # C. 拼接 -> [9, H, W]
        aux_final_map = np.concatenate([aux_feats, coord_ch], axis=0)

    # 4. 严格切片
    X_list, Y_list = [], []
    
    for (r, c) in valid_coords:
        # 1km GT 切片
        gt_patch = gt_data_full[r : r+PATCH_SIZE_1KM, c : c+PATCH_SIZE_1KM]
        
        # 100m Aux 切片 (已经有了 9 个通道)
        r_100, c_100 = r * SCALE_FACTOR, c * SCALE_FACTOR
        aux_patch = aux_final_map[:, r_100 : r_100 + PATCH_SIZE_100M, c_100 : c_100 + PATCH_SIZE_100M]

        # 边缘保护
        if aux_patch.shape[1] != PATCH_SIZE_100M or aux_patch.shape[2] != PATCH_SIZE_100M:
            padded = np.zeros((9, PATCH_SIZE_100M, PATCH_SIZE_100M), dtype=np.float32)
            ph, pw = aux_patch.shape[1], aux_patch.shape[2]
            padded[:, :ph, :pw] = aux_patch
            aux_patch = padded

        # 物理修正
        gt_expanded = gt_patch.repeat(SCALE_FACTOR, axis=0).repeat(SCALE_FACTOR, axis=1)
        gt_expanded = gt_expanded / float(AREA_RATIO)
        gt_expanded = gt_expanded[np.newaxis, :, :]

        X_list.append(aux_patch)
        Y_list.append(gt_expanded)

    # 保存
    if len(X_list) > 0:
        np.save(os.path.join(OUTPUT_DIR, f"X_{year}.npy"), np.stack(X_list))
        np.save(os.path.join(OUTPUT_DIR, f"Y_{year}.npy"), np.stack(Y_list))
        print(f"   ✅ {year}: {len(X_list)} 个样本 (Shape: {np.stack(X_list).shape})")
    
    return len(X_list)

def main():
    coords = get_union_coordinates()
    for year in YEARS:
        process_year_with_coords(year, coords)
    print(f"🎉 全部完成! 数据格式已修复为 9 通道 (7 Feature + 2 Coord)。")

if __name__ == "__main__":
    main()