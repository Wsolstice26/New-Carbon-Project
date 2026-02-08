# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import rasterio
from tqdm import tqdm
import pandas as pd

# ==========================================
# CONFIG
# ==========================================
INPUT_DIR = "/home/wdc/Carbon-Emission-Super-Resolution/data/Raw_TIF_Input"
OUT_DIR   = "/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_120"

PATCH_SIZE = 120
STRIDE     = 80

# MultiBand: first 7 bands are aux features (0..6)
AUX_BANDS = list(range(7))     # 0-6
ROAD_BAND = 0                 # used for valid check
NTL_BAND  = 6                 # used for valid check

# valid criteria (no longer depends on label band)
VALID_THRESHOLD = 0.001

def list_year_tifs(folder):
    tifs = sorted([f for f in os.listdir(folder) if f.lower().endswith(".tif")])
    if not tifs:
        raise FileNotFoundError(f"No .tif found in {folder}")
    # map year->path
    mp = {}
    for fn in tifs:
        m = re.search(r"(\d{4})", fn)
        if m:
            y = int(m.group(1))
            mp[y] = os.path.join(folder, fn)
    if not mp:
        raise FileNotFoundError("No year parsed from tif names. Ensure filenames contain YYYY.")
    return dict(sorted(mp.items()))

def build_global_coords(year2path, patch_size, stride, thr):
    years = list(year2path.keys())
    first = year2path[years[0]]

    with rasterio.open(first) as src:
        H, W = src.height, src.width

    rows = range(0, H - patch_size + 1, stride)
    cols = range(0, W - patch_size + 1, stride)

    global_valid = set()

    print(f"ℹ️ Grid size (from first tif): H={H}, W={W}")
    print("🔍 Scanning global valid coords using ROAD/NTL only...")

    for y in years:
        path = year2path[y]
        with rasterio.open(path) as src:
            img = src.read()  # [bands, H, W]
        img = np.nan_to_num(img, nan=0.0)

        road = img[ROAD_BAND]
        ntl  = img[NTL_BAND]

        for r in rows:
            for c in cols:
                if (road[r:r+patch_size, c:c+patch_size].max() > thr) or \
                   (ntl[r:r+patch_size, c:c+patch_size].max() > thr):
                    global_valid.add((r, c))

    coords = sorted(global_valid)
    print(f"✅ Global valid coords: {len(coords)}")
    return coords, H, W

def make_coord_channels(H, W):
    # normalized coords in [0,1]
    y_grid = np.linspace(0, 1, H, dtype=np.float32)
    x_grid = np.linspace(0, 1, W, dtype=np.float32)
    y_map = np.tile(y_grid[:, None], (1, W))
    x_map = np.tile(x_grid[None, :], (H, 1))
    return np.stack([y_map, x_map], axis=0)  # [2,H,W]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    year2path = list_year_tifs(INPUT_DIR)
    years = list(year2path.keys())
    print("Years:", years)

    # 1) global coords (same patch positions across all years)
    coords, H, W = build_global_coords(year2path, PATCH_SIZE, STRIDE, VALID_THRESHOLD)

    # 2) global coord channels
    coord_ch = make_coord_channels(H, W)  # [2,H,W]
    print(f"🌐 Coord channels: {coord_ch.shape}")

    # 3) metadata (shared across years)
    # Store patch_index -> (r,c) and derived 1km indices (for aligned 1km GT)
    # Assumes 100m grid, so 10x10 = 1km
    meta_rows = []
    for idx, (r, c) in enumerate(coords):
        meta_rows.append({
            "patch_index": idx,
            "r0": r,
            "c0": c,
            "r1": r + PATCH_SIZE,
            "c1": c + PATCH_SIZE,
            "i1km": r // 10,
            "j1km": c // 10,
            "h1km": PATCH_SIZE // 10,   # 12
            "w1km": PATCH_SIZE // 10,   # 12
        })
    meta_df = pd.DataFrame(meta_rows)
    meta_csv = os.path.join(OUT_DIR, "patch_meta_120_stride80.csv")
    meta_df.to_csv(meta_csv, index=False)
    print("✅ Saved patch meta:", meta_csv)

    # 4) per-year X_YYYY.npy generation
    for y in years:
        path = year2path[y]
        with rasterio.open(path) as src:
            # basic consistency checks
            if src.height != H or src.width != W:
                raise ValueError(f"{path} size mismatch: got {(src.height, src.width)} != {(H,W)}")
            img = src.read()  # [bands, H, W]
            transform = src.transform
            crs = src.crs

        img = np.nan_to_num(img, nan=0.0).astype(np.float32)

        # aux features: 7 bands
        raw_feats = img[AUX_BANDS, :, :]  # [7,H,W]

        # combined aux: 7 + 2 coords = 9
        aux_map = np.concatenate([raw_feats, coord_ch], axis=0)  # [9,H,W]

        patches = []
        for (r, c) in tqdm(coords, desc=f"Cut X_{y}"):
            p = aux_map[:, r:r+PATCH_SIZE, c:c+PATCH_SIZE]  # [9,120,120]
            patches.append(p)

        X = np.stack(patches, axis=0).astype(np.float32)  # [N,9,120,120]
        out_x = os.path.join(OUT_DIR, f"X_{y}.npy")
        np.save(out_x, X)

        print(f"✅ Saved {out_x} | shape={X.shape} | CRS={crs} | pixel_size=({transform.a},{transform.e})")

    print("\n🎉 Done. Note: Y_YYYY.npy is no longer generated here (GT comes from aligned 1km ODIAC).")

if __name__ == "__main__":
    main()
