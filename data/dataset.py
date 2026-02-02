import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# =========================
# Global normalization
# =========================
NORM_ROAD = 11.0
NORM_NTL_LOG = 6.0
NORM_MAIN_LOG = 11.0


# ============================================================
# Global statistics helper (NEW)
# ============================================================
def compute_global_emission_stats_from_arrays(y_arrays):
    """
    Compute:
      - global_nz_ratio = Nz / Nnz
      - global_cv_log   = std(log1p(y)) / mean(log1p(y))
    from a list of numpy arrays [N, 1, H, W] in LINEAR domain.
    """
    total_zero = 0
    total_nonzero = 0
    log_values = []

    for arr in y_arrays:
        arr = arr.astype(np.float64)
        arr[arr < 0] = 0.0

        mask_nz = arr > 0
        nz = np.count_nonzero(mask_nz)
        z = arr.size - nz

        total_nonzero += nz
        total_zero += z

        if nz > 0:
            log_values.append(np.log1p(arr[mask_nz]))

    if total_nonzero == 0:
        raise RuntimeError("No non-zero emission pixels found in training set!")

    log_all = np.concatenate(log_values, axis=0)
    mu = log_all.mean()
    std = log_all.std()

    global_nz_ratio = total_zero / total_nonzero
    global_cv_log = std / (mu + 1e-12)

    return float(global_nz_ratio), float(global_cv_log)


class DualStreamDataset(Dataset):
    """
    Dual-stream dataset for weakly-supervised downscaling.

    Key properties:
    - Global emission statistics (Nz/Nnz, CV_log) are computed ONCE in train mode
      and cached for loss construction.
    - Validation / test datasets do NOT recompute statistics.
    """

    def __init__(self, data_dir, split_config_path, mode="train", time_window=3):
        # Force data folder (keep your current behavior)
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

        self.all_years = list(range(2014, 2024))

        # Aux normalization factors
        self.aux_factors = torch.tensor([1.0] * 9).float().view(9, 1, 1, 1)

        # Build (patch_idx, sliding time window) samples
        print(f"🚀 [{mode}] 正在加载数据...")
        self.samples = []
        for idx in self.indices:
            for i in range(len(self.all_years) - self.window + 1):
                years = self.all_years[i : i + self.window]
                self.samples.append({"patch_idx": idx, "years": years})

        # Preload X/Y into memory by year
        self.cache_X = {}
        self.cache_Y = {}

        for y in self.all_years:
            x_path = os.path.join(self.data_dir, f"X_{y}.npy")
            y_path = os.path.join(self.data_dir, f"Y_{y}.npy")
            if os.path.exists(x_path) and os.path.exists(y_path):
                self.cache_X[y] = np.load(x_path).copy()
                self.cache_Y[y] = np.load(y_path).copy()

        print(f"✅ [{mode}] 加载完成，样本数: {len(self.samples)}")

        # ============================================================
        # 🔥 NEW: Global emission statistics (TRAIN ONLY)
        # ============================================================
        self.global_nz_ratio = None
        self.global_cv_log = None

        if self.mode == "train":
            print("📊 [train] Computing global emission statistics...")

            # Collect ALL training Y in LINEAR domain (100m-unit)
            y_collect = []
            scale_factor = 10
            area_ratio = float(scale_factor * scale_factor)

            for y in self.all_years:
                raw_Y = self.cache_Y[y]  # [N, 1, 120, 120]

                # --- Extract true 1km totals ---
                raw_tensor = torch.from_numpy(raw_Y).float()
                down_1km_total = F.avg_pool3d(
                    raw_tensor,
                    kernel_size=(1, scale_factor, scale_factor),
                    stride=(1, scale_factor, scale_factor),
                )  # [N, 1, 12, 12]

                # --- Map back to 120x120 and convert to 100m-unit ---
                target_100m = F.interpolate(
                    down_1km_total,
                    size=(120, 120),
                    mode="nearest",
                ) / area_ratio

                y_collect.append(target_100m.numpy())

            self.global_nz_ratio, self.global_cv_log = \
                compute_global_emission_stats_from_arrays(y_collect)

            print(
                f"📊 [Global Stats] Nz/Nnz = {self.global_nz_ratio:.2f}, "
                f"CV_log = {self.global_cv_log:.3f}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        p_idx = item["patch_idx"]
        years = item["years"]

        feat_stack = []
        y_stack = []

        for y in years:
            feat_stack.append(self.cache_X[y][p_idx])
            y_stack.append(self.cache_Y[y][p_idx])

        # X: stack by time -> [9, T, 120, 120]
        feat_tensor = torch.from_numpy(np.stack(feat_stack, axis=1)).float()

        # Y: stack by time -> [1, T, 120, 120]
        raw_1km_tensor = torch.from_numpy(np.stack(y_stack, axis=1)).float()

        # ============================================================
        # Supervision definition
        # ============================================================
        scale_factor = 10
        area_ratio = float(scale_factor * scale_factor)

        down_1km_total = F.avg_pool3d(
            raw_1km_tensor,
            kernel_size=(1, scale_factor, scale_factor),
            stride=(1, scale_factor, scale_factor),
        )

        target_tensor = F.interpolate(
            down_1km_total,
            size=(120, 120),
            mode="nearest",
        ) / area_ratio

        patch_mean_1x1 = target_tensor.mean(dim=(-2, -1), keepdim=True)
        input_main = F.interpolate(
            patch_mean_1x1,
            size=(120, 120),
            mode="nearest",
        )

        # ============================================================
        # Normalization (Log1p)
        # ============================================================
        feat_norm = feat_tensor / self.aux_factors
        feat_norm[0] = feat_tensor[0] / NORM_ROAD
        feat_norm[6] = torch.log1p(feat_tensor[6].clamp(min=0)) / NORM_NTL_LOG

        input_norm = torch.log1p(input_main.clamp(min=0)) / NORM_MAIN_LOG
        target_norm = torch.log1p(target_tensor.clamp(min=0)) / NORM_MAIN_LOG

        return feat_norm, input_norm, target_norm
