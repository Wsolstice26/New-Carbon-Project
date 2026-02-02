# -*- coding: utf-8 -*-
import os
import warnings
import csv
import sys
import glob
from collections import deque

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# ==========================================
# 🔇 [日志静音 & 环境优化]
# ==========================================
warnings.filterwarnings("ignore", message=".*Dynamo does not know how to trace the builtin.*")
warnings.filterwarnings("ignore", message=".*Unable to hit fast path of CUDAGraphs.*")
warnings.filterwarnings("ignore", message=".*TensorFloat32 tensor cores.*")

cache_dir = os.path.expanduser("~/.cache/miopen")
os.makedirs(cache_dir, exist_ok=True)

os.environ.setdefault("MIOPEN_USER_DB_PATH", cache_dir)
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", cache_dir)
os.environ.setdefault("MIOPEN_LOG_LEVEL", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "1")
os.environ.setdefault("MIOPEN_FORCE_USE_WORKSPACE", "1")

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

from data.dataset import DualStreamDataset
from models.network import DSTCarbonFormer
from models.losses import HybridLoss
from config import CONFIG

# ==========================================
# 🛠️ Checkpoint 工具函数
# ==========================================

def get_latest_checkpoint(save_dir):
    latest = os.path.join(save_dir, "autosave_latest.pth")
    if os.path.exists(latest):
        return latest
    return None


def save_checkpoint(path, epoch, step, model, criterion, optimizer, scheduler, scaler, best_nonzero_mae):
    torch.save(
        {
            "epoch": epoch,
            "global_step": step,
            "model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "best_nonzero_mae": best_nonzero_mae,
        },
        path,
    )


# ==========================================
# 📊 Validation Metrics (GPU only)
# ==========================================
def calc_metrics_tensor(
    pred_real: torch.Tensor,
    target_real: torch.Tensor,
    thr: float = 1e-6,
    top_p_list=(0.01, 0.05),
):
    diff = torch.abs(pred_real - target_real)

    global_mae = diff.mean()

    mask_nz = target_real > thr
    mask_z = ~mask_nz

    nz_cnt = mask_nz.sum().clamp(min=1)
    z_cnt = mask_z.sum().clamp(min=1)

    nz_mae = diff[mask_nz].sum() / nz_cnt
    z_mae = diff[mask_z].sum() / z_cnt

    balanced_mae = 0.5 * nz_mae + 0.5 * z_mae

    metrics = [global_mae, nz_mae, balanced_mae]

    nz_target = target_real[mask_nz]

    if nz_target.numel() < 10:
        for _ in top_p_list:
            metrics.append(nz_mae)
    else:
        for p in top_p_list:
            q = torch.quantile(nz_target, 1.0 - p)
            mask_top = target_real >= q
            top_mae = diff[mask_top].sum() / mask_top.sum().clamp(min=1)
            metrics.append(top_mae)

    return torch.stack(metrics)


# ==========================================
# 🏃 主训练逻辑
# ==========================================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"

    # -------------------------
    # Config shortcuts
    # -------------------------
    SAVE_EVERY_STEPS = CONFIG["save_every_steps"]
    KEEP_LAST_STEPS = CONFIG["keep_last_steps"]
    SAVE_EVERY_EPOCHS = CONFIG["save_every_epochs"]

    TRAIN_SCALE = CONFIG.get("train_constraint_scale", 120)

    print(f"🔥 Device: {device}")

    # -------------------------
    # Reproducibility
    # -------------------------
    if CONFIG.get("deterministic", False):
        torch.manual_seed(CONFIG["seed"])
        np.random.seed(CONFIG["seed"])
        if is_cuda:
            torch.cuda.manual_seed_all(CONFIG["seed"])

    scaler = torch.amp.GradScaler("cuda", init_scale=65535.0) if is_cuda else None
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # ==========================
    # Dataset
    # ==========================
    train_ds = DualStreamDataset(
        CONFIG["data_dir"], CONFIG["split_config"], "train",
        time_window=CONFIG["time_window"]
    )
    val_ds = DualStreamDataset(
        CONFIG["data_dir"], CONFIG["split_config"], "val",
        time_window=CONFIG["time_window"]
    )

    global_nz_ratio = train_ds.global_nz_ratio
    global_cv_log = train_ds.global_cv_log

    print(
        f"📊 [Global Stats] Nz/Nnz={global_nz_ratio:.2f}, "
        f"CV_log={global_cv_log:.3f}"
    )

    loader_args = dict(
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=CONFIG["num_workers"] > 0,
    )

    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_args)
    val_dl = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_args)

    # ==========================
    # Model & Loss
    # ==========================
    model = DSTCarbonFormer(aux_c=9, main_c=1, dim=CONFIG["dim"]).to(device)

    criterion = HybridLoss(
        consistency_scale=CONFIG["consistency_scale"],
        norm_factor=CONFIG["norm_factor"],
        global_nz_ratio=global_nz_ratio,
        global_cv_log=global_cv_log,
    ).to(device)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=CONFIG["lr"],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=1e-6
    )

    # ==========================
    # Resume
    # ==========================
    start_epoch = 1
    global_step = 0
    best_nonzero_mae = float("inf")

    if CONFIG.get("resume", False):
        ckpt_path = get_latest_checkpoint(CONFIG["save_dir"])
        if ckpt_path:
            print(f"🔁 Resuming from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)

            model.load_state_dict(ckpt["model_state_dict"])
            criterion.load_state_dict(ckpt["criterion_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

            if scaler and ckpt.get("scaler_state_dict") is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])

            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt.get("global_step", 0)
            best_nonzero_mae = ckpt.get("best_nonzero_mae", best_nonzero_mae)

            print(
                f"✅ Resume OK | start_epoch={start_epoch}, "
                f"best_nonzero_mae={best_nonzero_mae:.4f}"
            )

    # ==========================
    # Training Loop
    # ==========================
    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_dl, desc=f"Ep {epoch}/{CONFIG['epochs']}")

        for aux, main, target in loop:
            aux = aux.to(device, non_blocking=True)
            main = main.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=is_cuda):
                pred, pred_raw = model(aux, main, constraint_scale=TRAIN_SCALE)
                loss = criterion(pred, target, aux, pred_raw)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            global_step += 1
            loop.set_postfix(L=f"{loss.item():.4f}")

            # -------------------------
            # 🔁 Step-level autosave (轮转 + latest)
            # -------------------------
            if global_step % SAVE_EVERY_STEPS == 0:
                slot = (global_step // SAVE_EVERY_STEPS) % KEEP_LAST_STEPS

                step_ckpt = os.path.join(
                    CONFIG["save_dir"],
                    f"autosave_step_{slot}.pth"
                )
                latest_ckpt = os.path.join(
                    CONFIG["save_dir"],
                    "autosave_latest.pth"
                )

                save_checkpoint(
                    step_ckpt, epoch, global_step,
                    model, criterion, optimizer, scheduler, scaler,
                    best_nonzero_mae
                )
                save_checkpoint(
                    latest_ckpt, epoch, global_step,
                    model, criterion, optimizer, scheduler, scaler,
                    best_nonzero_mae
                )

        # ==========================
        # Validation
        # ==========================
        model.eval()
        val_metrics = torch.zeros(5, device=device)
        cnt = 0

        with torch.no_grad():
            for aux, main, target in val_dl:
                aux = aux.to(device, non_blocking=True)
                main = main.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=is_cuda):
                    pred, _ = model(aux, main, constraint_scale=TRAIN_SCALE)

                pred_real = torch.expm1(pred * CONFIG["norm_factor"]).clamp(min=0)
                target_real = torch.expm1(target * CONFIG["norm_factor"]).clamp(min=0)

                val_metrics += calc_metrics_tensor(pred_real, target_real)
                cnt += 1

        avg_metrics = (val_metrics / cnt).cpu().numpy()
        global_mae, nz_mae, bal_mae, top1p, top5p = avg_metrics

        print(
            f"📊 Val | "
            f"Nonzero={nz_mae:.4f} | "
            f"Top-1%={top1p:.4f} | "
            f"Top-5%={top5p:.4f} | "
            f"Balanced={bal_mae:.4f}"
        )

        # -------------------------
        # 🏆 Best model (Nonzero-MAE)
        # -------------------------
        if nz_mae < best_nonzero_mae:
            best_nonzero_mae = nz_mae
            torch.save(
                model.state_dict(),
                os.path.join(CONFIG["save_dir"], "best_model.pth"),
            )
            print(f"🏆 New Best Model! Nonzero-MAE={best_nonzero_mae:.4f}")

        # -------------------------
        # 🧊 Epoch-level permanent save
        # -------------------------
        if epoch % SAVE_EVERY_EPOCHS == 0:
            epoch_ckpt = os.path.join(
                CONFIG["save_dir"],
                f"epoch_{epoch:03d}.pth"
            )
            save_checkpoint(
                epoch_ckpt, epoch, global_step,
                model, criterion, optimizer, scheduler, scaler,
                best_nonzero_mae
            )
            print(f"🧊 Saved permanent checkpoint: epoch_{epoch:03d}.pth")

        scheduler.step()


if __name__ == "__main__":
    train()

