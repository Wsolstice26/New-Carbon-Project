# -*- coding: utf-8 -*-
"""
train_profile_v2.py
===================
目的：只做“速度画像”，不做完整训练（不跑 val、不做 expm1 指标、不做频繁 save），但能精确定位慢点。

用法示例：
  # 纯 profile（默认）
  python train_profile_v2.py

  # 每 20 step 打一次分段耗时
  PROFILE_EVERY=20 python train_profile_v2.py

  # 最多跑 200 step（默认 200）
  MAX_STEPS=200 python train_profile_v2.py

  # 关闭 constraint（让你对比 constraint 开销）
  SKIP_CONSTRAINT=1 python train_profile_v2.py

  # 保存 profile 到 CSV（每 20 step 记录一次）
  PROFILE_EVERY=20 PROFILE_CSV=1 python train_profile_v2.py

  # 需要恢复（读取 latest.pth / autosave_latest.pth）
  RESUME=1 python train_profile_v2.py
"""

import os
import time
import csv
import glob
from collections import deque

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from data.dataset import DualStreamDataset
from models.network import DSTCarbonFormer
from models.losses import HybridLoss
from config import CONFIG


# -----------------------------
# ROCm/MIOpen: 安全默认（不覆盖 shell）
# -----------------------------
def _env_setdefault(k: str, v: str):
    os.environ.setdefault(k, v)

cache_dir = os.path.expanduser("~/.cache/miopen")
os.makedirs(cache_dir, exist_ok=True)
_env_setdefault("MIOPEN_USER_DB_PATH", cache_dir)
_env_setdefault("MIOPEN_CUSTOM_CACHE_DIR", cache_dir)
_env_setdefault("MIOPEN_LOG_LEVEL", "0")
_env_setdefault("MIOPEN_FIND_MODE", "1")
_env_setdefault("MIOPEN_FIND_ENFORCE", "0")
_env_setdefault("MIOPEN_FORCE_USE_WORKSPACE", "1")
_env_setdefault("MIOPEN_DEBUG_CONV_GEMM", "0")


def _pick_ckpt(save_dir: str):
    p1 = os.path.join(save_dir, "autosave_latest.pth")
    p2 = os.path.join(save_dir, "latest.pth")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    # fallback：epoch_*.pth
    files = glob.glob(os.path.join(save_dir, "epoch_*.pth"))
    if files:
        return max(files, key=os.path.getmtime)
    return None


def main():
    # -----------------------------
    # 运行参数（环境变量可覆盖）
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = (device.type == "cuda")
    amp = os.environ.get("AMP", "1") == "1" and is_cuda

    batch = int(os.environ.get("BATCH", str(CONFIG["batch_size"])))
    workers = int(os.environ.get("WORKERS", str(CONFIG.get("num_workers", 0))))
    dim = int(os.environ.get("DIM", str(CONFIG.get("dim", 48))))
    max_steps = int(os.environ.get("MAX_STEPS", "200"))

    profile_every = int(os.environ.get("PROFILE_EVERY", "20"))  # 0=不打印
    iter_window = int(os.environ.get("ITER_WINDOW", "50"))

    # constraint
    train_scale = int(os.environ.get("CONSTRAINT_SCALE", str(CONFIG.get("train_constraint_scale", 120))))
    skip_constraint = os.environ.get("SKIP_CONSTRAINT", "0") == "1"
    constraint_scale = None if skip_constraint else train_scale

    # resume
    resume = os.environ.get("RESUME", "0") == "1"

    # profile csv
    write_csv = os.environ.get("PROFILE_CSV", "0") == "1"

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    print(f"🔥 device={device} | AMP={amp} | batch={batch} | workers={workers} | dim={dim}")
    print(f"⚙️ constraint_scale={constraint_scale} | PROFILE_EVERY={profile_every} | MAX_STEPS={max_steps}")
    print(
        "🧩 MIOpen env:",
        "LOG_LEVEL=", os.environ.get("MIOPEN_LOG_LEVEL"),
        "FIND_MODE=", os.environ.get("MIOPEN_FIND_MODE"),
        "FIND_ENFORCE=", os.environ.get("MIOPEN_FIND_ENFORCE"),
        "WS=", os.environ.get("MIOPEN_FORCE_USE_WORKSPACE"),
    )

    # -----------------------------
    # DataLoader（尽量接近 train.py，但仍保持 profile 简洁）
    # -----------------------------
    train_ds = DualStreamDataset(CONFIG["data_dir"], CONFIG["split_config"], "train", time_window=CONFIG["time_window"])
    steps_per_epoch = max(1, int(np.ceil(len(train_ds) / batch)))
    print(f"✅ train samples={len(train_ds)} | steps/epoch≈{steps_per_epoch}")

    use_persistent = workers > 0
    train_dl = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
    )

    # -----------------------------
    # Model / Loss / Opt
    # -----------------------------
    model = DSTCarbonFormer(aux_c=9, main_c=1, dim=dim).to(device)
    criterion = HybridLoss(
        consistency_scale=CONFIG["consistency_scale"],
        norm_factor=CONFIG["norm_factor"],
    ).to(device)

    opt = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=CONFIG["lr"], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if is_cuda else None

    # resume（可选）
    if resume:
        ckpt_path = _pick_ckpt(CONFIG["save_dir"])
        if ckpt_path:
            print(f"🔄 resume from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "criterion_state_dict" in ckpt:
                try:
                    criterion.load_state_dict(ckpt["criterion_state_dict"])
                except Exception:
                    print("⚠️ criterion state mismatch, skipped.")
            if "optimizer_state_dict" in ckpt:
                opt.load_state_dict(ckpt["optimizer_state_dict"])
            if scaler is not None and "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
        else:
            print("⚠️ RESUME=1 but no checkpoint found, start fresh.")

    # -----------------------------
    # CUDA Events for accurate timing
    # -----------------------------
    def ev():
        return torch.cuda.Event(enable_timing=True) if is_cuda else None

    ev0 = ev()  # iter start
    ev1 = ev()  # after H2D
    ev2 = ev()  # after model
    ev3 = ev()  # after loss
    ev4 = ev()  # after bwd
    ev5 = ev()  # after step

    # iter history
    it_hist = deque(maxlen=iter_window)

    # dl wait measure (CPU)
    last_iter_end = None

    # profile csv
    csv_path = os.path.join(CONFIG["save_dir"], "profile_log.csv")
    if write_csv and (not os.path.exists(csv_path)):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["step", "dl_wait_ms", "h2d_ms", "model_ms", "loss_ms", "bwd_ms", "step_ms", "iter_ms"]
            )
        print(f"📝 profile csv: {csv_path}")

    # -----------------------------
    # Warmup（避免第一步 MIOpen 搜索/缓存导致误判）
    # -----------------------------
    warmup_steps = int(os.environ.get("WARMUP", "5"))
    print(f"🔥 warmup_steps={warmup_steps}")

    model.train()
    criterion.train()

    pbar = tqdm(train_dl, desc="profile-train", total=steps_per_epoch)

    global_step = 0
    for aux, main, target in pbar:
        if global_step >= max_steps:
            break

        # dl wait (CPU estimate)
        now = time.perf_counter()
        dl_wait_ms = (now - last_iter_end) * 1000.0 if last_iter_end is not None else 0.0

        if is_cuda:
            ev0.record()

        # H2D
        aux = aux.to(device, non_blocking=True)
        main = main.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if is_cuda:
            ev1.record()

        opt.zero_grad(set_to_none=True)

        # forward model
        with torch.amp.autocast("cuda", enabled=amp):
            pred, pred_raw = model(aux, main, constraint_scale=constraint_scale)

        if is_cuda:
            ev2.record()

        # loss forward (单独计时)
        with torch.amp.autocast("cuda", enabled=amp):
            loss = criterion(pred, target, aux, pred_raw=pred_raw)

        if is_cuda:
            ev3.record()

        # backward
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()

        if is_cuda:
            ev4.record()

        # step
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        if is_cuda:
            ev5.record()
            ev5.synchronize()

            # segment times
            h2d_ms = ev0.elapsed_time(ev1)
            model_ms = ev1.elapsed_time(ev2)
            loss_ms = ev2.elapsed_time(ev3)
            bwd_ms = ev3.elapsed_time(ev4)
            step_ms = ev4.elapsed_time(ev5)
            iter_ms = ev0.elapsed_time(ev5)
        else:
            # CPU fallback（不精准，只做兜底）
            h2d_ms = model_ms = loss_ms = bwd_ms = step_ms = 0.0
            iter_ms = 0.0

        # warmup discard
        if global_step >= warmup_steps:
            it_hist.append(iter_ms)

        avg_ms = float(np.mean(it_hist)) if it_hist else iter_ms
        it_s = (1000.0 / avg_ms) if avg_ms > 0 else 0.0

        pbar.set_postfix(L=f"{loss.item():.3f}", dl_ms=f"{dl_wait_ms:.0f}", it_ms=f"{avg_ms:.0f}", it_s=f"{it_s:.2f}")

        # print profile
        if profile_every > 0 and (global_step % profile_every == 0):
            print(
                f"\n[profile] step={global_step:6d} "
                f"dl_wait={dl_wait_ms:.1f}ms | h2d={h2d_ms:.2f}ms | "
                f"model_fwd={model_ms:.2f}ms | loss_fwd={loss_ms:.2f}ms | "
                f"bwd={bwd_ms:.2f}ms | step={step_ms:.2f}ms | iter={iter_ms:.2f}ms"
            )

        # csv log
        if write_csv and profile_every > 0 and (global_step % profile_every == 0):
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [global_step, f"{dl_wait_ms:.3f}", f"{h2d_ms:.3f}", f"{model_ms:.3f}", f"{loss_ms:.3f}",
                     f"{bwd_ms:.3f}", f"{step_ms:.3f}", f"{iter_ms:.3f}"]
                )

        global_step += 1
        last_iter_end = time.perf_counter()

    print("\n✅ done.")
    if it_hist:
        print(f"📌 avg_iter_ms(window={iter_window})={np.mean(it_hist):.2f} | it/s={1000.0/np.mean(it_hist):.3f}")
    else:
        print("📌 not enough steps to compute avg.")


if __name__ == "__main__":
    main()
