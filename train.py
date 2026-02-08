# -*- coding: utf-8 -*-
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import warnings
import csv
import glob
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

warnings.filterwarnings("ignore", message=".*Dynamo does not know how to trace the builtin.*")
warnings.filterwarnings("ignore", message=".*Unable to hit fast path of CUDAGraphs.*")
warnings.filterwarnings("ignore", message=".*TensorFloat32 tensor cores.*")

cache_dir = os.path.expanduser("~/.cache/miopen")
os.makedirs(cache_dir, exist_ok=True)

os.environ.setdefault("MIOPEN_USER_DB_PATH", cache_dir)
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", cache_dir)
os.environ.setdefault("MIOPEN_LOG_LEVEL", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "1") 
os.environ.setdefault("MIOPEN_FORCE_INT8", "0")
os.environ.setdefault("MIOPEN_FORCE_USE_WORKSPACE", "1")

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = False 

from data.dataset import DualStreamDataset
from models.network import DSTCarbonFormer
from models.losses import HybridLoss
from config import CONFIG

CO2_NORM_FACTOR = 1000.0

class DynamicScaleSampler:
    def __init__(self, scales=[1, 2, 3, 4, 6], momentum=0.9):
        self.scales = scales
        self.momentum = momentum
        self.avg_losses = {s: 1.0 for s in scales}
        
    def update(self, scale, current_loss):
        if current_loss > 0 and np.isfinite(current_loss):
            old = self.avg_losses[scale]
            self.avg_losses[scale] = self.momentum * old + (1 - self.momentum) * current_loss

    def get_scale(self):
        losses = np.array([self.avg_losses[s] for s in self.scales])
        losses = np.nan_to_num(losses, nan=1.0, posinf=1.0, neginf=1.0)
        total_loss = np.sum(losses)
        if total_loss < 1e-9:
            probs = np.ones_like(losses) / len(self.scales)
        else:
            probs = losses / total_loss
        probs = probs / np.sum(probs)
        return np.random.choice(self.scales, p=probs), probs

def get_latest_checkpoint(save_dir):
    latest = os.path.join(save_dir, "autosave_latest.pth")
    return latest if os.path.exists(latest) else None

def save_checkpoint(path, epoch, step, model, criterion, optimizer, scheduler, scaler, best_r2, patience_counter):
    torch.save({
        "epoch": epoch, 
        "global_step": step, 
        "model_state_dict": model.state_dict(),
        "criterion_state_dict": criterion.state_dict(), 
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(), 
        "scaler_state_dict": scaler.state_dict() if scaler else None, 
        "best_r2": best_r2,
        "patience_counter": patience_counter
    }, path)

def sum_pool3d(x: torch.Tensor, s: int) -> torch.Tensor:
    return F.avg_pool3d(x, kernel_size=(1, s, s), stride=(1, s, s)) * (s * s)

def calc_metrics_1km_tensor(pred_1km, gt_1km, thr=1e-6, top_p_list=(0.01, 0.05)):
    pred_real = pred_1km * CO2_NORM_FACTOR
    gt_real = gt_1km * CO2_NORM_FACTOR
    
    diff = pred_real - gt_real
    abs_diff = torch.abs(diff)
    global_mae = abs_diff.mean()
    mask_nz = gt_real > thr
    nz_mae = abs_diff[mask_nz].sum() / mask_nz.sum().clamp(min=1)
    balanced_mae = 0.5 * nz_mae + 0.5 * (abs_diff[~mask_nz].sum() / (~mask_nz).sum().clamp(min=1))
    metrics = [global_mae, nz_mae, balanced_mae]
    nz_target = gt_real[mask_nz]
    if nz_target.numel() < 10:
        for _ in top_p_list: metrics.append(nz_mae)
    else:
        for p in top_p_list:
            q = torch.quantile(nz_target, 1.0 - p)
            mask_top = gt_real >= q
            top_mae = abs_diff[mask_top].sum() / mask_top.sum().clamp(min=1)
            metrics.append(top_mae)
    mse = (diff ** 2).mean()
    metrics.append(mse)
    
    r2 = 1 - torch.sum(diff ** 2) / (torch.sum((gt_real - torch.mean(gt_real)) ** 2) + 1e-8)
    metrics.append(r2)
    
    p_flat, g_flat = pred_real.view(-1), gt_real.view(-1)
    vx, vy = p_flat - torch.mean(p_flat), g_flat - torch.mean(g_flat)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    metrics.append(corr)
    return torch.stack(metrics)

def train():
    device = torch.device(CONFIG["device"] if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    
    SAVE_EVERY_STEPS = CONFIG["save_every_steps"]
    SAVE_EVERY_EPOCHS = CONFIG["save_every_epochs"]
    CONSIST_SCALE = int(CONFIG.get("consistency_scale", 10))
    PATIENCE = CONFIG.get("patience", 200)
    GRAD_ACCUM_STEPS = CONFIG.get("grad_accum_steps", 1)
    
    if CONFIG.get("deterministic", False):
        torch.manual_seed(CONFIG["seed"])
        np.random.seed(CONFIG["seed"])
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(CONFIG["seed"])

    print(f" [Mode] 混合精度训练 (FP16/AMP) | 已应用稳定性补丁")
    print(f" [Grad Accum] 每 {GRAD_ACCUM_STEPS} 个物理步骤执行一次参数更新")
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda) 
    
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    log_file = os.path.join(CONFIG["save_dir"], "training_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            # 🚀 [修改] 增加 w_L1 和 w_MSE 列
            csv.writer(f).writerow([
                "Epoch", "LR", "Train_Loss", "MAE_Global", "MAE_Nonzero", "MAE_Balanced", 
                "MAE_Top1pct", "MAE_Top5pct", "RMSE_Global", "R2_Score", "Pearson_Corr",
                "Pred_Mean", "Pred_Max", "Pred_NZRatio", "ConsMAE_1km", "w_L1", "w_MSE", "GlobalStep", "Best_R2"
            ])

    train_ds = DualStreamDataset(CONFIG["data_dir"], CONFIG["split_config"], mode="train", time_window=CONFIG["time_window"])
    val_ds = DualStreamDataset(CONFIG["data_dir"], CONFIG["split_config"], mode="val", time_window=CONFIG["time_window"])
    
    loader_args = dict(batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"], pin_memory=True, persistent_workers=CONFIG["num_workers"] > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_args)
    val_dl = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_args)

    norm_const = CONFIG.get("norm_factor", 1.0)
    print(f" Model Norm Const: {norm_const} (Loaded from Config)")
    model = DSTCarbonFormer(
        aux_c=9, 
        main_c=1, 
        dim=CONFIG["dim"], 
        norm_const=norm_const,
        num_mamba_layers=CONFIG.get("num_mamba_layers", 1),
        num_res_blocks=CONFIG.get("num_res_blocks", 2)
    ).to(device)

    # 🚀 [修改] 加入 w_mse 确保启用 MSE Loss
    criterion = HybridLoss(
        consistency_scale=CONSIST_SCALE, 
        w_sparse=CONFIG.get("w_sparse", 1e-3), 
        w_ent=CONFIG.get("w_ent", 1e-3), 
        w_mse=CONFIG.get("w_mse", 1.0),   # 确保 MSE 开启
        ent_mode=CONFIG.get("ent_mode", "max"), 
        target_entropy=CONFIG.get("target_entropy", 1.5), 
        use_charbonnier_A=CONFIG.get("use_charbonnier_A", False)
    ).to(device)

    learning_rate = CONFIG.get("lr", 1e-5)
    print(f" Optimizer LR: {learning_rate}")
    
    # criterion.parameters() 已经在列表里了，这是正确的，确保 log_vars 可以更新
    optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)
    scale_sampler = DynamicScaleSampler(scales=[1, 2, 3, 4, 6])
    
    start_epoch = 1
    global_step = 0
    best_r2 = -float("inf")
    patience_counter = 0

    if CONFIG.get("resume", False):
        ckpt_path = get_latest_checkpoint(CONFIG["save_dir"])
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"]:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            # 尝试加载 criterion 状态 (兼容旧版存档)
            if "criterion_state_dict" in ckpt:
                try:
                    criterion.load_state_dict(ckpt["criterion_state_dict"])
                except Exception as e:
                    print(f"⚠️ Warning: Failed to load criterion state (log_vars reset): {e}")
            
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt.get("global_step", 0)
            best_r2 = ckpt.get("best_r2", -float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            print(f" Resumed from epoch {start_epoch-1}. Current Best R2: {best_r2}")

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_dl, desc=f"Ep {epoch}/{CONFIG['epochs']}")
        last_batch_cache = None
        
        optimizer.zero_grad()

        for batch_idx, (aux, _, gt_1km, nz_ratio_win, cv_log_win) in enumerate(loop):
            aux = aux.to(device, non_blocking=True)
            gt_1km = gt_1km.to(device, non_blocking=True)
            nz_ratio_win = nz_ratio_win.to(device, non_blocking=True)
            cv_log_win = cv_log_win.to(device, non_blocking=True)

            current_scale, current_probs = scale_sampler.get_scale()
            
            main_degraded = F.adaptive_avg_pool3d(gt_1km, output_size=(CONFIG["time_window"], current_scale, current_scale))
            area_ratio = CONSIST_SCALE ** 2
            main_input = main_degraded / area_ratio

            with torch.amp.autocast("cuda", enabled=is_cuda):
                pred_100m, _ = model(aux, main_input)
                
                pred_1km = sum_pool3d(pred_100m, CONSIST_SCALE)
                loss = criterion(
                    pred=pred_1km, 
                    target=gt_1km, 
                    pred_100m=pred_100m, 
                    nz_ratio_win=nz_ratio_win, 
                    cv_log_win=cv_log_win
                )
                loss = loss / GRAD_ACCUM_STEPS

            if not torch.isnan(loss) and not torch.isinf(loss):
                scaler.scale(loss).backward()

                if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                    scaler.unscale_(optimizer) 
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                loss_item = float(loss.detach().item() * GRAD_ACCUM_STEPS)
                train_loss += loss_item
                
                scale_sampler.update(current_scale, loss_item)
                
                # 🚀 [修改] 获取动态权重用于显示
                with torch.no_grad():
                    # log_vars[0] -> L1, log_vars[1] -> MSE
                    w_l1_disp = torch.exp(-criterion.log_vars[0]).item()
                    w_mse_disp = torch.exp(-criterion.log_vars[1]).item()

                mem_reserved = torch.cuda.memory_reserved() / 1024**3 if is_cuda else 0
                # 🚀 [修改] 进度条增加 wL 和 wM
                loop.set_postfix(L=f"{loss_item:.3f}", wL=f"{w_l1_disp:.2f}", wM=f"{w_mse_disp:.2f}", S=f"{current_scale}", Mem=f"{mem_reserved:.1f}G") 

                last_batch_cache = (pred_1km.detach(), gt_1km.detach())
                
                if global_step % SAVE_EVERY_STEPS == 0 and (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                    save_checkpoint(os.path.join(CONFIG["save_dir"], "autosave_latest.pth"), 
                                  epoch, global_step, model, criterion, optimizer, scheduler, scaler, best_r2, patience_counter)
            else:
                optimizer.zero_grad()
                loop.set_postfix(L="NAN_SKIP", S=f"{current_scale}")

        # ==========================
        # 验证循环
        # ==========================
        model.eval()
        val_metrics = torch.zeros(8, device=device)
        torch.cuda.empty_cache() 
        
        with torch.no_grad():
            for aux, _, gt_1km, _, _ in val_dl:
                aux = aux.to(device, non_blocking=True)
                gt_1km = gt_1km.to(device, non_blocking=True)
                
                main_val_lowres = F.adaptive_avg_pool3d(gt_1km, output_size=(CONFIG["time_window"], 1, 1))
                main_val_input = main_val_lowres / (CONSIST_SCALE ** 2)
                
                with torch.amp.autocast("cuda", enabled=is_cuda):
                    pred_100m_val, _ = model(aux, main_val_input)
                    val_metrics += calc_metrics_1km_tensor(sum_pool3d(pred_100m_val, CONSIST_SCALE), gt_1km)
        
        avg_metrics = (val_metrics / len(val_dl)).cpu().numpy()
        nz_mae = avg_metrics[1]
        current_r2 = avg_metrics[6]
        
        avg_train_loss = train_loss / len(train_dl)
        print(f" Val | NZ_MAE={nz_mae:.2f} | R2={current_r2:.4f} | Patience={patience_counter}/{PATIENCE}")
        
        lr_curr = optimizer.param_groups[0]["lr"]
        
        # 🚀 [修改] 记录权重到 CSV
        with torch.no_grad():
            w_l1_val = torch.exp(-criterion.log_vars[0]).item()
            w_mse_val = torch.exp(-criterion.log_vars[1]).item()
        
        with open(log_file, "a", newline="") as f:
            pred_mean, pred_max, pred_nz, cons_mae = 0, 0, 0, 0
            if last_batch_cache:
                pb, gb = last_batch_cache
                pb_r, gb_r = pb * CO2_NORM_FACTOR, gb * CO2_NORM_FACTOR
                pred_mean, pred_max = pb_r.mean().item(), pb_r.max().item()
                pred_nz = (pb > 0).float().mean().item()
                cons_mae = (pb_r - gb_r).abs().mean().item()
            
            # 🚀 [修改] 写入 w_L1 和 w_MSE
            csv.writer(f).writerow([
                epoch, f"{lr_curr:.3e}", f"{avg_train_loss:.6f}", 
                f"{avg_metrics[0]:.6f}", f"{nz_mae:.6f}", f"{avg_metrics[2]:.6f}", 
                f"{avg_metrics[3]:.6f}", f"{avg_metrics[4]:.6f}", 
                f"{avg_metrics[5]:.6f}", f"{current_r2:.6f}", f"{avg_metrics[7]:.6f}",
                f"{pred_mean:.6f}", f"{pred_max:.6f}", f"{pred_nz:.6f}", 
                f"{cons_mae:.6f}", f"{w_l1_val:.4f}", f"{w_mse_val:.4f}", global_step, f"{best_r2:.6f}"
            ])
        
        scheduler.step(current_r2)
        
        if current_r2 > best_r2:
            best_r2 = float(current_r2)
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_model.pth"))
            print(f" Best Model Updated! (R2: {best_r2:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(" Early Stopping Triggered! (R2 stopped improving)")
                break
                
        if epoch % SAVE_EVERY_EPOCHS == 0:
            save_checkpoint(os.path.join(CONFIG["save_dir"], f"epoch_{epoch:03d}.pth"), 
                          epoch, global_step, model, criterion, optimizer, scheduler, scaler, best_r2, patience_counter)

    # ==========================
    # 🚀 新增：测试集最终评估
    # ==========================
    print("\n" + "="*50)
    print("🚀 Starting Final Evaluation on Test Set...")
    print("="*50)

    test_ds = DualStreamDataset(CONFIG["data_dir"], CONFIG["split_config"], mode="test", time_window=CONFIG["time_window"])
    test_dl = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_args)

    best_ckpt_path = os.path.join(CONFIG["save_dir"], "best_model.pth")
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"✅ Loaded Best Model from {best_ckpt_path}")
    else:
        print("⚠️ Best model not found, evaluating with current weights.")

    model.eval()
    test_metrics = torch.zeros(8, device=device)
    
    with torch.no_grad():
        for aux, _, gt_1km, _, _ in tqdm(test_dl, desc="Testing"):
            aux = aux.to(device)
            gt_1km = gt_1km.to(device)
            
            main_test_lowres = F.adaptive_avg_pool3d(gt_1km, output_size=(CONFIG["time_window"], 1, 1))
            main_test_input = main_test_lowres / (CONSIST_SCALE ** 2)

            with torch.amp.autocast("cuda", enabled=is_cuda):
                pred_100m_test, _ = model(aux, main_test_input)
                val_metrics_batch = calc_metrics_1km_tensor(sum_pool3d(pred_100m_test, CONSIST_SCALE), gt_1km)
                test_metrics += val_metrics_batch

    avg_test_metrics = (test_metrics / len(test_dl)).cpu().numpy()
    
    print("\n📊 [Test Set Results]")
    print(f"MAE Global:      {avg_test_metrics[0]:.4f}")
    print(f"MAE Nonzero:     {avg_test_metrics[1]:.4f}")
    print(f"MAE Balanced:    {avg_test_metrics[2]:.4f}")
    print(f"MAE Top1%:       {avg_test_metrics[3]:.4f}")
    print(f"MAE Top5%:       {avg_test_metrics[4]:.4f}")
    print(f"RMSE Global:     {avg_test_metrics[5]:.4f}")
    print(f"R2 Score:        {avg_test_metrics[6]:.4f}")
    print(f"Pearson Corr:    {avg_test_metrics[7]:.4f}")
    print("="*50 + "\n")

    test_log_file = os.path.join(CONFIG["save_dir"], "test_results.txt")
    with open(test_log_file, "w") as f:
        f.write("MAE_Global,MAE_Nonzero,MAE_Balanced,MAE_Top1,MAE_Top5,RMSE,R2,Corr\n")
        f.write(",".join([f"{x:.6f}" for x in avg_test_metrics]))
    print(f"✅ Test results saved to {test_log_file}")

if __name__ == "__main__":
    train()