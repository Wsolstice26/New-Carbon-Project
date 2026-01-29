import os
import csv
import shutil # 用于清理旧日志

# ==========================================
# 🛡️ 1. 核心设置：安全与性能优化
# ==========================================
os.environ['MIOPEN_DEBUG_CONV_GEMM'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '2' 
os.environ['MIOPEN_ENABLE_LOGGING'] = '0'
os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import glob
import numpy as np

from data.dataset import DualStreamDataset
from models.network import DSTCarbonFormer
from models.losses import HybridLoss
from config import CONFIG 

NORM_FACTOR = 11.0

# ==========================================
# 📊 v1.9 五维指标计算函数
# ==========================================
def calc_detailed_metrics(pred_real, target_real, threshold=1e-6):
    abs_diff = torch.abs(pred_real - target_real)
    global_mae = abs_diff.mean().item()
    
    mask_nonzero = target_real > threshold
    mask_zero = ~mask_nonzero
    
    nonzero_mae = abs_diff[mask_nonzero].mean().item() if mask_nonzero.sum() > 0 else 0.0
    zero_mae = abs_diff[mask_zero].mean().item() if mask_zero.sum() > 0 else 0.0
    
    # Top 1% Ext (Threshold > 1830)
    mask_top1 = target_real > 1830
    top1_mae = abs_diff[mask_top1].mean().item() if mask_top1.sum() > 0 else 0.0
    
    balanced_mae = 0.5 * nonzero_mae + 0.5 * zero_mae
    return global_mae, nonzero_mae, zero_mae, balanced_mae, top1_mae

def get_latest_checkpoint(save_dir):
    if not os.path.exists(save_dir): return None
    latest_path = os.path.join(save_dir, "latest.pth")
    if os.path.exists(latest_path): return latest_path
    files = glob.glob(os.path.join(save_dir, "epoch_*.pth"))
    if not files: return None
    return max(files, key=os.path.getmtime)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 使用设备: {device}")
    
    scaler = torch.amp.GradScaler('cuda', init_scale=2048.0)
    print(f"⚡ 模式: v1.9 Paper-Ready Logging (LR & Adaptive Weights)")

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # ----------------------------------------
    # 📝 CSV 日志初始化 (含权重和学习率)
    # ----------------------------------------
    log_file = os.path.join(CONFIG['save_dir'], 'training_log.csv')
    
    # 如果是重新开始，且没有检查点，建议手动清理一下 log_file，或者这里会自动追加
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # 完美的论文数据表头
            header = [
                'Epoch', 'LR',                 # 训练进度与学习率
                'Train_Loss', 'Val_Loss',      # Loss 曲线
                'MAE_Global', 'MAE_Balanced',  # 核心指标
                'MAE_City', 'MAE_Bg', 'MAE_Ext', # 细节指标
                'W_Pixel', 'W_SSIM', 'W_TV'    # 🔥 自适应权重变化 (论文核心创新证据)
            ]
            writer.writerow(header)
            print(f"📝 已创建全能日志文件: {log_file}")

    # ----------------------------------------
    # 数据加载
    # ----------------------------------------
    print(f"📦 加载数据...")
    train_ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'train')
    val_ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'val')
    
    # ⚠️ 改为 False 以保证长时间运行的稳定性
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                          num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                        num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    
    model = DSTCarbonFormer(aux_c=9, main_c=1, dim=64).to(device)
    criterion = HybridLoss().to(device)
    
    optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
    
    start_epoch = 1
    best_balanced_mae = float('inf')
    early_stop_counter = 0
    
    # ----------------------------------------
    # 断点续训
    # ----------------------------------------
    if CONFIG['resume']:
        latest_ckpt = get_latest_checkpoint(CONFIG['save_dir'])
        if latest_ckpt:
            print(f"🔄 正在恢复检查点: {latest_ckpt}")
            try:
                checkpoint = torch.load(latest_ckpt, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                criterion.load_state_dict(checkpoint['criterion_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                # 兼容旧版本的 checkpoint (如果没有保存 scaler)
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
                start_epoch = checkpoint['epoch'] + 1
                best_balanced_mae = checkpoint.get('best_balanced_mae', float('inf')) 
                early_stop_counter = checkpoint.get('early_stop_counter', 0)
                print(f"✅ 恢复成功! 从 Epoch {start_epoch} 继续")
            except Exception as e:
                print(f"⚠️ 恢复失败 ({e})，将从头开始训练。")

    print(f"\n🚀 开始训练 (v1.9)...")
    
    for epoch in range(start_epoch, CONFIG['epochs']+1):
        model.train()
        train_loss = 0
        loop = tqdm(train_dl, desc=f"Ep {epoch}/{CONFIG['epochs']}")
        
        for aux, main, target in loop:
            aux, main, target = aux.to(device, non_blocking=True), main.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred = model(aux, main)
                loss = criterion(pred.float(), target.float(), input_main=main.float())
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            with torch.no_grad():
                pred_real = torch.expm1(pred.float().detach() * NORM_FACTOR).clamp(min=0)
                target_real = torch.expm1(target.float() * NORM_FACTOR).clamp(min=0)
                g_mae, nz_mae, z_mae, bal_mae, ext_mae = calc_detailed_metrics(pred_real, target_real)
            
            loop.set_postfix(L=f"{loss.item():.3f}", G=f"{g_mae:.2f}", C=f"{nz_mae:.2f}", E=f"{ext_mae:.1f}")
            
        avg_train_loss = train_loss / len(train_dl) if len(train_dl) > 0 else 0
        
        # --- 验证阶段 ---
        model.eval()
        val_loss = 0
        total_metrics = np.zeros(5) 
        batch_count = 0
        
        with torch.no_grad():
            for aux, main, target in val_dl:
                aux, main, target = aux.to(device), main.to(device), target.to(device)
                with torch.amp.autocast('cuda'):
                    pred = model(aux, main)
                    val_loss += criterion(pred.float(), target.float()).item()
                    pred_real = torch.expm1(pred.float() * NORM_FACTOR).clamp(min=0)
                    target_real = torch.expm1(target.float() * NORM_FACTOR).clamp(min=0)
                    m = calc_detailed_metrics(pred_real, target_real)
                    total_metrics += np.array(m)
                    batch_count += 1
        
        avg_val_loss = val_loss / batch_count if batch_count > 0 else 0
        avg_metrics = total_metrics / batch_count if batch_count > 0 else np.zeros(5)
        
        # 获取当前的学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 获取当前的权重状态
        weights = torch.exp(criterion.w_params)
        weights = (weights / weights.sum() * 3.0).detach().cpu().numpy()
        w_pixel, w_ssim, w_tv = weights[0], weights[1], weights[2]
        
        print(f"   📊 [Val] Bal={avg_metrics[3]:.3f} | 🏙️Nz={avg_metrics[1]:.3f} | 🏭Ext={avg_metrics[4]:.3f}")
        print(f"   ⚖️ [Weights] Px:{w_pixel:.2f} | SSIM:{w_ssim:.2f} | TV:{w_tv:.2f} | LR:{current_lr:.2e}")
        
        # 🔥 全能写入 CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                f"{current_lr:.2e}",  # LR
                f"{avg_train_loss:.5f}", 
                f"{avg_val_loss:.5f}", 
                f"{avg_metrics[0]:.4f}", # Global
                f"{avg_metrics[3]:.4f}", # Balanced
                f"{avg_metrics[1]:.4f}", # City
                f"{avg_metrics[2]:.4f}", # Bg
                f"{avg_metrics[4]:.4f}", # Ext
                f"{w_pixel:.4f}",       # W_Pixel
                f"{w_ssim:.4f}",        # W_SSIM
                f"{w_tv:.4f}"           # W_TV
            ])

        # --- 保存与早停 ---
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'criterion_state_dict': criterion.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_balanced_mae': best_balanced_mae,
            'early_stop_counter': early_stop_counter
        }
        torch.save(checkpoint_dict, os.path.join(CONFIG['save_dir'], "latest.pth"))

        if avg_metrics[3] < best_balanced_mae:
            best_balanced_mae = avg_metrics[3]
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], "best_model.pth"))
            torch.save(checkpoint_dict, os.path.join(CONFIG['save_dir'], "best_checkpoint.pth"))
            print(f"   🏆 New Best! (Balanced: {best_balanced_mae:.4f})")
        else:
            early_stop_counter += 1
            print(f"   ⏳ Patience ({early_stop_counter}/{CONFIG['patience']})")
        
        if epoch % CONFIG['save_freq'] == 0:
            torch.save(checkpoint_dict, os.path.join(CONFIG['save_dir'], f"epoch_{epoch}.pth"))
            
        if early_stop_counter >= CONFIG['patience']:
            print(f"\n🛑 早停触发。")
            break
            
        scheduler.step()

    print(f"\n🏁 训练结束！")

if __name__ == "__main__":
    train()