import os

# ==========================================
# 🚀 [环境补丁] AMD ROCm 缓存与优化
# ==========================================
# 1. 设置持久化缓存目录 (加速二次启动)
cache_dir = os.path.expanduser("~/.cache/miopen")
os.makedirs(cache_dir, exist_ok=True)
os.environ['MIOPEN_USER_DB_PATH'] = cache_dir
os.environ['MIOPEN_CUSTOM_CACHE_DIR'] = cache_dir

# 2. 强制开启 Workspace (防止显存警告)
os.environ['MIOPEN_FORCE_USE_WORKSPACE'] = '1'

# 3. 日志与线程优化
os.environ['MIOPEN_LOG_LEVEL'] = '4'
os.environ['MIOPEN_DEBUG_CONV_GEMM'] = '0'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# ❌ [已移除] 显存锁
# 刚才的测试证明这行代码会导致显存分配失败(OOM)，让 PyTorch 自动管理更安全
# os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:128'

import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import numpy as np

# 导入项目模块
from data.dataset import DualStreamDataset
from models.network import DSTCarbonFormer
from models.losses import HybridLoss 
from config import CONFIG 

# 开启 cudnn/miopen 自动寻优
torch.backends.cudnn.benchmark = True

def calc_detailed_metrics(pred_real, target_real, threshold=1e-6):
    """计算详细评估指标"""
    abs_diff = torch.abs(pred_real - target_real)
    global_mae = abs_diff.mean().item()
    
    mask_nonzero = target_real > threshold
    mask_zero = ~mask_nonzero
    
    nonzero_mae = abs_diff[mask_nonzero].mean().item() if mask_nonzero.sum() > 0 else 0.0
    zero_mae = abs_diff[mask_zero].mean().item() if mask_zero.sum() > 0 else 0.0
    
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
    print(f"🔥 设备: {device} | 模式: 120x120 Final (Loss Scaled x100)")
    print(f"📂 数据集: {CONFIG['data_dir']}")
    print(f"📏 Dim: {CONFIG.get('dim', 48)} | Batch: {CONFIG['batch_size']}")
    
    scaler = torch.amp.GradScaler('cuda', init_scale=65535.0)
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # 初始化日志
    log_file = os.path.join(CONFIG['save_dir'], 'training_log.csv')
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'LR', 'Train_Loss', 'Val_Loss', 
                             'MAE_Global', 'MAE_Balanced', 'MAE_Ext', 
                             'W_Pixel', 'W_SSIM', 'W_TV', 'W_Cons'])

    # --- 加载数据 ---
    print(f"📦 加载数据 (Workers={CONFIG['num_workers']})...")
    train_ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'train', time_window=CONFIG['time_window'])
    val_ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'val', time_window=CONFIG['time_window'])
    
    # 动态设置 persistent_workers
    use_persistent = (CONFIG['num_workers'] > 0)
    
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                          num_workers=CONFIG['num_workers'], pin_memory=True, 
                          persistent_workers=use_persistent)
                          
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                        num_workers=CONFIG['num_workers'], pin_memory=True, 
                        persistent_workers=use_persistent)
    
    print(f"✅ 样本数: Train={len(train_ds)} | Val={len(val_ds)}")

    # --- 模型与 Loss ---
    model = DSTCarbonFormer(aux_c=9, main_c=1, dim=CONFIG.get('dim', 48)).to(device)
    criterion = HybridLoss(consistency_scale=CONFIG['consistency_scale']).to(device)
    
    optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
    
    start_epoch = 1
    best_balanced_mae = float('inf')
    early_stop_counter = 0

    # --- 恢复断点 ---
    if CONFIG['resume']:
        latest_ckpt = get_latest_checkpoint(CONFIG['save_dir'])
        if latest_ckpt:
            print(f"🔄 恢复检查点: {latest_ckpt}")
            try:
                checkpoint = torch.load(latest_ckpt, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'criterion_state_dict' in checkpoint:
                     try: criterion.load_state_dict(checkpoint['criterion_state_dict'])
                     except: print("⚠️ Loss权重不匹配，已重置")
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_balanced_mae = checkpoint.get('best_balanced_mae', float('inf'))
                print(f"✅ 恢复成功! 从 Ep {start_epoch} 开始")
            except Exception as e:
                print(f"⚠️ 恢复失败 ({e})，重新开始")

    # --- 训练循环 ---
    print(f"\n🚀 开始训练...")
    for epoch in range(start_epoch, CONFIG['epochs']+1):
        model.train()
        train_loss = 0
        loop = tqdm(train_dl, desc=f"Ep {epoch}/{CONFIG['epochs']}")
        
        for aux, main, target in loop:
            aux, main, target = aux.to(device, non_blocking=True), main.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred = model(aux, main)
                # 🔥 [修改] 将 Loss 放大 100 倍
                # 这样 log 里的 loss 值会变成 0.x 或 1.x，看起来更直观
                loss = criterion(pred, target, main) * 100.0
            
            if torch.isnan(loss): 
                print("⚠️ Loss is NaN!"); continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            with torch.no_grad():
                pred_real = torch.expm1(pred.float().detach() * CONFIG['norm_factor']).clamp(min=0)
                target_real = torch.expm1(target.float() * CONFIG['norm_factor']).clamp(min=0)
                _, _, _, b_mae, _ = calc_detailed_metrics(pred_real, target_real)
            
            loop.set_postfix(L=f"{loss.item():.3f}", B=f"{b_mae:.2f}")
            
        avg_train_loss = train_loss / len(train_dl) if len(train_dl) > 0 else 0
        
        # --- 验证 ---
        model.eval()
        val_loss = 0
        total_metrics = np.zeros(5) 
        batch_count = 0
        
        with torch.no_grad():
            for aux, main, target in val_dl:
                aux, main, target = aux.to(device), main.to(device), target.to(device)
                with torch.amp.autocast('cuda'):
                    pred = model(aux, main)
                    # 🔥 [修改] 验证集 Loss 也要记得放大，保持一致
                    val_loss += (criterion(pred, target, main) * 100.0).item()
                    
                    pred_real = torch.expm1(pred.float() * CONFIG['norm_factor']).clamp(min=0)
                    target_real = torch.expm1(target.float() * CONFIG['norm_factor']).clamp(min=0)
                    m = calc_detailed_metrics(pred_real, target_real)
                    total_metrics += np.array(m)
                    batch_count += 1
        
        avg_val_loss = val_loss / batch_count if batch_count > 0 else 0
        avg_metrics = total_metrics / batch_count if batch_count > 0 else np.zeros(5)
        
        lr = optimizer.param_groups[0]['lr']
        ws = torch.exp(criterion.w_params)
        ws = (ws / ws.sum() * 4.0).detach().cpu().numpy()
        
        print(f"   📊 Val Loss={avg_val_loss:.4f} | Bal MAE={avg_metrics[3]:.3f}")
        print(f"   ⚖️ Weights -> Px:{ws[0]:.2f} SSIM:{ws[1]:.2f} TV:{ws[2]:.2f} Cons:{ws[3]:.2f}")

        # --- 保存记录 ---
        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, f"{lr:.2e}", 
                f"{avg_train_loss:.5f}", f"{avg_val_loss:.5f}", 
                f"{avg_metrics[0]:.4f}", f"{avg_metrics[3]:.4f}", f"{avg_metrics[4]:.4f}", 
                f"{ws[0]:.3f}", f"{ws[1]:.3f}", f"{ws[2]:.3f}", f"{ws[3]:.3f}"
            ])

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_balanced_mae': best_balanced_mae
        }
        torch.save(ckpt, os.path.join(CONFIG['save_dir'], "latest.pth"))

        if avg_metrics[3] < best_balanced_mae:
            best_balanced_mae = avg_metrics[3]
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], "best_model.pth"))
            print(f"   🏆 New Best Model Saved!")
        else:
            early_stop_counter += 1
            print(f"   ⏳ No improve {early_stop_counter}/{CONFIG['patience']}")
            
        if early_stop_counter >= CONFIG['patience']: break
        scheduler.step()

if __name__ == "__main__":
    train()