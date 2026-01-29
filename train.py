import os

# ==========================================
# 🛡️ 1. 核心设置：安全与性能优化
# ==========================================
os.environ['MIOPEN_DEBUG_CONV_GEMM'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '2' 
os.environ['MIOPEN_ENABLE_LOGGING'] = '0'
os.environ['MIOPEN_USER_DB_PATH'] = './miopen_cache'

# ✅ [修正] AMD 显卡专用的防显存碎片化设置
# max_split_size_mb:128 是解决 ROCm 显存 OOM 的最佳实践
os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import glob
import numpy as np

# 导入项目模块
from data.dataset import DualStreamDataset
from models.network import DSTCarbonFormer
from models.losses import HybridLoss
from config import CONFIG 

NORM_FACTOR = 11.0

def get_latest_checkpoint(save_dir):
    if not os.path.exists(save_dir): return None
    latest_path = os.path.join(save_dir, "latest.pth")
    if os.path.exists(latest_path): return latest_path
    files = glob.glob(os.path.join(save_dir, "epoch_*.pth"))
    return max(files, key=os.path.getmtime) if files else None

def train():
    # 🕵️ 暂时关闭侦探模式，提高速度 (除非再次报错)
    # torch.autograd.set_detect_anomaly(True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"   显卡型号: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = False 
        torch.backends.cudnn.deterministic = True
    
    # ✅ [修正] 降低初始 Scale (65536 -> 2048)
    # 这能极大减少训练初期的 NaN 概率
    scaler = torch.amp.GradScaler('cuda', init_scale=2048.0)
    print(f"⚡ 模式: Smart AMP (Init Scale=2048) + AMD Optimized")

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # ----------------------------------------
    # 2. 数据准备
    # ----------------------------------------
    print(f"📦 加载数据 (Batch Size: {CONFIG['batch_size']})...")
    train_ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'train')
    val_ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'val')
    
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                          num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                        num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    
    # ----------------------------------------
    # 3. 模型与优化器
    # ----------------------------------------
    print("🏗️ 初始化 DSTCarbonFormer 模型 (v1.6)...")
    model = DSTCarbonFormer(aux_c=9, main_c=1, dim=64).to(device)
    
    criterion = HybridLoss(alpha=1.0, beta=0.1, gamma=0.1, delta=0.05, eta=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
    
    start_epoch = 1
    best_loss = float('inf')
    early_stop_counter = 0 
    
    if CONFIG['resume']:
        latest_ckpt = get_latest_checkpoint(CONFIG['save_dir'])
        if latest_ckpt:
            print(f"🔄 正在恢复检查点: {latest_ckpt}")
            try:
                checkpoint = torch.load(latest_ckpt, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_loss = checkpoint.get('best_loss', float('inf'))
                early_stop_counter = checkpoint.get('early_stop_counter', 0)
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print(f"✅ 恢复成功! 从 Epoch {start_epoch} 继续")
            except Exception as e:
                print(f"⚠️ 恢复失败 ({e})，将从头开始训练。")

    # ----------------------------------------
    # 4. 训练主循环
    # ----------------------------------------
    print(f"\n🚀 开始训练 | 总轮数: {CONFIG['epochs']}")
    total_start = time.time()
    
    for epoch in range(start_epoch, CONFIG['epochs']+1):
        model.train()
        train_loss = 0
        loop = tqdm(train_dl, desc=f"Ep {epoch}/{CONFIG['epochs']}")
        
        for aux, main, target in loop:
            aux = aux.to(device, non_blocking=True)
            main = main.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # ✅ 开启 AMP
            with torch.amp.autocast('cuda'):
                pred = model(aux, main)
                # 强制 Loss 走 FP32
                loss = criterion(pred.float(), target.float(), input_main=main.float())
            
            # ✅ [修正] 健壮的 NaN 处理逻辑
            # 如果 Loss 是 NaN/Inf，直接跳过，千万不要 update scaler
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ 警告: Epoch {epoch} 出现 NaN/Inf Loss，跳过此 Batch")
                optimizer.zero_grad()
                # 🔴 关键点：这里绝不能调用 scaler.update()，否则会报错 "No inf checks..."
                continue

            # 正常反向传播
            scaler.scale(loss).backward()
            
            # 先 unscale 再裁剪梯度
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            with torch.no_grad():
                pred_real = torch.expm1(pred.float().detach() * NORM_FACTOR).clamp(min=0)
                target_real = torch.expm1(target.float() * NORM_FACTOR).clamp(min=0)
                batch_mae = torch.abs(pred_real - target_real).mean().item()

            loop.set_postfix(loss=f"{loss.item():.4f}", mae=f"{batch_mae:.2f}")
            
        avg_train_loss = train_loss / len(train_dl) if len(train_dl) > 0 else 0
        
        # --- 验证阶段 ---
        model.eval()
        val_loss = 0
        total_real_mae = 0 
        
        with torch.no_grad():
            for aux, main, target in val_dl:
                aux, main, target = aux.to(device), main.to(device), target.to(device)
                
                with torch.amp.autocast('cuda'):
                    pred = model(aux, main)
                    val_loss += criterion(pred.float(), target.float(), input_main=main.float()).item()
                    
                    pred_real = torch.expm1(pred.float() * NORM_FACTOR).clamp(min=0)
                    target_real = torch.expm1(target.float() * NORM_FACTOR).clamp(min=0)
                    total_real_mae += torch.abs(pred_real - target_real).mean().item()
        
        avg_val_loss = val_loss / len(val_dl)
        avg_real_mae = total_real_mae / len(val_dl)
        
        print(f"   📊 Summary: Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f} | MAE={avg_real_mae:.3f} | LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_loss': best_loss,
            'early_stop_counter': early_stop_counter
        }
        
        torch.save(checkpoint_dict, os.path.join(CONFIG['save_dir'], "latest.pth"))

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], "best_model.pth"))
            torch.save(checkpoint_dict, os.path.join(CONFIG['save_dir'], "best_checkpoint.pth"))
            print(f"   🏆 最佳模型已更新!")
        else:
            early_stop_counter += 1
            print(f"   ⏳ Loss 未下降 ({early_stop_counter}/{CONFIG['patience']})")
        
        if epoch % CONFIG['save_freq'] == 0:
            torch.save(checkpoint_dict, os.path.join(CONFIG['save_dir'], f"epoch_{epoch}.pth"))
            
        if early_stop_counter >= CONFIG['patience']:
            print(f"\n🛑 早停。")
            break
            
        scheduler.step()

    print(f"\n🏁 训练全部完成！总耗时: {(time.time()-total_start)/60:.2f} 分钟")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\n💥 训练崩溃: {e}")