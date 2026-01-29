import os

# ==========================================
# 🛡️ 1. 核心设置：安全与性能优化
# ==========================================
# 开启 GEMM 以获得最佳性能
os.environ['MIOPEN_DEBUG_CONV_GEMM'] = '1'
# 禁用 MIOpen 日志
os.environ['MIOPEN_LOG_LEVEL'] = '2' 
os.environ['MIOPEN_ENABLE_LOGGING'] = '0'
os.environ['MIOPEN_USER_DB_PATH'] = './miopen_cache'

# ✅ AMD 显卡防显存碎片化关键设置
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

# ==========================================
# 📊 [新增] 精细化指标计算函数
# ==========================================
def calc_detailed_metrics(pred_real, target_real, threshold=1e-6):
    """
    计算三个维度的 MAE：
    1. Global: 全局平均 (用于早停)
    2. Non-Zero: 只看高排放区 (城市/工业区)
    3. Zero: 只看背景 (森林/荒地)
    """
    abs_diff = torch.abs(pred_real - target_real)
    
    # 1. 全局 MAE
    global_mae = abs_diff.mean().item()
    
    # 2. 生成掩码
    mask_nonzero = target_real > threshold
    mask_zero = ~mask_nonzero
    
    # 3. Non-Zero MAE (攻坚指标)
    if mask_nonzero.sum() > 0:
        nonzero_mae = abs_diff[mask_nonzero].mean().item()
    else:
        nonzero_mae = 0.0
        
    # 4. Zero MAE (防守指标)
    if mask_zero.sum() > 0:
        zero_mae = abs_diff[mask_zero].mean().item()
    else:
        zero_mae = 0.0
        
    return global_mae, nonzero_mae, zero_mae

def get_latest_checkpoint(save_dir):
    if not os.path.exists(save_dir): return None
    latest_path = os.path.join(save_dir, "latest.pth")
    if os.path.exists(latest_path): return latest_path
    files = glob.glob(os.path.join(save_dir, "epoch_*.pth"))
    return max(files, key=os.path.getmtime) if files else None

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"   显卡型号: {torch.cuda.get_device_name(0)}")
        # 显式关闭 Benchmark 以保证在 ROCm 上的稳定性
        torch.backends.cudnn.benchmark = False 
        torch.backends.cudnn.deterministic = True
    
    # 初始化 AMP (混合精度)，初始 Scale 设为 2048 防止 NaN
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
    print("🏗️ 初始化 DSTCarbonFormer 模型 (v1.6 Mamba+MoE+FFT)...")
    model = DSTCarbonFormer(aux_c=9, main_c=1, dim=64).to(device)
    
    # 初始化自适应混合损失 (不需要传参数了，它自己学)
    criterion = HybridLoss().to(device)
    
    # 🔥 关键修改：将 Loss 的可学习参数也加入优化器
    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()), # 同时优化网络和权重
        lr=CONFIG['lr'], 
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
    
    start_epoch = 1
    # 🔥 [修改] 不再记录 best_loss，而是记录 best_mae
    best_mae = float('inf') 
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
                
                # 兼容旧版 checkpoint
                best_mae = checkpoint.get('best_mae', float('inf')) 
                early_stop_counter = checkpoint.get('early_stop_counter', 0)
                
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print(f"✅ 恢复成功! 从 Epoch {start_epoch} 继续 (当前最佳MAE: {best_mae:.4f})")
            except Exception as e:
                print(f"⚠️ 恢复失败 ({e})，将从头开始训练。")

    # ----------------------------------------
    # 4. 训练主循环
    # ----------------------------------------
    print(f"\n🚀 开始训练 (v1.7 Auto-Weighting) | 目标: MAE-based Optimization")
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
            
            with torch.amp.autocast('cuda'):
                pred = model(aux, main)
                # Loss 自动加权
                loss = criterion(pred.float(), target.float(), input_main=main.float())
            
            # NaN 熔断机制
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ 警告: Epoch {epoch} 出现 NaN/Inf Loss，跳过此 Batch")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # 训练进度条只看个大概的 MAE
            with torch.no_grad():
                pred_real = torch.expm1(pred.float().detach() * NORM_FACTOR).clamp(min=0)
                target_real = torch.expm1(target.float() * NORM_FACTOR).clamp(min=0)
                batch_mae = torch.abs(pred_real - target_real).mean().item()

            loop.set_postfix(loss=f"{loss.item():.4f}", mae=f"{batch_mae:.2f}")
            
        avg_train_loss = train_loss / len(train_dl) if len(train_dl) > 0 else 0
        
        # --- 验证阶段 (精细化监控) ---
        model.eval()
        val_loss = 0
        
        # 三大指标累加器
        total_global_mae = 0
        total_nonzero_mae = 0
        total_zero_mae = 0
        
        with torch.no_grad():
            for aux, main, target in val_dl:
                aux, main, target = aux.to(device), main.to(device), target.to(device)
                
                with torch.amp.autocast('cuda'):
                    pred = model(aux, main)
                    val_loss += criterion(pred.float(), target.float(), input_main=main.float()).item()
                    
                    pred_real = torch.expm1(pred.float() * NORM_FACTOR).clamp(min=0)
                    target_real = torch.expm1(target.float() * NORM_FACTOR).clamp(min=0)
                    
                    # 🔥 调用精细化计算函数
                    g_mae, nz_mae, z_mae = calc_detailed_metrics(pred_real, target_real)
                    
                    total_global_mae += g_mae
                    total_nonzero_mae += nz_mae
                    total_zero_mae += z_mae
        
        # 计算平均值
        avg_val_loss = val_loss / len(val_dl)
        avg_global_mae = total_global_mae / len(val_dl)
        avg_nonzero_mae = total_nonzero_mae / len(val_dl)
        avg_zero_mae = total_zero_mae / len(val_dl)
        
        # 📝 打印详细战报 (监控权重变化)
        # 获取当前学习到的权重参数 (转回正常的 sigma 值以便观察)
        w_pix = torch.exp(-criterion.log_vars[0]).item()
        w_ssim = torch.exp(-criterion.log_vars[1]).item()
        w_tv = torch.exp(-criterion.log_vars[2]).item()
        
        print(f"   📊 [Val] Loss={avg_val_loss:.4f} | 🌍Global={avg_global_mae:.3f} | 🏙️City={avg_nonzero_mae:.3f} | 🌲Bg={avg_zero_mae:.3f}")
        print(f"   ⚖️ [Weights] Pixel: {w_pix:.2f} | SSIM: {w_ssim:.2f} | TV: {w_tv:.2f}")
        
        # 保存 Latest
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 记得保存 criterion 的状态 (也就是学习到的权重)
            'criterion_state_dict': criterion.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_mae': best_mae,
            'early_stop_counter': early_stop_counter
        }
        torch.save(checkpoint_dict, os.path.join(CONFIG['save_dir'], "latest.pth"))

        # 🔥 核心修改：基于 Global MAE 的早停
        if avg_global_mae < best_mae:
            best_mae = avg_global_mae
            early_stop_counter = 0
            
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], "best_model.pth"))
            torch.save(checkpoint_dict, os.path.join(CONFIG['save_dir'], "best_checkpoint.pth"))
            
            print(f"   🏆 最佳模型已更新! (New Best MAE: {best_mae:.3f})")
        else:
            early_stop_counter += 1
            print(f"   ⏳ MAE 未改善 ({early_stop_counter}/{CONFIG['patience']}) | 最佳: {best_mae:.3f}")
        
        if epoch % CONFIG['save_freq'] == 0:
            torch.save(checkpoint_dict, os.path.join(CONFIG['save_dir'], f"epoch_{epoch}.pth"))
            
        if early_stop_counter >= CONFIG['patience']:
            print(f"\n🛑 早停触发 (Patience={CONFIG['patience']})。")
            break
            
        scheduler.step()

    print(f"\n🏁 训练结束！总耗时: {(time.time()-total_start)/60:.2f} 分钟")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\n💥 训练崩溃: {e}")