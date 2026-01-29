import os

# ==========================================
# 🛡️ 1. 核心设置：安全与性能优化
# ==========================================
# 开启 MIOpen GEMM 算法以获得 AMD 显卡最佳卷积性能
os.environ['MIOPEN_DEBUG_CONV_GEMM'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '2' 
os.environ['MIOPEN_ENABLE_LOGGING'] = '0'
os.environ['MIOPEN_USER_DB_PATH'] = './miopen_cache'

# ✅ [关键] AMD 显卡防止显存碎片化设置
os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import glob
import numpy as np

# 导入项目自定义模块
from data.dataset import DualStreamDataset
from models.network import DSTCarbonFormer
from models.losses import HybridLoss
from config import CONFIG 

# 碳排放数据的 Log 归一化因子
NORM_FACTOR = 11.0

# ==========================================
# 📊 [核心] v1.8 五维指标计算函数
# ==========================================
def calc_detailed_metrics(pred_real, target_real, threshold=1e-6):
    """
    计算五维 MAE 指标。
    返回顺序: Global, NonZero(City), Zero(Bg), Balanced, Top1%(Ext)
    """
    abs_diff = torch.abs(pred_real - target_real)
    
    # 1. Global MAE
    global_mae = abs_diff.mean().item()
    
    # 2. 基础掩码
    mask_nonzero = target_real > threshold
    mask_zero = ~mask_nonzero
    
    # 3. Non-Zero MAE (City/普通排放)
    if mask_nonzero.sum() > 0:
        nonzero_mae = abs_diff[mask_nonzero].mean().item()
    else:
        nonzero_mae = 0.0
        
    # 4. Zero MAE (Bg/背景)
    if mask_zero.sum() > 0:
        zero_mae = abs_diff[mask_zero].mean().item()
    else:
        zero_mae = 0.0
        
    # 5. Top 1% MAE (Extreme Values)
    # 阈值 1830 来自全量数据分析的 Q99
    mask_top1 = target_real > 1830
    if mask_top1.sum() > 0:
        top1_mae = abs_diff[mask_top1].mean().item()
    else:
        # 如果当前 Batch 没有超级电厂，返回 0.0 或 None (这里返回0方便打印)
        top1_mae = 0.0
        
    # 6. Balanced MAE (核心指挥棒)
    # 50% 城市 + 50% 背景
    balanced_mae = 0.5 * nonzero_mae + 0.5 * zero_mae
        
    return global_mae, nonzero_mae, zero_mae, balanced_mae, top1_mae

def get_latest_checkpoint(save_dir):
    """自动查找保存目录中最新的模型检查点文件"""
    if not os.path.exists(save_dir): return None
    latest_path = os.path.join(save_dir, "latest.pth")
    if os.path.exists(latest_path): return latest_path
    files = glob.glob(os.path.join(save_dir, "epoch_*.pth"))
    if not files: return None
    return max(files, key=os.path.getmtime)

def train():
    # ----------------------------------------
    # 1. 环境初始化
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"   显卡型号: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = False 
        torch.backends.cudnn.deterministic = True
    
    scaler = torch.amp.GradScaler('cuda', init_scale=2048.0)
    print(f"⚡ 模式: v1.8 Full Monitor (Bg/Nz/Ext)")

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # ----------------------------------------
    # 2. 数据加载
    # ----------------------------------------
    print(f"📦 加载数据 (Batch Size: {CONFIG['batch_size']})...")
    train_ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'train')
    val_ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'val')
    
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                          num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                        num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    
    # ----------------------------------------
    # 3. 模型与 Loss
    # ----------------------------------------
    print("🏗️ 初始化 DSTCarbonFormer 模型...")
    model = DSTCarbonFormer(aux_c=9, main_c=1, dim=64).to(device)
    
    # 使用包含 AdaptiveCVLoss 的 HybridLoss
    criterion = HybridLoss().to(device)
    
    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()), 
        lr=CONFIG['lr'], 
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
    
    start_epoch = 1
    best_balanced_mae = float('inf')
    early_stop_counter = 0
    
    # ----------------------------------------
    # 4. 断点续训
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
                start_epoch = checkpoint['epoch'] + 1
                best_balanced_mae = checkpoint.get('best_balanced_mae', float('inf')) 
                early_stop_counter = checkpoint.get('early_stop_counter', 0)
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print(f"✅ 恢复成功! 从 Epoch {start_epoch} 继续")
            except Exception as e:
                print(f"⚠️ 恢复失败 ({e})，将从头开始训练。")

    # ----------------------------------------
    # 5. 训练主循环
    # ----------------------------------------
    print(f"\n🚀 开始训练 | Top 1% Threshold: >1830 tons")
    total_start = time.time()
    
    for epoch in range(start_epoch, CONFIG['epochs']+1):
        model.train()
        train_loss = 0
        loop = tqdm(train_dl, desc=f"Ep {epoch}/{CONFIG['epochs']}")
        
        for aux, main, target in loop:
            aux, main, target = aux.to(device, non_blocking=True), main.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                pred = model(aux, main)
                # HybridLoss 内部已包含 AdaptiveCVLoss
                loss = criterion(pred.float(), target.float(), input_main=main.float())
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ NaN Warning at Epoch {epoch}")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # 🔥 [核心修改] 实时全监控
            with torch.no_grad():
                pred_real = torch.expm1(pred.float().detach() * NORM_FACTOR).clamp(min=0)
                target_real = torch.expm1(target.float() * NORM_FACTOR).clamp(min=0)
                
                # 计算所有指标
                g_mae, nz_mae, z_mae, bal_mae, ext_mae = calc_detailed_metrics(pred_real, target_real)
            
            # 🔥 显示在进度条上 (为了节省空间，使用简写)
            # L=Loss, A=All(Global), NZ=NonZero, BG=Background, E=Extreme(Top1%)
            loop.set_postfix(
                loss=f"{loss.item():.3f}", 
                all=f"{g_mae:.2f}",   # 全局
                nz=f"{nz_mae:.2f}",   # 城市
                bg=f"{z_mae:.2f}",    # 背景 (重要！看是否干净)
                ext=f"{ext_mae:.1f}"  # 极端值 (重要！看电厂)
            )
            
        # ----------------------------------------
        # 6. 验证阶段
        # ----------------------------------------
        model.eval()
        val_loss = 0
        
        # 累加器: Global, NonZero, Zero, Balanced, Top1
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
        
        avg_metrics = total_metrics / batch_count if batch_count > 0 else np.zeros(5)
        
        weights = torch.exp(criterion.w_params)
        weights = (weights / weights.sum() * 3.0).detach().cpu().numpy()
        
        # 打印详细战报
        print(f"   📊 [Val] Bal={avg_metrics[3]:.3f} | 🏙️Nz={avg_metrics[1]:.3f} | 🌲Bg={avg_metrics[2]:.3f} | 🏭Ext={avg_metrics[4]:.3f}")
        print(f"   ⚖️ [Weights] Pixel: {weights[0]:.2f} | SSIM: {weights[1]:.2f} | TV: {weights[2]:.2f}")
        
        # ----------------------------------------
        # 7. 保存与早停
        # ----------------------------------------
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