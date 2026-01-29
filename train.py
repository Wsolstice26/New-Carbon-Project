import os

# ==========================================
# 🛡️ 1. 核心设置：安全与性能优化
# ==========================================
# 开启 MIOpen GEMM 算法以获得 AMD 显卡最佳卷积性能
os.environ['MIOPEN_DEBUG_CONV_GEMM'] = '1'
# 禁用 MIOpen 过于详细的日志输出，保持控制台清爽
os.environ['MIOPEN_LOG_LEVEL'] = '2' 
os.environ['MIOPEN_ENABLE_LOGGING'] = '0'
os.environ['MIOPEN_USER_DB_PATH'] = './miopen_cache'

# ✅ [关键] AMD 显卡防止显存碎片化设置
# 这能有效解决长时间训练后出现的 "Out of Memory" 错误
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

# 碳排放数据的 Log 归一化因子 (对应原始数据的 max log 值)
NORM_FACTOR = 11.0

# ==========================================
# 📊 [v1.7 升级] 精细化指标计算函数
# ==========================================
def calc_detailed_metrics(pred_real, target_real, threshold=1e-6):
    """
    计算四个维度的 MAE (Mean Absolute Error)，全方位评估模型性能。
    
    参数:
        pred_real: 反归一化后的预测值 (真实吨数)
        target_real: 反归一化后的真实标签 (真实吨数)
    
    返回:
        global_mae: 全局平均误差
        nz_mae: 城市/高排放区域误差
        z_mae: 背景区域误差
        balanced_mae: 平衡后的核心指挥棒指标
    """
    abs_diff = torch.abs(pred_real - target_real)
    
    # 1. Global MAE (传统指标)
    global_mae = abs_diff.mean().item()
    
    # 2. 生成掩码 (区分城市和背景)
    mask_nonzero = target_real > threshold
    mask_zero = ~mask_nonzero
    
    # 3. Non-Zero MAE (城市区域 - 攻坚重点)
    if mask_nonzero.sum() > 0:
        nonzero_mae = abs_diff[mask_nonzero].mean().item()
    else:
        nonzero_mae = 0.0
        
    # 4. Zero MAE (背景区域 - 监控噪点)
    if mask_zero.sum() > 0:
        zero_mae = abs_diff[mask_zero].mean().item()
    else:
        zero_mae = 0.0
        
    # 🔥 [v1.7 核心改进] Balanced MAE
    # 即使城市只占 1% 的面积，它在评价体系中也必须占 50% 的权重。
    # 这是早停 (Early Stopping) 的唯一依据。
    balanced_mae = 0.5 * nonzero_mae + 0.5 * zero_mae
        
    return global_mae, nonzero_mae, zero_mae, balanced_mae

def get_latest_checkpoint(save_dir):
    """
    自动查找保存目录中最新的模型检查点文件 (.pth)
    """
    if not os.path.exists(save_dir): 
        return None
    
    # 优先查找 latest.pth
    latest_path = os.path.join(save_dir, "latest.pth")
    if os.path.exists(latest_path): 
        return latest_path
        
    # 否则按时间戳查找最新的 epoch_*.pth
    files = glob.glob(os.path.join(save_dir, "epoch_*.pth"))
    if not files:
        return None
        
    return max(files, key=os.path.getmtime)

def train():
    # ----------------------------------------
    # 1. 环境与设备初始化
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"   显卡型号: {torch.cuda.get_device_name(0)}")
        # 显式关闭 Benchmark 以保证在 ROCm 上的稳定性
        torch.backends.cudnn.benchmark = False 
        torch.backends.cudnn.deterministic = True
    
    # 初始化 AMP (混合精度训练)
    # init_scale 设为 2048 可以防止初始梯度过小下溢
    scaler = torch.amp.GradScaler('cuda', init_scale=2048.0)
    print(f"⚡ 模式: Smart AMP (Init Scale=2048) + AMD Optimized")

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # ----------------------------------------
    # 2. 数据集加载
    # ----------------------------------------
    print(f"📦 加载数据 (Batch Size: {CONFIG['batch_size']})...")
    # 加载训练集和验证集
    train_ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'train')
    val_ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'val')
    
    # 设置 DataLoader
    # pin_memory=True 加速 CPU 到 GPU 的数据传输
    # persistent_workers=True 避免每个 Epoch 重启进程的开销
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                          num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                        num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    
    # ----------------------------------------
    # 3. 模型构建与优化器设置
    # ----------------------------------------
    print("🏗️ 初始化 DSTCarbonFormer 模型 (v1.7 Balanced Edition)...")
    model = DSTCarbonFormer(aux_c=9, main_c=1, dim=64).to(device)
    
    # 初始化自适应混合损失 (HybridLoss v1.7)
    # 内部已包含 BalancedCharbonnierLoss 和可学习权重
    criterion = HybridLoss().to(device)
    
    # 🔥 [关键] 将 Loss 的参数 (w_params) 也加入优化器
    # 这样 AdamW 就会同时优化网络权重和 Loss 的平衡系数
    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()), 
        lr=CONFIG['lr'], 
        weight_decay=1e-4
    )
    
    # 余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
    
    # 训练状态变量初始化
    start_epoch = 1
    best_balanced_mae = float('inf')  # 记录最佳平衡指标
    early_stop_counter = 0            # 早停计数器
    
    # ----------------------------------------
    # 4. 断点续训逻辑 (Resume)
    # ----------------------------------------
    if CONFIG['resume']:
        latest_ckpt = get_latest_checkpoint(CONFIG['save_dir'])
        if latest_ckpt:
            print(f"🔄 正在恢复检查点: {latest_ckpt}")
            try:
                checkpoint = torch.load(latest_ckpt, map_location=device)
                
                # 恢复模型和 Loss 状态
                model.load_state_dict(checkpoint['model_state_dict'])
                criterion.load_state_dict(checkpoint['criterion_state_dict']) # 恢复学习到的权重
                
                # 恢复优化器和调度器
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                # 恢复训练进度
                start_epoch = checkpoint['epoch'] + 1
                best_balanced_mae = checkpoint.get('best_balanced_mae', float('inf')) 
                early_stop_counter = checkpoint.get('early_stop_counter', 0)
                
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    
                print(f"✅ 恢复成功! 从 Epoch {start_epoch} 继续 (当前最佳平衡MAE: {best_balanced_mae:.4f})")
            except Exception as e:
                print(f"⚠️ 恢复失败 ({e})，将从头开始训练。")

    # ----------------------------------------
    # 5. 训练主循环
    # ----------------------------------------
    print(f"\n🚀 开始训练 (v1.7) | 目标: Balanced Loss & Balanced MAE Optimization")
    total_start = time.time()
    
    for epoch in range(start_epoch, CONFIG['epochs']+1):
        model.train()
        train_loss = 0
        loop = tqdm(train_dl, desc=f"Ep {epoch}/{CONFIG['epochs']}")
        
        for aux, main, target in loop:
            # 数据搬运到 GPU
            aux = aux.to(device, non_blocking=True)
            main = main.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            with torch.amp.autocast('cuda'):
                pred = model(aux, main)
                # 计算 Loss (自动应用 50/50 平衡和自适应权重)
                loss = criterion(pred.float(), target.float(), input_main=main.float())
            
            # NaN 熔断保护
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ 警告: Epoch {epoch} 出现 NaN/Inf Loss，跳过此 Batch")
                optimizer.zero_grad()
                continue

            # 反向传播与参数更新
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # 实时计算当前 Batch 的 MAE (仅供观察)
            with torch.no_grad():
                # 这里的 NORM_FACTOR 必须和上面定义的保持一致 (11.0)
                pred_real = torch.expm1(pred.float().detach() * NORM_FACTOR).clamp(min=0)
                target_real = torch.expm1(target.float() * NORM_FACTOR).clamp(min=0)
                # 简单计算一个全局 MAE 看看大概情况
                batch_mae = torch.abs(pred_real - target_real).mean().item()
            
            # 修改进度条显示，加上 mae
            loop.set_postfix(loss=f"{loss.item():.4f}", mae=f"{batch_mae:.2f}")
            
        avg_train_loss = train_loss / len(train_dl) if len(train_dl) > 0 else 0
        
        # ----------------------------------------
        # 6. 验证阶段 (Validation)
        # ----------------------------------------
        model.eval()
        val_loss = 0
        
        # 初始化指标累加器 [Global, NZ, Z, Balanced]
        total_metrics = np.zeros(4) 
        
        with torch.no_grad():
            for aux, main, target in val_dl:
                aux, main, target = aux.to(device), main.to(device), target.to(device)
                
                with torch.amp.autocast('cuda'):
                    pred = model(aux, main)
                    val_loss += criterion(pred.float(), target.float()).item()
                    
                    # 反归一化：将 Log 值还原为真实碳排放吨数
                    pred_real = torch.expm1(pred.float() * NORM_FACTOR).clamp(min=0)
                    target_real = torch.expm1(target.float() * NORM_FACTOR).clamp(min=0)
                    
                    # 计算精细化指标
                    m = calc_detailed_metrics(pred_real, target_real)
                    total_metrics += np.array(m)
        
        # 计算验证集平均指标
        avg_val_loss = val_loss / len(val_dl)
        avg_metrics = total_metrics / len(val_dl)
        
        # 获取当前学习到的 Loss 权重 (用于监控)
        weights = torch.exp(criterion.w_params)
        weights = (weights / weights.sum() * 3.0).detach().cpu().numpy()
        
        # 打印详细战报
        print(f"   📊 [Val] Balanced MAE={avg_metrics[3]:.3f} | 🏙️City={avg_metrics[1]:.3f} | 🌲Bg={avg_metrics[2]:.3f} | 🌍Global={avg_metrics[0]:.3f}")
        print(f"   ⚖️ [Weights] Pixel: {weights[0]:.2f} | SSIM: {weights[1]:.2f} | TV: {weights[2]:.2f}")
        
        # ----------------------------------------
        # 7. 模型保存与早停逻辑
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
        # 保存最新断点
        torch.save(checkpoint_dict, os.path.join(CONFIG['save_dir'], "latest.pth"))

        # 🔥 早停判断：只看 Balanced MAE
        if avg_metrics[3] < best_balanced_mae:
            best_balanced_mae = avg_metrics[3]
            early_stop_counter = 0
            
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], "best_model.pth"))
            torch.save(checkpoint_dict, os.path.join(CONFIG['save_dir'], "best_checkpoint.pth"))
            
            print(f"   🏆 发现更优模型! (New Best Balanced MAE: {best_balanced_mae:.4f})")
        else:
            early_stop_counter += 1
            print(f"   ⏳ 指标未改善 ({early_stop_counter}/{CONFIG['patience']}) | 最佳: {best_balanced_mae:.4f}")
        
        # 定期保存 (作为备份)
        if epoch % CONFIG['save_freq'] == 0:
            torch.save(checkpoint_dict, os.path.join(CONFIG['save_dir'], f"epoch_{epoch}.pth"))
            
        # 触发早停
        if early_stop_counter >= CONFIG['patience']:
            print(f"\n🛑 早停触发 (Patience={CONFIG['patience']})。训练结束。")
            break
            
        scheduler.step()

    print(f"\n🏁 训练结束！总耗时: {(time.time()-total_start)/60:.2f} 分钟")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\n💥 训练崩溃: {e}")