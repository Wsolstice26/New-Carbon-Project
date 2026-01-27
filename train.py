import os

# ==========================================
# 🤫 1. 核心设置：环境配置 (针对 RX 9060 XT)
# ==========================================
# 屏蔽 MIOpen 的繁琐警告 (只显示 Error)
os.environ['MIOPEN_LOG_LEVEL'] = '2' 
# 禁止输出日志文件，防止磁盘垃圾
os.environ['MIOPEN_ENABLE_LOGGING'] = '0'
# 将编译缓存放在当前目录，防止多进程冲突
os.environ['MIOPEN_USER_DB_PATH'] = './miopen_cache'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast # 使用兼容性更好的导入方式
from tqdm import tqdm
import time
import glob
import numpy as np

# 导入项目模块
from data.dataset import DualStreamDataset
from models.network import DSTCarbonFormer
from models.losses import HybridLoss
from config import CONFIG

# 定义还原真实值所需的参数 (用于显示 MAE)
NORM_FACTOR = 11.0

def get_latest_checkpoint(save_dir):
    """查找最新的检查点文件"""
    if not os.path.exists(save_dir):
        return None
    files = glob.glob(os.path.join(save_dir, "epoch_*.pth"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def train():
    # ----------------------------------------
    # 1. 设备与性能设置
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"   显卡型号: {torch.cuda.get_device_name(0)}")
        # [关键] 开启 Benchmark，允许 MIOpen 搜索算法 (虽然会 fallback 到 GEMM，但这是正常的)
        torch.backends.cudnn.benchmark = True 
        # 关闭确定性模式，追求速度
        torch.backends.cudnn.deterministic = False
    
    # 初始化混合精度训练
    scaler = GradScaler()
    print(f"⚡ 已启用混合精度训练 (AMP)")

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
    print("🏗️ 初始化 DSTCarbonFormer 模型...")
    # 确保通道数正确: 辅助数据9通道, 主数据1通道
    model = DSTCarbonFormer(aux_c=9, main_c=1, dim=64).to(device)
    
    print("⚖️ 初始化损失函数 (HybridLoss)...")
    # alpha=MSE, beta=SSIM, gamma=Grad, delta=FFT, eta=TV
    criterion = HybridLoss(alpha=1.0, beta=0.1, gamma=0.1, delta=0.05, eta=0.1).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
    
    # ----------------------------------------
    # 4. 断点续训 (Resume)
    # ----------------------------------------
    start_epoch = 1
    best_loss = float('inf')
    early_stop_counter = 0 
    
    if CONFIG.get('resume', False):
        latest_ckpt = get_latest_checkpoint(CONFIG['save_dir'])
        if latest_ckpt:
            print(f"🔄 发现检查点: {latest_ckpt}，正在恢复...")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            early_stop_counter = checkpoint.get('early_stop_counter', 0)
            print(f"✅ 恢复成功! 从 Epoch {start_epoch} 继续")
        else:
            print("⚠️ 配置要求 Resume 但未找到文件，将重新开始。")
    
    # ----------------------------------------
    # 5. 训练主循环
    # ----------------------------------------
    print(f"\n🚀 开始训练 | 总轮数: {CONFIG['epochs']} | 早停耐心: {CONFIG['patience']}")
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
            
            # 混合精度前向传播
            with autocast():
                pred = model(aux, main)
                # 计算损失 (传入 main 用于 FFT 约束)
                loss = criterion(pred, target, input_main=main)
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪防止爆炸
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # 计算简单的 MAE 用于进度条展示 (还原到真实值)
            with torch.no_grad():
                # 假设使用了 log1p 归一化: real = exp(x * factor) - 1
                pred_real = torch.expm1(pred.detach() * NORM_FACTOR).clamp(min=0)
                target_real = torch.expm1(target * NORM_FACTOR).clamp(min=0)
                batch_mae = torch.abs(pred_real - target_real).mean().item()

            loop.set_postfix(loss=f"{loss.item():.4f}", mae=f"{batch_mae:.2f}")
            
        avg_train_loss = train_loss / len(train_dl)
        
        # --- 验证阶段 ---
        model.eval()
        val_loss = 0
        total_real_mae = 0 
        
        with torch.no_grad():
            for aux, main, target in val_dl:
                aux = aux.to(device)
                main = main.to(device)
                target = target.to(device)
                
                with autocast():
                    pred = model(aux, main)
                    val_loss += criterion(pred, target, input_main=main).item()
                    
                    pred_real = torch.expm1(pred * NORM_FACTOR).clamp(min=0)
                    target_real = torch.expm1(target * NORM_FACTOR).clamp(min=0)
                    total_real_mae += torch.abs(pred_real - target_real).mean().item()
        
        avg_val_loss = val_loss / len(val_dl)
        avg_real_mae = total_real_mae / len(val_dl)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"   📊 Summary: Train_Loss={avg_train_loss:.5f} | Val_Loss={avg_val_loss:.5f} | 🌍 MAE={avg_real_mae:.3f} | LR={current_lr:.2e}")
        
        # ----------------------------------------
        # 6. 保存与早停逻辑
        # ----------------------------------------
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'early_stop_counter': early_stop_counter
        }
        
        # 保存最新的检查点 (latest.pth) 用于方便 resume
        torch.save(checkpoint_dict, os.path.join(CONFIG['save_dir'], "latest.pth"))

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stop_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], "best_model.pth"))
            # 同时也保存一份完整的 checkpoint 以防万一
            torch.save(checkpoint_dict, os.path.join(CONFIG['save_dir'], "best_checkpoint.pth"))
            print(f"   🏆 最佳模型已更新!")
        else:
            early_stop_counter += 1
            print(f"   ⏳ Loss 未下降 ({early_stop_counter}/{CONFIG['patience']})")
        
        # 定期存档
        if epoch % CONFIG['save_freq'] == 0:
            torch.save(checkpoint_dict, os.path.join(CONFIG['save_dir'], f"epoch_{epoch}.pth"))
            
        if early_stop_counter >= CONFIG['patience']:
            print(f"\n🛑 触发早停机制! 训练提前结束。")
            break
            
        scheduler.step()

    print(f"\n🏁 训练全部完成！总耗时: {(time.time()-total_start)/60:.2f} 分钟")

if __name__ == "__main__":
    train()