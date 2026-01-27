import os

# ==========================================
# 🔇 1. 核心设置：让 MIOpen 闭嘴
# ==========================================
os.environ['MIOPEN_LOG_LEVEL'] = '3'
os.environ['MIOPEN_ENABLE_LOGGING'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# 导入你的模块
from data.dataset import DualStreamDataset
from models.network import DSTCarbonFormer
from models.losses import HybridLoss
from config import CONFIG

# 定义还原真实值所需的参数 (必须与 dataset.py 中的 NORM_MAIN_LOG 保持一致)
NORM_FACTOR = 11.0

def train():
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 使用设备: {device}")
    
    # ==========================================
    # 🛑 2. 稳定性设置 (防卡死)
    # ==========================================
    # 禁止过度寻优，防止 Windows TDR 杀死进程
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True
    
    # 启用混合精度 (AMP)
    scaler = torch.amp.GradScaler('cuda')
    print(f"⚡ 已启用混合精度训练 (Batch Size = {CONFIG['batch_size']})")

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # 3. 准备数据
    print(f"📦 加载数据...")
    train_ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'train')
    val_ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'val')
    
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                          num_workers=CONFIG['num_workers'], pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                        num_workers=CONFIG['num_workers'], pin_memory=True)
    
    # 4. 初始化模型 (含 FFT 硬约束层)
    print("🏗️ 初始化 SEN2SR 增强版模型...")
    model = DSTCarbonFormer(aux_c=9, main_c=1).to(device)
    
    # 5. 优化器与损失
    print("⚖️ 使用 SEN2SR 物理一致性损失函数...")
    criterion = HybridLoss(alpha=1.0, beta=0.1, gamma=0.05, delta=1.0).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    best_loss = float('inf')
    
    print(f"🚀 开始训练! (SEN2SR Mode) | 反归一化因子: {NORM_FACTOR}")
    start_time = time.time()
    
    for epoch in range(1, CONFIG['epochs']+1):
        model.train()
        train_loss = 0
        loop = tqdm(train_dl, desc=f"Ep {epoch}/{CONFIG['epochs']}")
        
        for aux, main, target in loop:
            aux = aux.to(device, non_blocking=True)
            main = main.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # 前向传播
                pred = model(aux, main)
                # 计算损失
                loss = criterion(pred, target, input_main=main)
            
            # 反向传播与梯度裁剪
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

            # ==========================================
            # 🔥 实时计算并显示真实误差 (MAE 吨)
            # ==========================================
            with torch.no_grad():
                # 1. 还原到真实物理值 (反Log + 反归一化)
                # clamp(min=0) 保证不出现负数碳排放
                pred_real = torch.expm1(pred.detach() * NORM_FACTOR).clamp(min=0)
                target_real = torch.expm1(target * NORM_FACTOR).clamp(min=0)
                
                # 2. 计算当前 Batch 的平均误差 (吨)
                batch_mae = torch.abs(pred_real - target_real).mean().item()

            # 更新进度条：同时显示 loss 和 mae (吨)
            loop.set_postfix(loss=loss.item(), mae=f"{batch_mae:.1f}")
            
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
                
                with torch.amp.autocast('cuda'):
                    pred = model(aux, main)
                    val_loss += criterion(pred, target, input_main=main).item()
                    
                    # 验证集也计算真实误差
                    pred_real = torch.expm1(pred * NORM_FACTOR).clamp(min=0)
                    target_real = torch.expm1(target * NORM_FACTOR).clamp(min=0)
                    batch_mae = torch.abs(pred_real - target_real).mean().item()
                    total_real_mae += batch_mae
        
        avg_val_loss = val_loss / len(val_dl)
        avg_real_mae = total_real_mae / len(val_dl)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"   📊 Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | 🌍 Real MAE: {avg_real_mae:.2f} (吨) | LR: {current_lr:.2e}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], "best_model.pth"))
            print(f"   🏆 最佳模型已更新 (Loss: {best_loss:.5f})")
            
        if epoch % CONFIG['save_freq'] == 0:
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], f"epoch_{epoch}.pth"))
            
        scheduler.step()

    print(f"\n🏁 训练结束！总耗时: {(time.time()-start_time)/60:.2f} 分钟")

if __name__ == "__main__":
    train()