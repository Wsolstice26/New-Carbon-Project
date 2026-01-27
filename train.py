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
import glob

# 导入你的模块
from data.dataset import DualStreamDataset
from models.network import DSTCarbonFormer
from models.losses import HybridLoss
from config import CONFIG

# 定义还原真实值所需的参数
NORM_FACTOR = 11.0

def get_latest_checkpoint(save_dir):
    """辅助函数：找到目录里最新的 epoch_*.pth 文件"""
    files = glob.glob(os.path.join(save_dir, "epoch_*.pth"))
    if not files:
        return None
    # 按修改时间排序，找最新的
    latest_file = max(files, key=os.path.getmtime)
    return latest_file
# 🔥🔥🔥 加这一行！运行程序时看第一行输出！
print(f"\n======== DEBUG: 当前 Worker 数 = {CONFIG['num_workers']} ========\n")

def train():
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 使用设备: {device}")
    
    # 2. 稳定性设置
    torch.backends.cudnn.benchmark = True # Batch=32时开启这个会快很多
    # torch.backends.cudnn.deterministic = True # 追求速度可以关掉这个
    
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
    
    # 4. 初始化模型
    print("🏗️ 初始化模型...")
    model = DSTCarbonFormer(aux_c=9, main_c=1).to(device)
    
    # 5. 优化器与损失
    print("⚖️ 初始化损失函数与优化器...")
    criterion = HybridLoss(alpha=1.0, beta=0.2, gamma=0.2, delta=1.0, eta=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    # ==========================================
    # 🔄 断点续训逻辑 (Resume)
    # ==========================================
    start_epoch = 1
    best_loss = float('inf')
    early_stop_counter = 0 # 早停计数器
    
    if CONFIG.get('resume', False):
        latest_ckpt = get_latest_checkpoint(CONFIG['save_dir'])
        if latest_ckpt:
            print(f"🔄 发现检查点: {latest_ckpt}，正在恢复训练...")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            
            # 恢复模型和优化器状态
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 恢复训练进度
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            early_stop_counter = checkpoint.get('early_stop_counter', 0)
            
            print(f"✅ 成功恢复！从第 {start_epoch} 轮继续 (最佳 Loss: {best_loss:.5f})")
        else:
            print("⚠️ 未找到检查点，将从头开始训练。")
    
    print(f"🚀 开始训练! | 目标 Epochs: {CONFIG['epochs']} | 早停耐心: {CONFIG['patience']}")
    start_time = time.time()
    
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
                # 使用更新后的 HybridLoss (支持 weight_map 和 TVLoss)
                loss = criterion(pred, target, input_main=main)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # 实时显示 MAE (仅供参考)
            with torch.no_grad():
                pred_real = torch.expm1(pred.detach() * NORM_FACTOR).clamp(min=0)
                target_real = torch.expm1(target * NORM_FACTOR).clamp(min=0)
                batch_mae = torch.abs(pred_real - target_real).mean().item()

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
                    
                    pred_real = torch.expm1(pred * NORM_FACTOR).clamp(min=0)
                    target_real = torch.expm1(target * NORM_FACTOR).clamp(min=0)
                    total_real_mae += torch.abs(pred_real - target_real).mean().item()
        
        avg_val_loss = val_loss / len(val_dl)
        avg_real_mae = total_real_mae / len(val_dl)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"   📊 Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | 🌍 Real MAE: {avg_real_mae:.2f} (吨) | LR: {current_lr:.2e}")
        
        # ==========================================
        # 💾 保存机制 (含断点信息)
        # ==========================================
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'early_stop_counter': early_stop_counter
        }
        
        # 1. 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stop_counter = 0 # Loss 创新低，重置早停计数器
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], "best_model.pth"))
            print(f"   🏆 最佳模型已更新 (Loss: {best_loss:.5f})")
        else:
            early_stop_counter += 1 # Loss 没降，计数器+1
            print(f"   ⏳ Loss 未下降 ({early_stop_counter}/{CONFIG['patience']})")
            
        # 2. 定期保存检查点 (用于续训)
        if epoch % CONFIG['save_freq'] == 0:
            save_path = os.path.join(CONFIG['save_dir'], f"epoch_{epoch}.pth")
            torch.save(checkpoint_dict, save_path)
            # 同时更新一个 latest.pth，确保下次一定能找到最新的
            torch.save(checkpoint_dict, os.path.join(CONFIG['save_dir'], "latest.pth"))
            
        # ==========================================
        # 🛑 早停判断
        # ==========================================
        if early_stop_counter >= CONFIG['patience']:
            print(f"\n🛑 触发早停机制！验证集 Loss 连续 {CONFIG['patience']} 轮未下降。")
            print("训练提前结束。")
            break
            
        scheduler.step()

    print(f"\n🏁 训练结束！总耗时: {(time.time()-start_time)/60:.2f} 分钟")

if __name__ == "__main__":
    train()