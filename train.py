import os

# ==========================================
# 🛡️ 1. 核心设置：安全模式 (Safe Mode)
# ==========================================
# 强制使用 GEMM 算法 (最稳，绝对不崩，MoE 必备)
os.environ['MIOPEN_DEBUG_CONV_GEMM'] = '1'
# 屏蔽 MIOpen 烦人的警告日志
os.environ['MIOPEN_LOG_LEVEL'] = '2' 
os.environ['MIOPEN_ENABLE_LOGGING'] = '0'
# 指定缓存路径，防止权限问题
os.environ['MIOPEN_USER_DB_PATH'] = './miopen_cache'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
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
from config import CONFIG  # <--- 直接从文件导入配置

NORM_FACTOR = 11.0

def get_latest_checkpoint(save_dir):
    """
    优先寻找 latest.pth (由脚本自动每轮保存)，
    如果找不到，再寻找 epoch_*.pth
    """
    if not os.path.exists(save_dir):
        return None
    
    # 1. 优先找 latest.pth
    latest_path = os.path.join(save_dir, "latest.pth")
    if os.path.exists(latest_path):
        return latest_path

    # 2. 找不到再找历史存档
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
        
        # [MoE 关键设置] 关闭 Benchmark 防止动态形状导致崩溃
        torch.backends.cudnn.benchmark = False 
        torch.backends.cudnn.deterministic = True
        print("🛡️ 安全模式已启动: Benchmark=False, GEMM=ON")
    
    scaler = torch.amp.GradScaler('cuda')
    print(f"⚡ 已启用混合精度训练 (AMP)")

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # ----------------------------------------
    # 2. 数据准备
    # ----------------------------------------
    print(f"📦 加载数据 (Batch Size: {CONFIG['batch_size']})...")
    
    # 双重检查路径
    if not os.path.exists(CONFIG['data_dir']):
        print(f"❌ 错误: 数据路径不存在 -> {CONFIG['data_dir']}")
        print("请检查 config.py 中的路径设置！")
        return

    train_ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'train')
    val_ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'val')
    
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                          num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                        num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    
    # ----------------------------------------
    # 3. 模型与优化器
    # ----------------------------------------
    print("🏗️ 初始化 DSTCarbonFormer 模型 (v1.3)...")
    model = DSTCarbonFormer(aux_c=9, main_c=1, dim=64).to(device)
    
    print("⚖️ 初始化损失函数...")
    criterion = HybridLoss(alpha=1.0, beta=0.1, gamma=0.1, delta=0.05, eta=0.1).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
    
    # ----------------------------------------
    # 4. 断点续训 (Resume)
    # ----------------------------------------
    start_epoch = 1
    best_loss = float('inf')
    early_stop_counter = 0 
    
    if CONFIG['resume']:
        latest_ckpt = get_latest_checkpoint(CONFIG['save_dir'])
        if latest_ckpt:
            print(f"🔄 发现检查点: {latest_ckpt}，正在恢复...")
            try:
                checkpoint = torch.load(latest_ckpt, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_loss = checkpoint.get('best_loss', float('inf'))
                early_stop_counter = checkpoint.get('early_stop_counter', 0)
                print(f"✅ 恢复成功! 从 Epoch {start_epoch} 继续")
            except Exception as e:
                print(f"⚠️ 恢复失败 ({e})，将从头开始训练。")
        else:
            print("⚠️ 未找到检查点，从头开始。")
    
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
            
            try:
                # 混合精度前向传播
                with torch.amp.autocast('cuda'):
                    pred = model(aux, main)
                    loss = criterion(pred, target, input_main=main)
                
                if torch.isnan(loss):
                    print(f"⚠️ 警告: Epoch {epoch} 出现 NaN Loss，跳过此 Batch")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                
                with torch.no_grad():
                    pred_real = torch.expm1(pred.detach() * NORM_FACTOR).clamp(min=0)
                    target_real = torch.expm1(target * NORM_FACTOR).clamp(min=0)
                    batch_mae = torch.abs(pred_real - target_real).mean().item()

                loop.set_postfix(loss=f"{loss.item():.4f}", mae=f"{batch_mae:.2f}")

            except RuntimeError as e:
                if "invalid configuration" in str(e) or "HIP error" in str(e):
                    print(f"\n❌ 严重错误: 显卡驱动异常。建议删除 miopen_cache 并重启。")
                    raise e
                else:
                    raise e
            
        avg_train_loss = train_loss / len(train_dl) if len(train_dl) > 0 else 0
        
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
        
        print(f"   📊 Summary: Train_Loss={avg_train_loss:.5f} | Val_Loss={avg_val_loss:.5f} | 🌍 MAE={avg_real_mae:.3f} | LR={current_lr:.2e}")
        
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'early_stop_counter': early_stop_counter
        }
        
        # 保存最新检查点 (用于 resume)
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
        
        # 定期保存历史存档
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