import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys

# 导入你的数据集类
from data.dataset import DualStreamDataset
from config import CONFIG

def check_data():
    print("🕵️‍♀️ 开始扫描数据集数值统计...")
    
    # 1. 加载数据集 (只加载训练集)
    try:
        ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'train')
        # num_workers=0 避免多进程打印混乱
        dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    print(f"📦 数据集样本总数: {len(ds)}")
    
    # 初始化统计变量
    global_min_aux = float('inf')
    global_max_aux = float('-inf')
    global_min_main = float('inf')
    global_max_main = float('-inf')
    
    has_nan = False
    has_inf = False
    
    # 2. 循环扫描
    for i, (aux, main, target) in enumerate(tqdm(dl, desc="正在扫描")):
        
        # --- 检查 NaN/Inf ---
        if torch.isnan(aux).any() or torch.isnan(main).any() or torch.isnan(target).any():
            print(f"\n⚠️ 发现 NaN! 在第 {i} 个 Batch")
            has_nan = True
            
        if torch.isinf(aux).any() or torch.isinf(main).any() or torch.isinf(target).any():
            print(f"\n⚠️ 发现 Inf (无穷大)! 在第 {i} 个 Batch")
            has_inf = True

        # --- 统计 Aux (辅助流) 范围 ---
        batch_min_aux = aux.min().item()
        batch_max_aux = aux.max().item()
        if batch_min_aux < global_min_aux: global_min_aux = batch_min_aux
        if batch_max_aux > global_max_aux: global_max_aux = batch_max_aux
        
        # --- 统计 Main (主流/Target) 范围 ---
        batch_min_main = main.min().item()
        batch_max_main = main.max().item()
        if batch_min_main < global_min_main: global_min_main = batch_min_main
        if batch_max_main > global_max_main: global_max_main = batch_max_main

        # --- 实时预警 ---
        # FP16 的最大表示范围大约是 65500
        if batch_max_main > 60000 or batch_max_aux > 60000:
            print(f"\n⚠️ 警告: 第 {i} 个 Batch 数值过大 (>{batch_max_main:.0f})，FP16 可能会溢出！")

    print("\n" + "="*40)
    print("📊 数据集统计报告")
    print("="*40)
    
    print(f"1️⃣ 辅助流 (Aux) 9通道:")
    print(f"   - 最小值: {global_min_aux:.4f}")
    print(f"   - 最大值: {global_max_aux:.4f}")
    
    print(f"\n2️⃣ 主流/标签 (Main/Target) 碳排放:")
    print(f"   - 最小值: {global_min_main:.4f}")
    print(f"   - 最大值: {global_max_main:.4f}")
    
    print("\n3️⃣ 异常检查:")
    if has_nan:
        print("   ❌ 包含 NaN (空值) -> 需要数据清洗")
    else:
        print("   ✅ 无 NaN")
        
    if has_inf:
        print("   ❌ 包含 Inf (无穷大) -> 需要处理异常值")
    else:
        print("   ✅ 无 Inf")
        
    if global_max_main > 100 or global_max_aux > 1000:
        print("\n💡 建议: 数据数值较大，建议进行【归一化】处理。")
        print("   (神经网络通常喜欢 0~1 或 -1~1 之间的输入)")
    else:
        print("\n✅ 数值范围适中，无需归一化。")

if __name__ == "__main__":
    check_data()