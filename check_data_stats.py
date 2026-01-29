import os
import numpy as np
from tqdm import tqdm
import multiprocessing

# ================= 配置 =================
# 你的真实数据路径
DATA_DIR = "/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_Coords"
# ========================================

def check_file(filename):
    """
    检查单个文件的健康状况
    """
    path = os.path.join(DATA_DIR, filename)
    try:
        # mmap_mode='r' 只读取元数据，不占用大量内存，速度极快
        data = np.load(path, mmap_mode='r')
        
        # 1. 检查 NaN (空值)
        if np.isnan(data).any():
            return f"❌ [NaN Found] {filename}"
            
        # 2. 检查 Inf (无穷大)
        if np.isinf(data).any():
            return f"❌ [Inf Found] {filename}"
            
        # 3. 检查极端数值 (比如大于 10000 的碳排放值，通常是异常的)
        # 注意：这里需要加载一部分数据到内存来比较，为了速度我们只抽查
        # 如果数据量巨大，可以只采样 max
        max_val = np.max(data)
        if max_val > 1e5: # 阈值可根据你的业务调整
            return f"⚠️ [Extreme Value] {filename} (Max: {max_val:.2f})"
            
        return None # 文件健康
        
    except Exception as e:
        return f"💀 [Load Failed] {filename} ({str(e)})"

def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ 路径不存在: {DATA_DIR}")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy')]
    print(f"🔍 开始扫描 {len(files)} 个数据文件...")
    
    bad_files = []
    
    # 使用多进程加速扫描
    with multiprocessing.Pool(processes=16) as pool:
        results = list(tqdm(pool.imap(check_file, files), total=len(files)))
    
    print("\n" + "="*40)
    print("📊 扫描报告")
    print("="*40)
    
    for res in results:
        if res:
            print(res)
            bad_files.append(res)
            
    if len(bad_files) == 0:
        print("✅ 完美！所有数据文件都是健康的。")
        print("🤔 如果数据没问题但训练还崩，可能是 DataAugmentation (数据增强) 产生了 NaN。")
    else:
        print(f"\n🚫 发现 {len(bad_files)} 个坏文件。建议删除或修复它们！")

if __name__ == "__main__":
    main()