import numpy as np
import os
from tqdm import tqdm
from config import CONFIG

# 定义波段名称 (根据您的描述猜测，仅供显示用)
# 辅助流 9 通道通常是: [0-NTL, 1-Road, 2-Water, 3-Build, 4-N01, 5-NDVI, ...]
# 这里我们暂时用 Band 0 ~ Band 8 代替
AUX_CHANNEL_NAMES = [f"Aux-Band_{i}" for i in range(9)]

def analyze_granular_stats():
    data_dir = CONFIG['data_dir']
    years = range(2014, 2024) # 2014-2023
    
    print(f"🕵️‍♀️ 开始深度扫描: {data_dir}")
    print("="*60)
    print(f"{'年份':<6} | {'波段名称':<12} | {'最小值':>12} | {'最大值':>12} | {'均值':>12}")
    print("="*60)

    global_max_main = -1
    global_max_aux = -1

    for year in years:
        x_path = os.path.join(data_dir, f"X_{year}.npy")
        y_path = os.path.join(data_dir, f"Y_{year}.npy")
        
        # --- 1. 分析 Aux (辅助数据) ---
        if os.path.exists(x_path):
            try:
                # mmap_mode='r' 防止内存溢出，像查字典一样读
                # shape: [N, 9, 128, 128]
                x_data = np.load(x_path, mmap_mode='r')
                
                # 循环 9 个通道
                for c in range(9):
                    # 读取该通道的所有数据 (自动展平)
                    # 注意：如果内存不够，这里可能会卡一下，但通常没事
                    band_data = x_data[:, c, :, :]
                    
                    b_min = float(np.min(band_data))
                    b_max = float(np.max(band_data))
                    b_mean = float(np.mean(band_data))
                    
                    if b_max > global_max_aux: global_max_aux = b_max
                    
                    print(f"{year:<6} | {AUX_CHANNEL_NAMES[c]:<12} | {b_min:>12.4f} | {b_max:>12.4f} | {b_mean:>12.4f}")
            except Exception as e:
                print(f"{year:<6} | Aux 读取失败: {e}")
        else:
            print(f"{year:<6} | ❌ X_{year}.npy 不存在")

        print("-" * 60)

        # --- 2. 分析 Main (碳排放标签) ---
        if os.path.exists(y_path):
            try:
                # shape: [N, 1, 128, 128]
                y_data = np.load(y_path, mmap_mode='r')
                
                b_min = float(np.min(y_data))
                b_max = float(np.max(y_data))
                b_mean = float(np.mean(y_data))
                
                if b_max > global_max_main: global_max_main = b_max
                
                print(f"{year:<6} | {'Main-Carbon':<12} | {b_min:>12.4f} | \033[91m{b_max:>12.4f}\033[0m | {b_mean:>12.4f}")
            except Exception as e:
                print(f"{year:<6} | Main 读取失败: {e}")
        else:
            print(f"{year:<6} | ❌ Y_{year}.npy 不存在")
            
        print("="*60)

    print("\n📊 最终诊断结论:")
    print(f"1. 碳排放 (Main) 全局最大值: {global_max_main:.4f}")
    print(f"2. 辅助数据 (Aux) 全局最大值: {global_max_aux:.4f}")
    
    if global_max_main > 100:
        suggested_norm = 10 ** np.ceil(np.log10(global_max_main)) # 比如 34480 -> 100000
        print(f"\n💡 解决方案: 必须归一化！")
        print(f"   建议在 dataset.py 中将 Main 除以 {suggested_norm} 或 {global_max_main:.0f}")

if __name__ == "__main__":
    analyze_granular_stats()