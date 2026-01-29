import numpy as np
import os

# 你的数据路径
FILE_PATH = "/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_Coords/X_2014.npy"

def inspect_file():
    if not os.path.exists(FILE_PATH):
        print("❌ 文件不存在！")
        return

    print(f"🔍 正在加载 {FILE_PATH} (可能需要几秒钟)...")
    # 加载数据
    data = np.load(FILE_PATH)
    
    print(f"📊 数据形状: {data.shape}")
    print(f"   (假设维度顺序为: [样本数, 通道数, 时间, 高, 宽] 或类似)")
    
    # 1. 统计 NaN 总量
    nan_mask = np.isnan(data)
    total_elements = data.size
    total_nans = np.sum(nan_mask)
    nan_ratio = total_nans / total_elements * 100
    
    print("\n" + "="*30)
    print("🏥 诊断报告")
    print("="*30)
    print(f"🔴 NaN 总数: {total_nans}")
    print(f"📉 NaN 占比: {nan_ratio:.4f}%")
    
    if total_nans == 0:
        print("✅ 数据是健康的（奇怪，check_data 说它有毒？）")
        return

    # 2. 定位病灶：是哪个通道 (Channel) 坏了？
    # 假设第 1 个维度是 Sample，第 2 个维度是 Channel
    # 我们检查每个通道的 NaN 情况
    num_channels = data.shape[1]
    print(f"\n🔬 按通道 (Channel) 检查:")
    for c in range(num_channels):
        # 提取该通道的所有数据
        # 假设 shape 是 [N, C, T, H, W]，则取 [:, c, ...]
        channel_data = data[:, c, ...] 
        n_nans = np.isnan(channel_data).sum()
        if n_nans > 0:
            print(f"   ⚠️ Channel {c}: 有 {n_nans} 个 NaN")
    
    # 3. 定位病灶：是哪个样本 (Sample) 坏了？
    num_samples = data.shape[0]
    bad_samples = []
    print(f"\n🔬 按样本 (Sample) 检查:")
    for i in range(num_samples):
        if np.isnan(data[i]).any():
            bad_samples.append(i)
            
    print(f"   ⚠️ 共有 {len(bad_samples)} 个样本包含 NaN")
    if len(bad_samples) < 10:
        print(f"   📍 坏样本索引: {bad_samples}")
    else:
        print(f"   📍 坏样本索引 (前10个): {bad_samples[:10]} ...")

    # 4. 检查 Inf (无穷大)
    inf_count = np.isinf(data).sum()
    if inf_count > 0:
        print(f"\n🔥 警告: 还有 {inf_count} 个 Inf (无穷大)！")

if __name__ == "__main__":
    inspect_file()