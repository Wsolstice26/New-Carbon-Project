import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import torch
import gc

def analyze_with_gpu():
    # -------------------------------------------------------
    # 1. 路径与设备配置
    # -------------------------------------------------------
    data_path = "/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_Coords"
    files = glob.glob(os.path.join(data_path, "*.npy"))
    
    if not files:
        print(f"❌ 未在 {data_path} 找到 .npy 文件，请检查路径。")
        return

    # 检查是否有 GPU (ROCm/CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device} (AMD ROCm / NVIDIA CUDA)")

    # -------------------------------------------------------
    # 2. 第一阶段：快速计数 (Pre-scan)
    # -------------------------------------------------------
    print(f"🔍 [Phase 1] 正在预扫描 {len(files)} 份文件以计算内存需求...")
    
    total_valid_pixels = 0
    total_pixels_count = 0
    zero_count_global = 0

    for i, f in enumerate(files):
        # 只是读取形状和简单的统计，不需要全部加载进显存，或者分块处理
        # 这里为了简单，还是加载 numpy，转 tensor 处理会很快
        try:
            # 即使是 Numpy Load 也可能由 CPU 瓶颈，但这里很难优化 IO，主要优化计算
            raw_data = np.load(f)
            total_pixels_count += raw_data.size
            
            # 转为 Tensor 扔进 GPU
            # 使用 float32 节省显存
            tensor_data = torch.from_numpy(raw_data).to(device, dtype=torch.float32)
            
            # GPU 极速处理 NaN
            tensor_data = torch.nan_to_num(tensor_data, nan=0.0)
            
            # GPU 逻辑筛选
            mask = tensor_data > 1e-6
            valid_count = mask.sum().item() # 获取数量
            
            total_valid_pixels += valid_count
            zero_count_global += (raw_data.size - valid_count)
            
            # 释放显存
            del tensor_data, mask, raw_data
            
            # 打印进度
            print(f"\r  Scanned {i+1}/{len(files)} | Found {total_valid_pixels} valid pixels", end="")
            
        except Exception as e:
            print(f"\n⚠️ 读取文件 {f} 出错: {e}")

    print(f"\n✅ 预扫描完成。需要存储 {total_valid_pixels} 个浮点数。")
    
    # -------------------------------------------------------
    # 3. 内存分配 (Allocation)
    # -------------------------------------------------------
    # float32 每个占 4 字节。7.6亿 * 4 ≈ 2.8 GB。这完全可以塞进内存。
    # 之前报错是因为 Python List 的额外开销。
    try:
        big_array = np.zeros(total_valid_pixels, dtype=np.float32)
        print(f"💾 已分配主机内存: {big_array.nbytes / 1024**3:.2f} GB")
    except MemoryError:
        print("❌ 内存不足！无法一次性加载所有非零数据。请考虑使用流式统计算法。")
        return

    # -------------------------------------------------------
    # 4. 第二阶段：GPU 填充 (Fill)
    # -------------------------------------------------------
    print(f"📥 [Phase 2] 正在利用 GPU 批量清洗并填入数据...")
    
    current_idx = 0
    
    for i, f in enumerate(files):
        raw_data = np.load(f)
        
        # CPU -> GPU
        tensor_data = torch.from_numpy(raw_data).to(device, dtype=torch.float32)
        
        # GPU 计算
        tensor_data = torch.nan_to_num(tensor_data, nan=0.0)
        valid_pixels = tensor_data[tensor_data > 1e-6] # Boolean Masking
        
        # GPU -> CPU (只传回有效数据)
        # 这一步将清洗好的数据块拉回 CPU
        valid_chunk = valid_pixels.cpu().numpy()
        
        # 填入大数组
        chunk_len = len(valid_chunk)
        big_array[current_idx : current_idx + chunk_len] = valid_chunk
        current_idx += chunk_len
        
        # 清理显存
        del tensor_data, valid_pixels, valid_chunk, raw_data
        # torch.cuda.empty_cache() # AMD ROCm 上频繁调用可能会慢，一般不需要
        
        if i % 5 == 0:
             print(f"\r  Processed {i+1}/{len(files)} | Filled: {current_idx/total_valid_pixels*100:.1f}%", end="")

    print("\n✅ 数据装载完成。开始统计...")

    # -------------------------------------------------------
    # 5. 统计分析 (Statistics)
    # -------------------------------------------------------
    sparsity = (zero_count_global / total_pixels_count) * 100
    
    print("\n===== 🌍 碳排放数据全量体检报告 =====")
    print(f"总像素数: {total_pixels_count}")
    print(f"稀疏度 (零值占比): {sparsity:.2f}%")
    print(f"有值像素数: {len(big_array)}")
    
    if len(big_array) > 0:
        # 计算分位数 (NumPy 在 CPU 上算这个很快)
        quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        print("⏳ 正在计算分位数...")
        values = np.quantile(big_array, quantiles)
        
        print("-" * 35)
        print("📊 非零像素数值分布 (吨/像素):")
        for q, v in zip(quantiles, values):
            print(f"  {int(q*100):2d}% 的像素值低于: {v:.6f}")
        
        print("-" * 35)
        print(f"最大值: {np.max(big_array):.6f}")
        print(f"平均值: {np.mean(big_array):.6f}")
        print(f"标准差: {np.std(big_array):.6f}")
    
        # 绘图
        print("🎨 正在绘制直方图...")
        plt.figure(figsize=(10, 6))
        # 这里的 bins 可以设大一点，因为数据量大
        plt.hist(big_array, bins=200, color='salmon', edgecolor='black', log=True)
        plt.title("Frequency Distribution of Carbon Emissions (Non-zero, Log-Scale Y)")
        plt.xlabel("Emission Value (Tons)")
        plt.ylabel("Frequency (Log Scale)")
        plt.grid(axis='y', alpha=0.3)
        plt.savefig("distribution_full_gpu.png")
        print("\n📈 分布直方图已保存至: distribution_full_gpu.png")
    else:
        print("⚠️ 未发现非零像素数据。")

if __name__ == "__main__":
    analyze_with_gpu()