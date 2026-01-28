import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

# ==========================================
# ⚙️ 配置：修改为 Linux 真实路径
# ==========================================
# 你的真实数据存放位置
DATA_DIR = "/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_Coords" 
# 配置文件生成的目录
CONFIG_DIR = "/home/wdc/Carbon-Emission-Super-Resolution/Configs"

def generate_split_config():
    print(f"🚀 准备开始处理...")
    steps = 4
    
    with tqdm(total=steps, desc="正在初始化", unit="step") as pbar:
        # --- 步骤 1: 扫描文件 ---
        pbar.set_description("📂 步骤 1/4: 扫描文件头信息")
        
        # 你的文件夹里应该有类似 X_2014.npy, X_2015.npy 等文件
        # 我们用其中一个来读取总数据量 (total_patches)
        # 如果你的文件名不同，请检查 DATA_DIR 下的文件
        ref_file = os.path.join(DATA_DIR, "X_2014.npy")
        
        if not os.path.exists(ref_file):
            print(f"\n❌ 错误：在 {DATA_DIR} 中找不到 X_2014.npy。")
            print("请确认你的 .npy 文件名是否正确 (例如是否叫 X_2014.npy 或其他年份)。")
            # 尝试自动寻找一个 .npy 文件作为替代
            npy_files = [f for f in os.listdir(DATA_DIR) if f.startswith("X_") and f.endswith(".npy")]
            if npy_files:
                ref_file = os.path.join(DATA_DIR, npy_files[0])
                print(f"🔄 自动切换到: {npy_files[0]}")
            else:
                return
            
        try:
            # mmap_mode='r' 允许读取大文件而不加载进内存
            data = np.load(ref_file, mmap_mode='r')
            total_patches = data.shape[0]
            print(f"   📊 检测到每个文件包含 {total_patches} 个样本")
        except Exception as e:
            print(f"\n❌ 读取文件出错: {e}")
            return
            
        pbar.update(1)

        # --- 步骤 2: 生成索引 ---
        pbar.set_description(f"🔢 步骤 2/4: 生成索引 (共 {total_patches} 个)")
        all_indices = np.arange(total_patches)
        pbar.update(1)

        # --- 步骤 3: 随机划分 ---
        pbar.set_description("✂️ 步骤 3/4: 正在进行随机划分 (8:1:1)")
        train_idx, temp_idx = train_test_split(
            all_indices, train_size=0.8, random_state=2026, shuffle=True
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=2026, shuffle=True
        )
        pbar.update(1)

        # --- 步骤 4: 保存结果 ---
        pbar.set_description("💾 步骤 4/4: 正在写入配置文件")
        
        config_data = {
            "total_patches": int(total_patches),
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "test_indices": test_idx.tolist()
        }

        if not os.path.exists(CONFIG_DIR):
            os.makedirs(CONFIG_DIR, exist_ok=True)
            
        out_path = os.path.join(CONFIG_DIR, "split_config.json")
        
        with open(out_path, 'w') as f:
            json.dump(config_data, f)
            
        pbar.update(1)

    print(f"\n✅ 处理完成！配置文件已保存至: {out_path}")
    print(f"👉 现在的训练集数量: {len(train_idx)}, 验证集: {len(val_idx)}, 测试集: {len(test_idx)}")

if __name__ == "__main__":
    generate_split_config()