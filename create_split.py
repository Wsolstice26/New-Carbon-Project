import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

# ==========================================
# ⚙️ 配置：路径必须与 Docker 挂载点一致
# ==========================================
# 你的数据挂载到了 /train_data
DATA_DIR = "/train_data" 
# 配置文件应放在工作区的 Configs 目录下
CONFIG_DIR = "/workspace/Configs"

def generate_split_config():
    print(f"🚀 准备开始处理...")
    steps = 4
    
    with tqdm(total=steps, desc="正在初始化", unit="step") as pbar:
        # --- 步骤 1: 扫描文件 ---
        pbar.set_description("📂 步骤 1/4: 扫描文件头信息")
        # 确保容器能看到这个文件
        ref_file = os.path.join(DATA_DIR, "X_2014.npy")
        
        if not os.path.exists(ref_file):
            print(f"\n❌ 错误：在 {DATA_DIR} 中找不到 X_2014.npy。请确认 Docker 挂载路径。")
            return
            
        try:
            data = np.load(ref_file, mmap_mode='r')
            total_patches = data.shape[0]
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
            os.makedirs(CONFIG_DIR)
            
        out_path = os.path.join(CONFIG_DIR, "split_config.json")
        
        with open(out_path, 'w') as f:
            json.dump(config_data, f)
            
        pbar.update(1)

    print(f"\n✅ 处理完成！配置文件已保存至: {out_path}")

if __name__ == "__main__":
    generate_split_config()