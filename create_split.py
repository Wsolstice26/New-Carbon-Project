import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 导入进度条库
import time

# ==========================================
# ⚙️ 配置：数据在哪里
# ==========================================
DATA_DIR = r"E:\superResulotion\Train_Data_Yearly_Coords"
CONFIG_DIR = r"E:\superResulotion\Configs"

# ==========================================
# 🛠️ 生成逻辑
# ==========================================
def generate_split_config():
    print(f"🚀 准备开始处理...")
    
    # 定义总步骤数，用来显示进度条
    steps = 4
    
    with tqdm(total=steps, desc="正在初始化", unit="step") as pbar:
        
        # --- 步骤 1: 检查文件 ---
        pbar.set_description("📂 步骤 1/4: 扫描文件头信息")
        ref_file = os.path.join(DATA_DIR, "X_2014.npy")
        
        if not os.path.exists(ref_file):
            print(f"\n❌ 错误：找不到文件 {ref_file}")
            return
            
        # mmap_mode='r' 极速模式，只读取形状，不加载数据
        try:
            data = np.load(ref_file, mmap_mode='r')
            total_patches = data.shape[0]
            # 模拟一点点延时让进度条能被肉眼看到（可选）
            # time.sleep(0.5) 
        except Exception as e:
            print(f"\n❌ 读取文件出错: {e}")
            return
            
        pbar.update(1) # 完成第1步

        # --- 步骤 2: 生成索引 ---
        pbar.set_description(f"🔢 步骤 2/4: 生成索引 (共 {total_patches} 个)")
        all_indices = np.arange(total_patches)
        pbar.update(1) # 完成第2步

        # --- 步骤 3: 随机划分 ---
        pbar.set_description("✂️ 步骤 3/4: 正在进行随机划分 (8:1:1)")
        
        # 80% 训练
        train_idx, temp_idx = train_test_split(
            all_indices, train_size=0.8, random_state=2026, shuffle=True
        )
        # 剩下一半一半 (10% 验证, 10% 测试)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=2026, shuffle=True
        )
        pbar.update(1) # 完成第3步

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
            
        pbar.update(1) # 完成第4步

    # 打印最终报告
    print("\n✅ 处理完成！")
    print(f"📊 数据集概览:")
    print(f"   - 总样本数: {total_patches}")
    print(f"   - 训练集 (Train): {len(train_idx)} 个")
    print(f"   - 验证集 (Val)  : {len(val_idx)} 个")
    print(f"   - 测试集 (Test) : {len(test_idx)} 个")
    print(f"📁 配置文件已保存至: {out_path}")

if __name__ == "__main__":
    generate_split_config()