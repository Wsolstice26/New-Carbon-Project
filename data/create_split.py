import os
import numpy as np
import json
import random

# ================= ⚙️ 配置 =================
# 必须指向你最新的数据文件夹
DATA_DIR = "/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_120"
CONFIG_DIR = "/home/wdc/Carbon-Emission-Super-Resolution/Configs"
os.makedirs(CONFIG_DIR, exist_ok=True)

# 划分比例 (Train / Val / Test)
RATIOS = [0.8, 0.1, 0.1] 

def make_split():
    # 1. 自动检测样本数量
    # 随便读取一年的 Y 文件来查看长度
    y_path = os.path.join(DATA_DIR, "Y_2020.npy")
    
    if not os.path.exists(y_path):
        print(f"❌ 找不到数据文件: {y_path}")
        print("请先运行 make_dataset.py 生成数据！")
        return

    # 只读 header 信息，不加载数据，速度极快
    y_data = np.load(y_path, mmap_mode='r')
    total_samples = y_data.shape[0]
    
    print(f"📊 检测到最新样本总数: {total_samples}")
    
    # 2. 生成索引并打乱
    indices = list(range(total_samples))
    random.seed(42) # 固定种子，保证复现
    random.shuffle(indices)
    
    # 3. 计算切分点
    n_train = int(total_samples * RATIOS[0])
    n_val = int(total_samples * RATIOS[1])
    # 剩下的给 test
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]
    
    print(f"   - 训练集: {len(train_indices)} 个")
    print(f"   - 验证集: {len(val_indices)} 个")
    print(f"   - 测试集: {len(test_indices)} 个")
    
    # 4. 保存配置
    config = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "total_samples": total_samples,
        "note": "Generated for Union-Set (Strict Aligned)"
    }
    
    save_path = os.path.join(CONFIG_DIR, "split_config.json")
    with open(save_path, "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"✅ 新的划分文件已保存: {save_path}")
    print("👉 现在可以重新运行 check_data.py 或 train.py 了！")

if __name__ == "__main__":
    make_split()