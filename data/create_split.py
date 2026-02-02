import json
import random
import os
import numpy as np

# ==========================================
# ⚙️ 1. 配置区域
# ==========================================
# 之前 preprocess_data.py 扫描出的有效位置数
NUM_TOTAL_PATCHES = 429 

# 🔥 [修正] 数据实际所在的文件夹 (虽然名字叫 120, 但里面其实是 160 的切片)
DATA_DIR = "/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_120"

# 索引配置文件输出路径
OUTPUT_JSON = "/home/wdc/Carbon-Emission-Super-Resolution/Configs/split_config.json"

# 划分比例 8:1:1
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

def create_indices():
    print(f"🚀 开始生成样本索引 (预期位置数: {NUM_TOTAL_PATCHES})...")
    
    # 检查文件夹是否存在
    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误: 找不到目录 {DATA_DIR}")
        return

    # 尝试找到第一个 X_*.npy 文件来验证数量
    import glob
    x_files = glob.glob(os.path.join(DATA_DIR, "X_*.npy"))
    if not x_files:
        print(f"❌ 错误: 在 {DATA_DIR} 下找不到任何 X_*.npy 文件！")
        return
    
    # 使用找到的第一个文件进行验证
    sample_file = x_files[0]
    print(f"🔍 正在通过文件验证样本数: {os.path.basename(sample_file)}")
    
    try:
        temp_data = np.load(sample_file)
        actual_count = temp_data.shape[0]
        print(f"📊 文件内实际样本数: {actual_count}")
    except Exception as e:
        print(f"❌ 读取 NPY 文件失败: {e}")
        return

    # 以实际探测到的数量为准
    count = actual_count
    
    # --- 核心划分逻辑 ---
    indices = list(range(count))
    
    # 随机打乱 (固定种子 42)
    random.seed(42)
    random.shuffle(indices)
    
    # 计算切分点
    train_end = int(count * TRAIN_RATIO)
    val_end = train_end + int(count * VAL_RATIO)
    
    # 执行切分
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # 构建配置字典
    split_dict = {
        "metadata": {
            "patch_size": 160,
            "actual_data_dir": DATA_DIR, # 记录真实路径
            "total_locations": count
        },
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices
    }
    
    # 保存为 JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(split_dict, f, indent=4)
    
    print(f"\n✅ 索引文件已成功生成: {OUTPUT_JSON}")
    print(f"📊 划分结果:")
    print(f"   [训练集]: {len(train_indices)} 个位置")
    print(f"   [验证集]: {len(val_indices)} 个位置")
    print(f"   [测试集]: {len(test_indices)} 个位置")
    print(f"\n👉 下一步: 请修改 data/dataset.py 确保它也读取正确的路径。")

if __name__ == "__main__":
    create_indices()