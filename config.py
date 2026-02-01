import os
import json

# ============================================================
# 🎛️ 实验参数控制台
# ============================================================
# 根据刚才的测试，这是目前最稳、最快的配置 (Config A)
PATCH_SIZE = 120    # 图像尺寸
DIM = 64            # 模型通道数 (如果想更高精度，可以改为 64)
BATCH_SIZE = 24     # 批次大小 (如果显存够大，可以改为 24)
TIME_WINDOW = 3     # 时间窗口

# 🏷️ 实验标签
# 修改为 "Final_Optimized" 以便区分，代表这是修复了所有 bug 的完全体
TAG = "Final_Optimized"  

# ============================================================
# 📂 自动路径生成系统 (保持不变)
# ============================================================
PROJECT_ROOT = "/home/wdc/Carbon-Emission-Super-Resolution"

# 1. 自动匹配数据集文件夹
# 请确保 /home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_120 存在
DATA_DIR = os.path.join(PROJECT_ROOT, "data", f"Train_Data_Yearly_{PATCH_SIZE}")

# 2. 自动生成保存路径 (Checkpoints)
exp_name = f"Run_Size{PATCH_SIZE}_Dim{DIM}_Batch{BATCH_SIZE}"
if TAG:
    exp_name += f"_{TAG}"

SAVE_DIR = os.path.join(PROJECT_ROOT, "Checkpoints", exp_name)

# ============================================================
# ⚙️ 最终配置字典
# ============================================================
CONFIG = {
    "project_root": PROJECT_ROOT,
    "data_dir": DATA_DIR,
    "save_dir": SAVE_DIR,
    "split_config": os.path.join(PROJECT_ROOT, "Configs", "split_config.json"),

    # 训练参数
    "patch_size": PATCH_SIZE,
    "dim": DIM,
    "batch_size": BATCH_SIZE,
    "time_window": TIME_WINDOW,
    
    "consistency_scale": 4,
    "epochs": 200,
    "num_workers": 6,
    
    "lr": 2e-4, 
    "patience": 20,
    
    # 🚨【关键修改】设为 False！
    # 因为我们换了网络结构(Depthwise Conv)，旧权重的 shape 对不上，不能加载。
    # 等跑完这一个 Epoch 生成了新的 latest.pth 后，再改回 True。
    "resume": False,    
    
    "seed": 42,
    "norm_factor": 11.0,
    "device": "cuda"
}

# 自动创建目录
os.makedirs(SAVE_DIR, exist_ok=True)

# 保存配置备份
config_save_path = os.path.join(SAVE_DIR, "experiment_config.json")
with open(config_save_path, 'w') as f:
    json.dump({k: v for k, v in CONFIG.items() if isinstance(v, (str, int, float, bool))}, f, indent=4)

print(f"✅ 配置已加载 | 实验目录: {exp_name}")
print(f"⚠️ 注意: Resume 已关闭，将从头开始训练新架构模型")