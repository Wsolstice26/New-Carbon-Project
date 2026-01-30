import os
import json

# ============================================================
# 🎛️ 实验参数控制台 (以后改这里就行)
# ============================================================
PATCH_SIZE = 120    # 图像尺寸 (120 或 160)
DIM = 48            # 模型通道数 (32, 48, 64)
BATCH_SIZE = 24     # 批次大小
TIME_WINDOW = 3     # 时间窗口

# 🏷️ 给这次实验加个备注 (比如: "Test_Mamba", "No_Lock" 等)
# 如果留空，文件名就只包含参数
TAG = "Mamba_Fix"  

# ============================================================
# 📂 自动路径生成系统 (不要改下面)
# ============================================================
PROJECT_ROOT = "/home/wdc/Carbon-Emission-Super-Resolution"

# 1. 自动匹配数据集文件夹
# 比如: .../data/Train_Data_Yearly_120
DATA_DIR = os.path.join(PROJECT_ROOT, "data", f"Train_Data_Yearly_{PATCH_SIZE}")

# 2. 自动生成保存路径 (Checkpoints)
# 格式: Run_Size120_Dim48_Batch16_Mamba_Fix
exp_name = f"Run_Size{PATCH_SIZE}_Dim{DIM}_Batch{BATCH_SIZE}"
if TAG:
    exp_name += f"_{TAG}"

SAVE_DIR = os.path.join(PROJECT_ROOT, "Checkpoints", exp_name)

# ============================================================
# ⚙️ 最终配置字典
# ============================================================
CONFIG = {
    "project_root": PROJECT_ROOT,
    
    # 自动填入上面生成的路径
    "data_dir": DATA_DIR,
    "save_dir": SAVE_DIR,
    
    # split_config 建议也跟尺寸绑定，防止混用
    "split_config": os.path.join(PROJECT_ROOT, "Configs", "split_config.json"),

    # 训练参数
    "patch_size": PATCH_SIZE,
    "dim": DIM,
    "batch_size": BATCH_SIZE,
    "time_window": TIME_WINDOW,
    
    "consistency_scale": 4,
    "epochs": 200,
    "num_workers": 6,   # 保持多进程
    
    "lr": 2e-4, 
    "patience": 20,
    "resume": False,    # 如果要恢复训练，把这里改成 True
    "seed": 42,
    "norm_factor": 11.0,
    "device": "cuda"
}

# 自动创建目录
os.makedirs(SAVE_DIR, exist_ok=True)

# 💡 [新增] 每次运行时，把当前的配置保存到文件夹里
# 这样以后你打开文件夹，看这个 json 就知道当时用了什么参数！
config_save_path = os.path.join(SAVE_DIR, "experiment_config.json")
with open(config_save_path, 'w') as f:
    # 过滤掉无法序列化的对象，只保存参数
    json.dump({k: v for k, v in CONFIG.items() if isinstance(v, (str, int, float, bool))}, f, indent=4)

print(f"✅ 配置已加载 | 实验目录: {exp_name}")