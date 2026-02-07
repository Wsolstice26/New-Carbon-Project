import os
import json
import glob
from datetime import datetime

# ============================================================
# 🎛️ 实验参数控制台
# ============================================================
RESUME = True  # 如果你想接着之前的模型跑，改成 True

# 显存优化配置
TARGET_BATCH_SIZE = 32
BATCH_SIZE = 8        
GRAD_ACCUM_STEPS = TARGET_BATCH_SIZE // BATCH_SIZE

PATCH_SIZE = 120
DIM = 64  
TIME_WINDOW = 3

# 🏷️ 实验标签：Linear Mode
TAG = "LinearLoss_WeightedL1_DirectR2"

# ============================================================
# 📂 自动路径生成系统
# ============================================================
PROJECT_ROOT = "/home/wdc/Carbon-Emission-Super-Resolution"

identity_suffix = f"Size{PATCH_SIZE}_Dim{DIM}_EffBatch{TARGET_BATCH_SIZE}"
if TAG:
    identity_suffix += f"_{TAG}"

checkpoints_root = os.path.join(PROJECT_ROOT, "Checkpoints")
os.makedirs(checkpoints_root, exist_ok=True)

if not RESUME:
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = f"Run_{current_time}_{identity_suffix}"
    print(f"🆕 [New Experiment] 创建新实验目录: {exp_name}")
else:
    search_pattern = os.path.join(checkpoints_root, f"Run_*_{identity_suffix}")
    candidates = glob.glob(search_pattern)
    if len(candidates) > 0:
        candidates.sort()
        latest_folder_path = candidates[-1]
        exp_name = os.path.basename(latest_folder_path)
        print(f"🔄 [Resume] 自动定位到最近的实验: {exp_name}")
    else:
        raise FileNotFoundError(f"❌ 无法 Resume：未找到参数匹配的旧实验文件夹。\n搜索模式: {search_pattern}")

SAVE_DIR = os.path.join(checkpoints_root, exp_name)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", f"Train_Data_Yearly_{PATCH_SIZE}")

# ============================================================
# ⚙️ 最终配置字典
# ============================================================
CONFIG = {
    "project_root": PROJECT_ROOT,
    "data_dir": DATA_DIR,
    "save_dir": SAVE_DIR,
    "split_config": os.path.join(PROJECT_ROOT, "Configs", "split_config.json"),

    # Data / Model
    "patch_size": PATCH_SIZE,
    "dim": DIM,
    "batch_size": BATCH_SIZE, 
    "grad_accum_steps": GRAD_ACCUM_STEPS,
    "target_batch_size": TARGET_BATCH_SIZE,
    "time_window": TIME_WINDOW,
    "consistency_scale": 10,
    
    # 深度控制参数
    "num_mamba_layers": 2,  
    "num_res_blocks": 4,    

    # Training
    "epochs": 500,
    "num_workers": 6,
    
    # 🚀 [LR 调整] 线性 Loss 建议从 1e-4 开始
    "lr": 1e-4,                 
    "main_metric": "r2_score",  

    "patience": 50, # 线性 Loss 收敛快，耐心可以给小点

    # Loss 权重 (Log 相关的参数已移除)
    "w_sparse": 1e-3,           
    "w_ent": 1e-3,              
    "ent_mode": "max",
    "target_entropy": 1.5,
    "use_charbonnier_A": False,

    "resume": RESUME,
    "seed": 42,
    "deterministic": True,
    
    # 🚀 [Norm 调整] 模型输出直接对齐 dataset 里的 /1000 数据
    "norm_factor": 1.0,  
    "device": "cuda",

    "save_every_steps": 200,
    "keep_last_steps": 5,
    "save_every_epochs": 10,
    "save_epoch_model_only": False,
}

os.makedirs(SAVE_DIR, exist_ok=True)
config_save_path = os.path.join(SAVE_DIR, "experiment_config.json")
with open(config_save_path, "w") as f:
    json.dump({k: v for k, v in CONFIG.items() if isinstance(v, (str, int, float, bool))}, f, indent=4)

print(f"✅ 配置已加载 | 保存路径: {SAVE_DIR}")
print(f"🔥 模式: 纯线性回归 (Weighted L1) | LR: {CONFIG['lr']}")