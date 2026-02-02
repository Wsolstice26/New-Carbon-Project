import os
import json

# ============================================================
# 🎛️ 实验参数控制台
# ============================================================
PATCH_SIZE = 120
DIM = 64
BATCH_SIZE = 24
TIME_WINDOW = 3

# 🏷️ 实验标签（明确主指标）
TAG = "WeakSupervision_Scale10_NZMAE"

# ============================================================
# 📂 自动路径生成系统
# ============================================================
PROJECT_ROOT = "/home/wdc/Carbon-Emission-Super-Resolution"

DATA_DIR = os.path.join(
    PROJECT_ROOT, "data", f"Train_Data_Yearly_{PATCH_SIZE}"
)

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

    # Data / Model
    "patch_size": PATCH_SIZE,
    "dim": DIM,
    "batch_size": BATCH_SIZE,
    "time_window": TIME_WINDOW,

    # Weak supervision constraint (1km -> 100m)
    "consistency_scale": 10,

    # Training
    "epochs": 200,
    "num_workers": 6,
    "lr": 2e-4,

    # 🔥 主验证指标（论文级明确）
    "main_metric": "nonzero_mae",

    # Resume / Reproducibility
    "resume": True,
    "seed": 42,
    "deterministic": True,

    # Normalization
    "norm_factor": 11.0,
    "device": "cuda",

    # ========================================================
    # 💾 Checkpoint Strategy（新增，但不影响旧逻辑）
    # ========================================================
    "save_every_steps": 200,      # step 级 autosave 频率
    "keep_last_steps": 5,         # 轮转保留几个 step checkpoint
    "save_every_epochs": 10,      # 每 N 个 epoch 永久保存
    "save_epoch_model_only": False,  # 是否只存 model（先保持 False）
}

# ============================================================
# 📁 自动创建目录 + 保存配置快照
# ============================================================
os.makedirs(SAVE_DIR, exist_ok=True)

config_save_path = os.path.join(SAVE_DIR, "experiment_config.json")
with open(config_save_path, "w") as f:
    json.dump(
        {k: v for k, v in CONFIG.items() if isinstance(v, (str, int, float, bool))},
        f,
        indent=4,
    )

print(f"✅ 配置已加载 | 实验目录: {exp_name}")
print("⚠️ 弱监督一致性约束: 1km → 100m (scale=10)")
print("🔥 主验证指标: Nonzero-MAE")

