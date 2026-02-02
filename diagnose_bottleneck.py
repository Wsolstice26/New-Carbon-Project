# -*- coding: utf-8 -*-
import os
import gc
import time
import warnings
from typing import Dict, Any, Union, Tuple

# ==========================================
# 🔇 [日志静音]
# ==========================================
warnings.filterwarnings("ignore", message=".*Dynamo does not know how to trace the builtin.*")
warnings.filterwarnings("ignore", message=".*Unable to hit fast path of CUDAGraphs.*")
warnings.filterwarnings("ignore", message=".*TensorFloat32 tensor cores.*")

# ==========================================
# 🚀 [环境补丁]
# ==========================================
cache_dir = os.path.expanduser("~/.cache/miopen")
os.makedirs(cache_dir, exist_ok=True)
os.environ["MIOPEN_USER_DB_PATH"] = cache_dir
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = cache_dir
os.environ["MIOPEN_FORCE_USE_WORKSPACE"] = "1"
os.environ["MIOPEN_LOG_LEVEL"] = "4"
os.environ["MIOPEN_DEBUG_CONV_GEMM"] = "0"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ==========================================
# 导入项目模块
# ==========================================
try:
    from models.blocks import MultiScaleBlock3D, SFTLayer3D, MoEBlock
    from models.network import DSTCarbonFormer
    from mamba_ssm import Mamba
    from models.losses import HybridLoss
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    raise SystemExit(1)

# ==========================================
# Utils
# ==========================================
def _to_device(inputs, device):
    if isinstance(inputs, (tuple, list)):
        return [x.to(device, non_blocking=True) for x in inputs]
    return [inputs.to(device, non_blocking=True)]

def _clean(*objs):
    for o in objs:
        try: del o
        except: pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==========================================
# 🔥 全模型 + Loss Trainstep Benchmark
# ==========================================
def benchmark_full_model_trainstep(
    name: str,
    model: nn.Module,
    criterion: nn.Module,
    aux_input: torch.Tensor,
    main_input: torch.Tensor,
    target: torch.Tensor,
    constraint_scale: int = 120,
    iters: int = 5,  # 诊断模式下减少迭代次数，节省时间
    warmup: int = 2,
    lr: float = 1e-4,
    use_amp: bool = True,
) -> float:
    print(f"   🧪 [场景: {name}] 测试中...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = model.to(device)
        criterion = criterion.to(device)
        model.train()
        criterion.train()

        aux = aux_input.to(device, non_blocking=True)
        main = main_input.to(device, non_blocking=True)
        tgt = target.to(device, non_blocking=True)

        opt = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=lr)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # Warmup
        for _ in range(warmup):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred, pred_raw = model(aux, main, constraint_scale=constraint_scale)
                loss = criterion(pred, tgt, aux, pred_raw=pred_raw)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()

        # Benchmark
        total_time = 0.0
        
        for _ in range(iters):
            t_start = time.time()
            
            opt.zero_grad(set_to_none=True)
            # Fwd
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred, pred_raw = model(aux, main, constraint_scale=constraint_scale)
                loss = criterion(pred, tgt, aux, pred_raw=pred_raw)
            
            # Bwd
            scaler.scale(loss).backward()
            
            # Step
            scaler.step(opt)
            scaler.update()
            
            torch.cuda.synchronize()
            total_time += (time.time() - t_start)

        avg_time = total_time / iters
        print(f"   ⏱️ {name} 平均耗时: {avg_time:.4f} s/iter")
        return avg_time

    except Exception as e:
        print(f"   ❌ {name} 失败: {e}")
        return float('inf')
    finally:
        _clean(model, criterion, aux_input, main_input, target)

# ==========================================
# 🛠️ 诊断辅助类
# ==========================================
class MSEWrapper(nn.Module):
    """把 MSELoss 包装成 HybridLoss 的接口"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred, target, aux, pred_raw=None):
        return self.mse(pred, target)

class IdentityConstraint(nn.Module):
    """直通物理层，不做任何计算"""
    def forward(self, pred, input_mosaic, scale=None):
        return pred

# ==========================================
# main
# ==========================================
if __name__ == "__main__":
    print(f"🔥 开始全流程诊断 (GPU: {torch.cuda.get_device_name(0)})")
    
    # 基础参数
    B_BASE, T, H, W = 24, 3, 120, 120
    DIM, AUX_C, MAIN_C = 64, 9, 1
    TRAIN_SCALE = 120
    NORM = 11.0

    # 构造数据
    def get_data(b):
        dra = torch.randn(b, AUX_C, T, H, W)
        dm = torch.log1p(torch.rand(b, MAIN_C, T, H, W) * 1000.0) / NORM
        tgt = torch.log1p(torch.rand(b, MAIN_C, T, H, W) * 1000.0) / NORM
        return dra, dm, tgt

    # ==========================================
    # 🕵️ 诊断 1: 基准测试 (Baseline)
    # ==========================================
    print("\n🔹 [诊断 1/4] 基准测试 (Batch=24, 原版模型)")
    aux, main, target = get_data(B_BASE)
    model = DSTCarbonFormer(aux_c=AUX_C, main_c=MAIN_C, dim=DIM)
    
    # 尝试兼容不同的 Loss 初始化方式
    try:
        loss_fn = HybridLoss(consistency_scale=10, norm_factor=NORM)
    except TypeError:
        loss_fn = HybridLoss(consistency_scale=10) # 旧版接口兼容

    t_base = benchmark_full_model_trainstep(
        "Baseline", model, loss_fn, aux, main, target, TRAIN_SCALE
    )

    # ==========================================
    # 🕵️ 诊断 2: 显存瓶颈测试 (Small Batch)
    # ==========================================
    print("\n🔹 [诊断 2/4] 显存测试 (Batch=4)")
    aux_s, main_s, target_s = get_data(4)
    model_s = DSTCarbonFormer(aux_c=AUX_C, main_c=MAIN_C, dim=DIM)
    
    t_mem = benchmark_full_model_trainstep(
        "SmallBatch", model_s, loss_fn, aux_s, main_s, target_s, TRAIN_SCALE
    )
    
    # ==========================================
    # 🕵️ 诊断 3: 物理层旁路 (Constraint Bypass)
    # ==========================================
    print("\n🔹 [诊断 3/4] 物理层旁路 (Bypass Constraint)")
    model_phy = DSTCarbonFormer(aux_c=AUX_C, main_c=MAIN_C, dim=DIM)
    # 😈 核心操作: 替换物理层为直通
    model_phy.constraint_layer = IdentityConstraint() 
    
    t_phy = benchmark_full_model_trainstep(
        "NoConstraint", model_phy, loss_fn, aux, main, target, TRAIN_SCALE
    )

    # ==========================================
    # 🕵️ 诊断 4: Loss 旁路 (MSE Loss)
    # ==========================================
    print("\n🔹 [诊断 4/4] Loss 旁路 (MSE Only)")
    model_loss = DSTCarbonFormer(aux_c=AUX_C, main_c=MAIN_C, dim=DIM)
    loss_simple = MSEWrapper()
    
    t_loss = benchmark_full_model_trainstep(
        "SimpleLoss", model_loss, loss_simple, aux, main, target, TRAIN_SCALE
    )

    # ==========================================
    # 📊 结论分析
    # ==========================================
    print("\n" + "="*40)
    print("📋 诊断结论")
    print("="*40)
    
    # 1. 显存分析
    expected_linear_speedup = B_BASE / 4.0 # 理论上 Batch小6倍，时间快6倍
    actual_speedup = t_base / (t_mem + 1e-8)
    print(f"1. 显存分析: Batch减小6倍，速度提升 {actual_speedup:.1f}倍")
    if actual_speedup > 10.0:
        print("   🔴 [严重] 显存不足! Batch=24 导致了严重 Swap。请减小 BatchSize。")
    else:
        print("   ✅ [正常] 显存未成为主要瓶颈。")

    # 2. 物理层分析
    phy_diff = t_base - t_phy
    print(f"\n2. 物理层分析: 旁路物理层节省了 {phy_diff:.2f} 秒")
    if phy_diff > 1.0:
        print("   🔴 [严重] PhysicsConstraintLayer 是主要瓶颈!")
        print("   👉 你的 Water-Filling 排序在大尺寸下太慢。请务必使用优化版 Block 代码。")
    else:
        print("   ✅ [正常] 物理层耗时可接受。")

    # 3. Loss 分析
    loss_diff = t_base - t_loss
    print(f"\n3. Loss分析: 旁路 HybridLoss 节省了 {loss_diff:.2f} 秒")
    if loss_diff > 1.0:
        print("   ⚠️ [警告] Loss 计算偏重，可能是梯度计算 (EdgeAwareTV) 导致的。")
    
    print("\n✅ 诊断结束。请根据上述🔴提示修改代码。")