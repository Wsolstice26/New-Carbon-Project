# -*- coding: utf-8 -*-
import os
import gc
import time
import warnings
from typing import Union, Tuple, List, Dict

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
torch.set_float32_matmul_precision('high')
import torch.nn as nn

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ==========================================
# 导入项目模块
# ==========================================
try:
    from models.blocks import MultiScaleBlock3D, SFTLayer3D, MoEBlock
    from models.network import DSTCarbonFormer
    from mamba_ssm import Mamba
    # 🔥 新增导入 Loss
    from models.losses import HybridLoss 
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("💡 请确保 losses.py 已保存到 models/losses.py，且 blocks.py/network.py 均存在。")
    raise SystemExit(1)

# ... (MambaAdapter 保持不变) ...
class MambaAdapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        out = self.mamba(x_flat)
        return out.transpose(1, 2).view(B, C, T, H, W)

# ... (_to_device, _clean 保持不变) ...
def _to_device(inputs, device):
    if isinstance(inputs, (tuple, list)):
        return [x.to(device, non_blocking=True) for x in inputs]
    return [inputs.to(device, non_blocking=True)]

def _clean(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ... (benchmark_forward 保持不变) ...
@torch.no_grad()
def benchmark_forward(name, module, inputs, iters=50, warmup=5):
    print("--------------------------------------------------")
    print(f"🧪 测试模块 (forward-only): {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        module = module.to(device)
        module.eval()
        inps = _to_device(inputs, device)
        
        # 预热
        print("   🔥 预热中...")
        for _ in range(warmup):
            torch.compiler.cudagraph_mark_step_begin()
            _ = module(*inps)
        if torch.cuda.is_available(): torch.cuda.synchronize()

        # 测试
        start = time.time()
        for _ in range(iters):
            torch.compiler.cudagraph_mark_step_begin()
            _ = module(*inps)
        if torch.cuda.is_available(): torch.cuda.synchronize()

        avg_ms = (time.time() - start) / iters * 1000.0
        print(f"   ⏱️ 平均耗时: {avg_ms:.2f} ms / batch")
        return avg_ms
    except Exception as e:
        print(f"   ❌ forward-only 测试失败: {e}")
        return float("inf")
    finally:
        _clean(module, inputs)

# ... (benchmark_trainstep 保持不变，用于单模块测试) ...
def benchmark_trainstep(name, module, inputs, iters=20, warmup=3, lr=1e-4, use_amp=False):
    # (此处代码与之前一致，省略以节省篇幅，重点是下面的 full_model 版)
    print("--------------------------------------------------")
    print(f"🧪 测试模块 (trainstep): {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        module = module.to(device)
        module.train()
        inps = _to_device(inputs, device)
        opt = torch.optim.AdamW(module.parameters(), lr=lr)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        for _ in range(warmup):
            torch.compiler.cudagraph_mark_step_begin()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = module(*inps)
                loss = out.float().mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if torch.cuda.is_available(): torch.cuda.synchronize()

        fwd_t = loss_t = bwd_t = step_t = 0.0
        for _ in range(iters):
            torch.compiler.cudagraph_mark_step_begin()
            opt.zero_grad(set_to_none=True)
            
            t0 = time.time()
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = module(*inps)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1 = time.time()
            
            loss = out.float().mean()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t2 = time.time()
            
            scaler.scale(loss).backward()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t3 = time.time()
            
            scaler.step(opt)
            scaler.update()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t4 = time.time()
            
            fwd_t += (t1 - t0); loss_t += (t2 - t1); bwd_t += (t3 - t2); step_t += (t4 - t3)

        n = iters
        fwd_ms, loss_ms = fwd_t/n*1000, loss_t/n*1000
        bwd_ms, step_ms = bwd_t/n*1000, step_t/n*1000
        total_ms = fwd_ms + loss_ms + bwd_ms + step_ms
        print(f"   ⏱️ fwd : {fwd_ms:.2f} ms")
        print(f"   ⏱️ loss: {loss_ms:.2f} ms")
        print(f"   ⏱️ bwd : {bwd_ms:.2f} ms")
        print(f"   ⏱️ step: {step_ms:.2f} ms")
        print(f"   ✅ total: {total_ms:.2f} ms / iter")
        return {"total_ms": total_ms, "fwd_ms": fwd_ms, "bwd_ms": bwd_ms}
    except Exception as e:
        print(f"   ❌ trainstep 测试失败: {e}")
        return {"total_ms": float("inf")}
    finally:
        _clean(module, inputs)

# ==========================================
# 🔥 [新增] 全模型 + Loss 专用测试函数
# ==========================================
def benchmark_full_model_trainstep(
    name: str,
    model: nn.Module,
    criterion: nn.Module,
    aux_input: torch.Tensor,
    main_input: torch.Tensor,
    target: torch.Tensor,
    iters: int = 20,
    warmup: int = 3,
    lr: float = 1e-4,
    use_amp: bool = True # 默认开启混合精度
):
    print("--------------------------------------------------")
    print(f"🧪 全流程测试 (Full Model + Loss): {name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = model.to(device)
        criterion = criterion.to(device)
        model.train()
        
        # 准备数据
        aux = aux_input.to(device, non_blocking=True)
        main = main_input.to(device, non_blocking=True)
        tgt = target.to(device, non_blocking=True)

        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        print("   🔥 预热中 (含反传)...")
        for _ in range(warmup):
            torch.compiler.cudagraph_mark_step_begin()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                # 1. Forward
                pred = model(aux, main)
                # 2. Loss (传入 main 作为 input_mosaic_low_res)
                loss = criterion(pred, tgt, input_mosaic_low_res=main)
            # 3. Backward
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if torch.cuda.is_available(): torch.cuda.synchronize()

        fwd_t = loss_t = bwd_t = step_t = 0.0

        for _ in range(iters):
            torch.compiler.cudagraph_mark_step_begin()
            opt.zero_grad(set_to_none=True)

            # --- Forward ---
            t0 = time.time()
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(aux, main)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1 = time.time()

            # --- Loss ---
            with torch.amp.autocast('cuda', enabled=use_amp):
                # HybridLoss 需要 (pred, target, low_res_input)
                loss = criterion(pred, tgt, input_mosaic_low_res=main)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t2 = time.time()

            # --- Backward ---
            scaler.scale(loss).backward()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t3 = time.time()

            # --- Optimizer ---
            scaler.step(opt)
            scaler.update()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t4 = time.time()

            fwd_t += (t1 - t0)
            loss_t += (t2 - t1)
            bwd_t += (t3 - t2)
            step_t += (t4 - t3)

        n = iters
        fwd_ms = fwd_t / n * 1000.0
        loss_ms = loss_t / n * 1000.0
        bwd_ms = bwd_t / n * 1000.0
        step_ms = step_t / n * 1000.0
        total_ms = fwd_ms + loss_ms + bwd_ms + step_ms

        print(f"   ⏱️ Model Fwd : {fwd_ms:.2f} ms")
        print(f"   ⏱️ Loss Calc : {loss_ms:.2f} ms")
        print(f"   ⏱️ Backward  : {bwd_ms:.2f} ms")
        print(f"   ⏱️ Opt Step  : {step_ms:.2f} ms")
        print(f"   ✅ Total Time: {total_ms:.2f} ms / iter")

        return {"total_ms": total_ms}

    except Exception as e:
        print(f"   ❌ Full Model 测试失败: {e}")
        return {"total_ms": float("inf")}
    finally:
        _clean(model, criterion, aux_input, main_input, target)


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"🔥 硬件: {torch.cuda.get_device_name(0)}")
    
    # =========================
    # 测试参数
    # =========================
    B, T, H, W = 24, 3, 120, 120
    DIM = 64
    AUX_C = 9
    MAIN_C = 1
    
    print(f"⚙️ 测试参数: Batch={B}, Dim={DIM}, Size={H}x{W}, T={T}")

    # 构造数据
    # 单模块用数据
    df = torch.randn(B, DIM, T, H, W)
    da = torch.randn(B, DIM, T, H, W)
    
    # 🔥 全模型用数据
    # dra: 辅助数据 (9通道)
    dra = torch.randn(B, AUX_C, T, H, W)
    # dm: 主输入数据 (1通道, 也是 Mosaic 低清输入)
    dm = torch.randn(B, MAIN_C, T, H, W)
    # target: 目标数据 (1通道, 高清真值)
    target = torch.randn(B, MAIN_C, T, H, W)

    results_fwd = {}
    results_step = {}

    # ========== 1. 单模块测试 ==========
    results_fwd["3D Conv"] = benchmark_forward("3D卷积", MultiScaleBlock3D(channels=DIM), df, iters=50)
    results_step["3D Conv"] = benchmark_trainstep("3D卷积", MultiScaleBlock3D(channels=DIM), df, iters=10)

    results_fwd["MoE"] = benchmark_forward("MoE", MoEBlock(dim=DIM, num_experts=3, top_k=1), df, iters=50)
    results_step["MoE"] = benchmark_trainstep("MoE", MoEBlock(dim=DIM, num_experts=3, top_k=1), df, iters=10)

    mamba_mod = MambaAdapter(dim=DIM)
    # 编译 Mamba (实际上是 disable)
    try:
        mamba_mod = torch.compile(mamba_mod, mode='reduce-overhead')
    except: pass
    
    results_fwd["Mamba"] = benchmark_forward("Mamba", mamba_mod, df, iters=50)
    results_step["Mamba"] = benchmark_trainstep("Mamba", mamba_mod, df, iters=10)

    results_fwd["Fusion"] = benchmark_forward("融合层", SFTLayer3D(channels=DIM), (df, da), iters=50)

    # ========== 2. 🔥 全模型 + Loss 测试 ==========
    print("\n=========================")
    print("🚀 准备进行 DSTCarbonFormer 全模型测试...")
    
    full_model = DSTCarbonFormer(aux_c=AUX_C, main_c=MAIN_C, dim=DIM)
    loss_fn = HybridLoss(consistency_scale=4) # 实例化 HybridLoss
    
    benchmark_full_model_trainstep(
        name="DSTCarbonFormer + HybridLoss",
        model=full_model,
        criterion=loss_fn,
        aux_input=dra,
        main_input=dm,
        target=target,
        iters=20,
        use_amp=True
    )

    # 汇总输出
    print("\n=========================")
    print("📊 Forward-only 结果汇总 (ms/batch)")
    for k, v in results_fwd.items():
        print(f"{k:12s}: {v:.2f} ms")

    print("\n=========================")
    print("📊 Trainstep 结果汇总 (ms/iter)")
    for k, d in results_step.items():
        if "total_ms" in d:
            print(f"{k:12s}: total={d['total_ms']:.2f}")

    print("\n✅ 测试完成。")