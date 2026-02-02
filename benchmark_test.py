# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# 你的项目里应该是类似这样导入（按你仓库实际路径调整）
# from model.network import DSTCarbonFormer
# from model.losses import HybridLoss
from models.network import DSTCarbonFormer
from models.losses import HybridLoss


torch.backends.cudnn.benchmark = True  # ROCm 下不一定等价，但保留无妨


def _sync():
    torch.cuda.synchronize()


def _timeit(fn):
    _sync()
    t0 = time.time()
    fn()
    _sync()
    return (time.time() - t0) * 1000.0


@torch.no_grad()
def benchmark_forward_only(module: nn.Module, x, name: str, iters=10, warmup=3):
    print(f"🧪 测试模块 (forward-only): {name}")
    print("   🔥 预热中...")
    for _ in range(warmup):
        _ = module(x)
    _sync()

    total = 0.0
    for _ in range(iters):
        total += _timeit(lambda: module(x))
    avg = total / iters
    print(f"   ⏱️ 平均耗时: {avg:.2f} ms / batch")
    print("--------------------------------------------------")
    return avg


def benchmark_trainstep(module: nn.Module, x, target, name: str, iters=10, warmup=3, amp=True):
    print(f"🧪 测试模块 (trainstep): {name}")
    module.train()
    opt = torch.optim.Adam(module.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # warmup
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            y = module(x)
            loss = F.mse_loss(y, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    _sync()

    fwd_ms = loss_ms = bwd_ms = step_ms = 0.0
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)

        # fwd
        def _fwd():
            nonlocal y
            with torch.amp.autocast("cuda", enabled=amp):
                y = module(x)

        y = None
        fwd_ms += _timeit(_fwd)

        # loss
        def _loss():
            nonlocal loss
            loss = F.mse_loss(y, target)

        loss = None
        loss_ms += _timeit(_loss)

        # bwd
        def _bwd():
            scaler.scale(loss).backward()

        bwd_ms += _timeit(_bwd)

        # step
        def _step():
            scaler.step(opt)
            scaler.update()

        step_ms += _timeit(_step)

    fwd_ms /= iters
    loss_ms /= iters
    bwd_ms /= iters
    step_ms /= iters
    total_ms = fwd_ms + loss_ms + bwd_ms + step_ms

    print(f"   ⏱️ fwd : {fwd_ms:.2f} ms")
    print(f"   ⏱️ loss: {loss_ms:.2f} ms")
    print(f"   ⏱️ bwd : {bwd_ms:.2f} ms")
    print(f"   ⏱️ step: {step_ms:.2f} ms")
    print(f"   ✅ total: {total_ms:.2f} ms / iter")
    print("--------------------------------------------------")
    return total_ms


def benchmark_full_model_trainstep(
    model: nn.Module,
    criterion: nn.Module,
    aux, main, target,
    name: str,
    constraint_scale,
    iters=5,
    warmup=1,
    amp=True,
):
    print(f"🧪 全流程测试 (Full Model + Loss): {name}")
    print(f"   ⚙️ constraint_scale={constraint_scale} | amp={amp}")
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    print("   🔥 预热中 (含反传)...")
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            pred, pred_raw = model(aux, main, constraint_scale=constraint_scale)
            loss = criterion(pred, target, aux, pred_raw=pred_raw)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    _sync()

    total_ms = 0.0
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)

        # model fwd
        def _mfwd():
            nonlocal pred, pred_raw
            with torch.amp.autocast("cuda", enabled=amp):
                pred, pred_raw = model(aux, main, constraint_scale=constraint_scale)

        pred = pred_raw = None
        t_fwd = _timeit(_mfwd)

        # loss
        def _l():
            nonlocal loss
            loss = criterion(pred, target, aux, pred_raw=pred_raw)

        loss = None
        t_loss = _timeit(_l)

        # backward
        t_bwd = _timeit(lambda: scaler.scale(loss).backward())

        # step
        def _st():
            scaler.step(opt)
            scaler.update()

        t_step = _timeit(_st)

        t_total = t_fwd + t_loss + t_bwd + t_step
        total_ms += t_total

        print(f"   ⏱️ Model Fwd : {t_fwd:.2f} ms")
        print(f"   ⏱️ Loss Calc : {t_loss:.2f} ms")
        print(f"   ⏱️ Backward  : {t_bwd:.2f} ms")
        print(f"   ⏱️ Opt Step  : {t_step:.2f} ms")
        print(f"   ✅ Total Time: {t_total:.2f} ms / iter")

        if hasattr(criterion, "log_vars"):
            w = torch.exp(-criterion.log_vars.detach()).cpu().numpy()
            print(f"   ⚖️ weights(exp(-log_vars)) = {w}")

    avg = total_ms / iters
    return avg


def main():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🔥 硬件: {gpu_name}")

    # 参数（与你输出一致）
    B = int(os.environ.get("BATCH", "4"))
    C_AUX = 9
    C_MAIN = 1
    DIM = int(os.environ.get("DIM", "64"))
    H = W = int(os.environ.get("SIZE", "120"))
    T = int(os.environ.get("T", "3"))

    TRAIN_SCALE = int(os.environ.get("CONSTRAINT_SCALE", "120"))
    AMP = os.environ.get("AMP", "1") == "1"

    # ✅ benchmark 时可选跳过 constraint（默认不跳过）
    # 设为 1 时，会把 constraint_scale 传 None，从而 network.py 里跳过 constraint 计算
    SKIP_CONSTRAINT = os.environ.get("BENCH_SKIP_CONSTRAINT", "0") == "1"
    constraint_scale = None if SKIP_CONSTRAINT else TRAIN_SCALE

    print(f"⚙️ 测试参数: Batch={B}, Dim={DIM}, Size={H}x{W}, T={T}")
    print(f"⚙️ 全模型 constraint_scale={constraint_scale}")

    # dummy inputs
    aux = torch.randn(B, C_AUX, T, H, W, device=device)
    main_in = torch.rand(B, C_MAIN, T, H, W, device=device)  # log_norm 域，>=0
    target = torch.rand(B, C_MAIN, T, H, W, device=device)

    # 3D卷积模块（示例）
    conv3d = nn.Conv3d(DIM, DIM, 3, padding=1).to(device)

    # MoE模块 / Mamba模块 / Fusion模块：你这里按项目实际构建
    # 这里沿用你目前脚本已有的构造方式（略）——重点是 full-model 部分
    # 如果你原脚本里有更完整的 module 构造，请保留并仅替换 full-model benchmark 调用与 constraint 传参逻辑。

    # 全模型
    model = DSTCarbonFormer(aux_c=C_AUX, main_c=C_MAIN, dim=DIM).to(device)
    criterion = HybridLoss(consistency_scale=10, norm_factor=11.0).to(device)

    print("=========================")
    print("🚀 准备进行 DSTCarbonFormer 全模型测试...")
    print("--------------------------------------------------")
    avg_ms = benchmark_full_model_trainstep(
        model=model,
        criterion=criterion,
        aux=aux,
        main=main_in,
        target=target,
        name="DSTCarbonFormer + HybridLoss",
        constraint_scale=constraint_scale,
        iters=20,
        warmup=3,
        amp=AMP,
    )
    _ = avg_ms


if __name__ == "__main__":
    main()
