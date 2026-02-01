import torch
import sys
import time

print(f"🔥 PyTorch Version: {torch.__version__}")
print(f"🔥 Device: {torch.cuda.get_device_name(0)}")

print("\n--------------------------------------------------")
print("⚡️ 正在运行 Mamba 完整速度测试 (Forward + Backward)...")

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    device = "cuda"
    B, L, D = 16, 4096, 64 # 加大一点负载
    
    # 🔥 修正点：显式开启 requires_grad=True，否则无法测试反向传播
    u = torch.randn(B, D, L, device=device).requires_grad_(True)
    delta = torch.randn(B, D, L, device=device).requires_grad_(True)
    A = torch.randn(D, 16, device=device).requires_grad_(True)
    B_ = torch.randn(B, 16, L, device=device).requires_grad_(True)
    C = torch.randn(B, 16, L, device=device).requires_grad_(True)
    D_ = torch.randn(D, device=device).requires_grad_(True)
    z = torch.randn(B, D, L, device=device).requires_grad_(True)
    delta_bias = torch.randn(D, device=device).requires_grad_(True)

    # 1. 前向传播
    torch.cuda.synchronize()
    t0 = time.time()
    out = selective_scan_fn(u, delta, A, B_, C, D_, z=z, delta_bias=delta_bias, delta_softplus=True)
    torch.cuda.synchronize()
    print(f"🚀 Mamba Forward 成功！耗时: {(time.time() - t0)*1000:.2f} ms")

    # 2. 反向传播
    t1 = time.time()
    out.sum().backward()
    torch.cuda.synchronize()
    print(f"🚀 Mamba Backward 成功！耗时: {(time.time() - t1)*1000:.2f} ms")

    print("\n🎉🎉🎉 恭喜！你的环境现在是【真·高性能 Mamba】！")

except ImportError as e:
    print(f"❌ 导入失败: {e}")
except Exception as e:
    print(f"❌ 运行报错: {e}")