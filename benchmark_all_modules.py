import os
import torch
import torch.nn as nn
import time
import numpy as np

# ==========================================
# 🚀 环境补丁 (针对 14600K + 9060 XT)
# ==========================================

# 1. 解决 MIOpen Workspace 报错，强制申请显存空间以换取 3D 卷积速度
os.environ['MIOPEN_FORCE_USE_WORKSPACE'] = '1'
# 允许 MIOpen 动态搜索最佳算法（配合 benchmark=True）
os.environ['MIOPEN_DEBUG_CONV_GEMM'] = '0' 

# 2. 解决 Intel CPU 在容器内可能引发的数学库冲突
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

# 3. 开启极限性能模式
torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.deterministic = False

# ==========================================
# 导入项目模块
# ==========================================
try:
    from models.blocks import (
        MultiScaleBlock3D, SFTLayer3D, EfficientContextBlock, 
        MoEBlock
        # SimpleMambaBlock  <-- 已移除旧版
    )
    from models.losses import HybridLoss
    from models.network import DSTCarbonFormer 
    
    # ✅ 导入官方 Mamba
    from mamba_ssm import Mamba
    
except ImportError as e:
    print(f"❌ 导入失败: {e}。请确认你在项目根目录运行此脚本。")
    exit()

# ==========================================
# 🛠️ Mamba 形状适配器 (用于单独测试模块)
# ==========================================
class MambaAdapter(nn.Module):
    """
    官方 Mamba 需要 (B, L, D) 输入，而我们的数据是 (B, C, T, H, W)。
    这个类专门用于 benchmark 单独的 Mamba 模块，负责形状转换。
    """
    def __init__(self, dim):
        super().__init__()
        # 使用官方 Mamba 初始化
        self.mamba = Mamba(
            d_model=dim, 
            d_state=16, 
            d_conv=4, 
            expand=2
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        # 展平: (B, C, L) -> (B, L, C)
        x_flat = x.flatten(2).transpose(1, 2)
        # 进官方 Mamba
        out = self.mamba(x_flat)
        # 还原: (B, L, C) -> (B, C, T, H, W)
        return out.transpose(1, 2).view(B, C, T, H, W)

# ==========================================
# 测试函数
# ==========================================
def benchmark(name, module, inputs, iters=50):
    print(f"--------------------------------------------------")
    print(f"🧪 测试模块: {name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        module = module.to(device)
        module.eval()
        
        if isinstance(inputs, (tuple, list)):
            inputs = [x.to(device) for x in inputs]
        else:
            inputs = [inputs.to(device)]
            
        # 1. 预热 (寻找最佳算法)
        print("   🔥 预热中 (AMD 显卡正在匹配最佳算子)...")
        with torch.no_grad():
            for _ in range(15): # 增加预热次数让 MIOpen 完成搜索
                _ = module(*inputs)
        torch.cuda.synchronize()
        
        # 2. 正式计时
        start = time.time()
        with torch.no_grad():
            for _ in range(iters):
                _ = module(*inputs)
        torch.cuda.synchronize()
        
        avg_time = (time.time() - start) / iters * 1000 
        throughput = 1000 / avg_time * inputs[0].shape[0] 
        
        print(f"   ⏱️ 平均耗时: {avg_time:.2f} ms / batch")
        print(f"   🚀 吞吐量: {throughput:.1f} samples/s")
        return avg_time

    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        # 打印更详细的错误栈以便调试
        import traceback
        traceback.print_exc()
        return float('inf')

if __name__ == "__main__":
    # 检测硬件
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n🔥 硬件就绪: {gpu_name}")
        print(f"📦 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("✅ 检测到 RDNA 架构，已启用 MIOPEN 优化补丁")
    
    # 设定参数：匹配你的 Carbon_SR_Project 实际数据
    B, T, H, W = 4, 3, 128, 128
    DIM = 64 
    print(f"⚙️ 测试参数: BatchSize={B}, Dim={DIM}, PatchSize={H}x{W}")
    print("-" * 50)
    
    # 构造假数据 (输入通道: aux=9, main=1)
    df = torch.randn(B, DIM, T, H, W)
    da = torch.randn(B, DIM, T, H, W) 
    dra = torch.randn(B, 9, T, H, W)
    dm = torch.randn(B, 1, T, H, W)
    
    results = {}

    try:
        # 1. 测试各核心组件
        results['MultiScaleBlock (3D Conv)'] = benchmark("3D卷积模块", MultiScaleBlock3D(channels=DIM), df)
        results['MoE Block (Expert)'] = benchmark("MoE专家模块", MoEBlock(dim=DIM, num_experts=3, top_k=1), df)
        
        # ✅ 重点关注：使用适配器测试官方 Mamba
        results['Official Mamba (ROCm/C++)'] = benchmark("官方Mamba(硬件加速)", MambaAdapter(dim=DIM), df)

        results['SFT Fusion'] = benchmark("特征融合模块", SFTLayer3D(channels=DIM), (df, da))
        results['Context Attn'] = benchmark("上下文注意力", EfficientContextBlock(dim=DIM), df)

        # 2. 完整模型测试
        # 这里会调用我们在 network.py 里改好的代码
        full_model = DSTCarbonFormer(aux_c=9, main_c=1, dim=DIM)
        results['>>> FULL MODEL'] = benchmark("DSTCarbonFormer全网测试", full_model, (dra, dm))

        # 3. 性能排行榜
        print("\n" + "="*50)
        print("📊 模块速度排行榜 (14600K + RX 9060 XT)")
        print("="*50)
        
        valid_res = sorted({k: v for k, v in results.items() if v != float('inf')}.items(), key=lambda x: x[1])
        for name, t in valid_res:
            bar = "█" * int(t/5) if t < 200 else "█" * 40
            print(f"{name:<30} : {t:>7.2f} ms  {bar}")
            
    except Exception as e:
        print(f"\n❌ 严重错误: {e}")