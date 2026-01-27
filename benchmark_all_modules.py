import os
import torch
import torch.nn as nn
import time
import numpy as np

# ==========================================
# 🧪 极限性能模式 (RISKY MODE)
# ==========================================

# 1. [核心修改] 开启 Benchmark
#    允许 PyTorch 运行一次试跑，来寻找那个需要 300MB 的高性能算法
#    如果这里卡住不动超过 1 分钟，请立即 Ctrl+C
torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.deterministic = False

# 2. [关键] 告诉 MIOpen 不要因为显存不够就轻易放弃
#    开启日志，看看它到底选了哪个算法 (Algo 1 是 GEMM，如果变了说明成功)
os.environ['MIOPEN_ENABLE_LOGGING'] = '1' 
os.environ['MIOPEN_LOG_LEVEL'] = '3' # 显示 Warning 和 Info

# 3. 解除 GEMM 锁定
if 'MIOPEN_DEBUG_CONV_GEMM' in os.environ:
    del os.environ['MIOPEN_DEBUG_CONV_GEMM']

# ==========================================
# 导入模块
# ==========================================
from models.blocks import (
    MultiScaleBlock3D, 
    SFTLayer3D, 
    EfficientContextBlock, 
    FrequencyHardConstraint,
    MoEBlock,           
    SimpleMambaBlock    
)
from models.losses import HybridLoss
from models.network import DSTCarbonFormer # 导入主模型

def benchmark(name, module, inputs, iters=50):
    print(f"--------------------------------------------------")
    print(f"🧪 测试模块: {name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        module = module.to(device)
        module.eval()
        
        # 处理输入
        if isinstance(inputs, (tuple, list)):
            inputs = [x.to(device) for x in inputs]
        else:
            inputs = [inputs.to(device)]
            
        # 1. 预热 (Warmup)
        # 注意：开启 Benchmark 后，第一次运行会非常慢（因为在搜算法）
        print("   🔥 预热中 (正在搜索最佳算法)...")
        with torch.no_grad():
            for _ in range(5):
                _ = module(*inputs)
        torch.cuda.synchronize()
        
        # 2. 正式计时
        start = time.time()
        with torch.no_grad():
            for _ in range(iters):
                _ = module(*inputs)
        torch.cuda.synchronize()
        
        # 计算结果
        avg_time = (time.time() - start) / iters * 1000 
        throughput = 1000 / avg_time * inputs[0].shape[0] 
        
        print(f"   ⏱️ 平均耗时: {avg_time:.2f} ms / batch")
        print(f"   🚀 吞吐量: {throughput:.1f} samples/s")
        return avg_time

    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        return float('inf')

if __name__ == "__main__":
    print(f"\n🔥 PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print("🚀 模式: 极限性能 (Benchmark=ON)")
    else:
        print("⚠️ 未检测到 GPU")

    # 设定测试参数
    B, T, H, W = 4, 3, 128, 128
    DIM = 64 
    print(f"📦 Batch Size: {B}, Dim: {DIM}, Input: {T}x{H}x{W}")
    print("-" * 50)
    
    # 构造假数据
    dummy_feat = torch.randn(B, DIM, T, H, W)
    dummy_aux = torch.randn(B, DIM, T, H, W) 
    dummy_pred = torch.randn(B, 1, T, H, W)
    dummy_target = torch.randn(B, 1, T, H, W)
    dummy_raw_aux = torch.randn(B, 9, T, H, W)
    dummy_raw_main = torch.randn(B, 1, T, H, W)
    
    results = {}

    try:
        # ==========================================
        # 1. 关键组件
        # ==========================================
        # 这是成败的关键，看它能否突破 147ms
        block_ms = MultiScaleBlock3D(channels=DIM)
        results['MultiScaleBlock (3D Conv)'] = benchmark("MultiScaleBlock3D", block_ms, dummy_feat)
        
        block_moe = MoEBlock(dim=DIM, num_experts=3, top_k=1)
        results['MoE Block (Expert)'] = benchmark("MoEBlock", block_moe, dummy_feat)
        
        block_mamba = SimpleMambaBlock(dim=DIM)
        results['Mamba Block (SSM)'] = benchmark("SimpleMambaBlock", block_mamba, dummy_feat)

        block_sft = SFTLayer3D(channels=DIM)
        results['SFT Fusion'] = benchmark("SFTLayer3D", block_sft, (dummy_feat, dummy_aux))
        
        block_ctx = EfficientContextBlock(dim=DIM)
        results['Context Attn'] = benchmark("EfficientContextBlock", block_ctx, dummy_feat)
        
        loss_fn = HybridLoss().cuda()
        results['Hybrid Loss'] = benchmark("HybridLoss", loss_fn, (dummy_pred, dummy_target, dummy_raw_main))

        # ==========================================
        # 2. 完整模型测试
        # ==========================================
        # 看看这一套组合拳下来的总速度
        full_model = DSTCarbonFormer(aux_c=9, main_c=1, dim=DIM)
        results['>>> FULL MODEL (DSTCarbonFormer)'] = benchmark("DSTCarbonFormer (Whole Net)", full_model, (dummy_raw_aux, dummy_raw_main))

        # ==========================================
        # 3. 排行榜
        # ==========================================
        print("\n" + "="*50)
        print("📊 模块速度排行榜 (越快越好)")
        print("="*50)
        
        valid_results = {k: v for k, v in results.items() if v != float('inf')}
        sorted_res = sorted(valid_results.items(), key=lambda x: x[1])
        
        for name, t in sorted_res:
            bar_len = int(t / 5) if t < 200 else 40
            bar = "█" * bar_len
            print(f"{name:<35} : {t:>6.2f} ms  {bar}")
            
    except Exception as e:
        print(f"\n❌ 测试中断: {e}")
        import traceback
        traceback.print_exc()