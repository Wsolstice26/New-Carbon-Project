import os
import gc  # 引入垃圾回收模块

# ==========================================
# 🚀 [环境补丁] 修正版
# ==========================================

# 1. MIOpen 缓存 (保留！这是好东西，加速启动)
cache_dir = os.path.expanduser("~/.cache/miopen")
os.makedirs(cache_dir, exist_ok=True)
os.environ['MIOPEN_USER_DB_PATH'] = cache_dir
os.environ['MIOPEN_CUSTOM_CACHE_DIR'] = cache_dir

# 2. 强制开启 Workspace (保留，防止报错，但要注意它会吃显存)
os.environ['MIOPEN_FORCE_USE_WORKSPACE'] = '1'

# 3. 日志优化 (保留)
os.environ['MIOPEN_LOG_LEVEL'] = '4'
os.environ['MIOPEN_DEBUG_CONV_GEMM'] = '0'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

# ❌ [删除/注释] 显存锁！
# 这行代码在极限显存下会导致分配失败，让 PyTorch 自动管理吧
# os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch
import torch.nn as nn
import time
import numpy as np

# 开启性能模式
torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.deterministic = False

# ==========================================
# 导入项目模块
# ==========================================
try:
    from models.blocks import (
        MultiScaleBlock3D, SFTLayer3D, MoEBlock
    )
    from models.network import DSTCarbonFormer 
    from mamba_ssm import Mamba
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    exit()

# ... MambaAdapter 保持不变 ...
class MambaAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
    def forward(self, x):
        B, C, T, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        out = self.mamba(x_flat)
        return out.transpose(1, 2).view(B, C, T, H, W)

# ==========================================
# ⚡️ 改进的测试函数 (增加显存清理)
# ==========================================
def benchmark(name, module, inputs, iters=50):
    print(f"--------------------------------------------------")
    print(f"🧪 测试模块: {name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        module = module.to(device)
        module.eval()
        
        # 确保输入在 GPU
        if isinstance(inputs, (tuple, list)):
            inputs = [x.to(device) for x in inputs]
        else:
            inputs = [inputs.to(device)]
            
        # 1. 预热
        print("   🔥 预热中...")
        with torch.no_grad():
            for _ in range(5): # 减少预热次数，省点时间
                _ = module(*inputs)
        torch.cuda.synchronize()
        
        # 2. 计时
        start = time.time()
        with torch.no_grad():
            for _ in range(iters):
                _ = module(*inputs)
        torch.cuda.synchronize()
        
        avg_time = (time.time() - start) / iters * 1000 
        print(f"   ⏱️ 平均耗时: {avg_time:.2f} ms / batch")
        return avg_time

    except Exception as e:
        print(f"   ❌ 测试失败 (OOM): {e}")
        return float('inf')
    
    finally:
        # 🔥🔥🔥 关键修改：每次测完，强制打扫战场！🔥🔥🔥
        del module
        del inputs
        gc.collect()           # Python 垃圾回收
        torch.cuda.empty_cache() # PyTorch 显存释放
        print("   🧹 显存已清理")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"🔥 硬件: {torch.cuda.get_device_name(0)}")
        # 打印当前显存使用情况
        print(f"📦 初始显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # ==========================================
    # ⚠️ 建议参数
    # Batch=16 在 120x120 下真的太极限了
    # 如果这次还爆，请务必改回 12
    # ==========================================
    B, T, H, W = 16, 3, 120, 120  
    DIM = 32 
    
    print(f"⚙️ 测试参数: Batch={B}, Dim={DIM}, Size={H}x{W}")
    
    # 构造数据
    df = torch.randn(B, DIM, T, H, W)
    da = torch.randn(B, DIM, T, H, W) 
    dra = torch.randn(B, 9, T, H, W)
    dm = torch.randn(B, 1, T, H, W)
    
    results = {}

    try:
        results['3D Conv'] = benchmark("3D卷积", MultiScaleBlock3D(channels=DIM), df)
        results['MoE'] = benchmark("MoE", MoEBlock(dim=DIM, num_experts=3, top_k=1), df)
        results['Mamba'] = benchmark("Mamba", MambaAdapter(dim=DIM), df)
        results['Fusion'] = benchmark("融合层", SFTLayer3D(channels=DIM), (df, da))
        
        # 全模型最后测
        full_model = DSTCarbonFormer(aux_c=9, main_c=1, dim=DIM)
        results['FULL MODEL'] = benchmark("DSTCarbonFormer", full_model, (dra, dm))

        print("\n📊 结果汇总:")
        for k, v in results.items():
            print(f"{k}: {v:.2f} ms")
            
    except Exception as e:
        print(f"\n❌ 严重错误: {e}")