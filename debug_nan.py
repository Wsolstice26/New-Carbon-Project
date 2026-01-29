import os
# 模拟训练环境配置
os.environ['MIOPEN_DEBUG_CONV_GEMM'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
from models.network import DSTCarbonFormer

def check_nan_hook(module, input, output):
    """
    这是一个钩子函数，用来挂在每一层网络上。
    一旦输出包含 NaN，立刻打印报错。
    """
    if isinstance(output, tuple):
        out_tensor = output[0]
    else:
        out_tensor = output
        
    if torch.isnan(out_tensor).any():
        print(f"💀 [NaN Detected] layer: {module.__class__.__name__}")
        print(f"   --> Input range: {input[0].min().item():.4f} to {input[0].max().item():.4f}")
        raise RuntimeError(f"NaN found in {module.__class__.__name__} output!")

def debug_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Debugging on {device}...")

    # 1. 初始化模型
    model = DSTCarbonFormer(aux_c=9, main_c=1, dim=64).to(device)
    model.train()

    # 2. 为模型所有子模块注册 NaN 检查钩子
    print("🕵️ 正在挂载 NaN 监控钩子...")
    for name, layer in model.named_modules():
        layer.register_forward_hook(check_nan_hook)

    # 3. 构造虚拟数据 (模拟你的输入形状)
    # Batch=4, Channel=9/1, Time=3, H=128, W=128
    B, T, H, W = 4, 3, 128, 128
    dummy_aux = torch.randn(B, 9, T, H, W).to(device)
    dummy_main = torch.randn(B, 1, T, H, W).to(device)
    dummy_target = torch.randn(B, 1, T, H, W).to(device)

    # 4. 开启 AMP 混合精度进行压力测试
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.L1Loss()

    print("\n🚀 开始高压测试 (Forward + Backward)...")
    try:
        with torch.amp.autocast('cuda'):
            # 前向传播
            print("   -> Forward pass...")
            pred = model(dummy_aux, dummy_main)
            
            # 计算 Loss (我们在 train.py 里加了 float() 保护，这里也模拟一下)
            loss = criterion(pred.float(), dummy_target.float())

        # 反向传播
        print("   -> Backward pass...")
        scaler.scale(loss).backward()
        
        print("\n✅ 测试通过！模型结构在当前数据下没有产生 NaN。")
        print("   如果训练时还报错，说明是真实数据的数值范围问题（比如有极端异常值）。")

    except RuntimeError as e:
        print(f"\n💥 抓到了！错误详情: \n{e}")

if __name__ == "__main__":
    # 打开 PyTorch 自带的异常检测（会定位到具体代码行，虽然慢但精准）
    torch.autograd.set_detect_anomaly(True)
    debug_model()