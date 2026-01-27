import os
import time
import torch
import torch.nn as nn

# ==========================================
# 🧪 进阶生存测试 (尝试解除 GEMM 封印)
# ==========================================

# [关键修改] 注释掉下面这行，让 MIOpen 自动选择默认算法
# 如果这次运行崩了(Core Dump)，说明必须把这行加回去！
# os.environ['MIOPEN_DEBUG_CONV_GEMM'] = '1'

# 2. 禁止 MIOpen 编译新内核 (防止编译过程炸机，保持开启)
os.environ['MIOPEN_DEBUG_COMPILE_ONLY'] = '0'

# 3. [必须保留] 彻底禁用 Benchmark
# 千万别开这个！在不稳定驱动上开 Benchmark = 自杀
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def survival_test():
    print(f"🚑 显卡生存测试启动 (尝试默认算法)...")
    
    if torch.cuda.is_available():
        print(f"🔥 GPU 识别: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ 致命错误: 无法识别 GPU！")
        return

    device = torch.device('cuda')

    # ------------------------------------------------
    # 阶段 1: 2D 卷积测试
    # ------------------------------------------------
    print("\n[Step 1] 测试 2D 卷积 (基础功能)...")
    try:
        # Batch=2, 64通道, 128x128
        x2d = torch.randn(2, 64, 128, 128).to(device)
        conv2d = nn.Conv2d(64, 64, 3, 1, 1).to(device)
        
        for _ in range(5):
            _ = conv2d(x2d)
        torch.cuda.synchronize()
        print("✅ 2D 卷积存活！")
    except Exception as e:
        print(f"❌ 2D 卷积崩溃: {e}")
        return

    # ------------------------------------------------
    # 阶段 2: 3D 卷积测试 (关键！)
    # ------------------------------------------------
    print("\n[Step 2] 测试 3D 卷积 (解除 GEMM 锁定)...")
    print("   ⚠️  警告: 如果这里卡住或报错，请准备重启电脑。")
    try:
        # 小心翼翼：Batch Size = 1
        x3d = torch.randn(1, 64, 3, 128, 128).to(device)
        conv3d = nn.Conv3d(64, 64, 3, 1, 1).to(device)
        
        start = time.time()
        for i in range(5):
            print(f"   -> 尝试第 {i+1} 次计算...", end="", flush=True)
            _ = conv3d(x3d)
            torch.cuda.synchronize()
            print(" 成功")
            
        print("\n🎉 恭喜！你的显卡原生支持默认 3D 算法！")
        print("🚀 现在的速度应该比 GEMM 模式快很多。")
        
    except Exception as e:
        print(f"\n\n❌ 3D 卷积崩溃: {e}")
        print("💀 结论: 还是得把 GEMM 限制加回去 (os.environ['MIOPEN_DEBUG_CONV_GEMM'] = '1')")

if __name__ == "__main__":
    survival_test()