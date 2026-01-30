import os
import torch
import torch.nn as nn
import time
import traceback

# ==========================================
# 🚀 0. 环境补丁 (针对 AMD ROCm)
# ==========================================
# 1. 设置持久化缓存目录 (加速启动)
cache_dir = os.path.expanduser("~/.cache/miopen")
os.makedirs(cache_dir, exist_ok=True)
os.environ['MIOPEN_USER_DB_PATH'] = cache_dir
os.environ['MIOPEN_CUSTOM_CACHE_DIR'] = cache_dir

# 2. 强制申请显存，防止 MIOpen 报错
os.environ['MIOPEN_FORCE_USE_WORKSPACE'] = '1'
# 3. 屏蔽调试日志
os.environ['MIOPEN_LOG_LEVEL'] = '4'
os.environ['MIOPEN_DEBUG_CONV_GEMM'] = '0' 
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# 导入你的模块
try:
    from models.network import DSTCarbonFormer
    from models.losses import HybridLoss
    print("✅ 成功导入模型定义文件")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保你在项目根目录下运行此脚本: python check_system.py")
    exit()

def check_everything():
    print("\n========== 🛠️ 全系统自检程序启动 (120x120 全功率版) ==========")
    
    # 1. 准备环境
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 检测计算设备: {device}")
    if torch.cuda.is_available():
        print(f"   显卡型号: {torch.cuda.get_device_name(0)}")
    
    # 2. 模拟假数据 (适配 120x120)
    # Batch=2 (测试Batch), Time=3 (时间窗口), H=120, W=120 (新尺寸)
    print("\n[Step 1] 生成模拟数据 (120x120, 9通道)...")
    
    B, T, H, W = 2, 3, 120, 120  # 🔥 修改点：尺寸改为 120
    
    # 辅助流: 9通道 (7特征 + 2坐标)
    dummy_aux = torch.randn(B, 9, T, H, W).to(device)
    # 主流: 1通道
    dummy_main = torch.randn(B, 1, T, H, W).to(device)
    # 标签: 1通道
    dummy_target = torch.randn(B, 1, T, H, W).to(device)
    
    print(f"   Aux Shape: {dummy_aux.shape}")
    print(f"   Main Shape: {dummy_main.shape}")
    print("✅ 模拟数据就绪")

    # 3. 测试模型 (含 Mamba + MoE + FFT硬约束)
    print("\n[Step 2] 测试模型前向传播...")
    # 🔥 修改点：测试 Dim=64 的高配模式
    test_dim = 64
    print(f"   ⚙️ 测试配置: Dim={test_dim}")
    
    try:
        model = DSTCarbonFormer(aux_c=9, main_c=1, dim=test_dim).to(device)
        
        # 记录初始显存
        if torch.cuda.is_available():
            mem_start = torch.cuda.memory_allocated()
            
        # 跑一次前向传播
        start_time = time.time()
        pred = model(dummy_aux, dummy_main)
        end_time = time.time()
        
        # 记录显存变化
        if torch.cuda.is_available():
            mem_used = (torch.cuda.memory_allocated() - mem_start) / 1024**2
            print(f"   前向显存增量: {mem_used:.2f} MB")
            
        print(f"   输出形状: {pred.shape}")
        print(f"   耗时: {(end_time - start_time)*1000:.2f} ms")
        
        if pred.shape == dummy_target.shape:
            print("✅ 模型结构测试通过！(120x120 尺寸匹配成功)")
        else:
            print(f"❌ 尺寸不匹配！期望 {dummy_target.shape}, 实际 {pred.shape}")
            return
            
    except Exception as e:
        print(f"❌ 模型报错: {e}")
        traceback.print_exc()
        return

    # 4. 测试损失函数 (含物理一致性)
    print("\n[Step 3] 测试 HybridLoss (含物理守恒检查)...")
    try:
        # 初始化 Loss (scale=4 对应 120->30)
        criterion = HybridLoss(consistency_scale=4).to(device)
        
        # 🔥 关键：传入 input_mosaic_low_res (即 dummy_main)
        loss = criterion(pred, dummy_target, input_mosaic_low_res=dummy_main)
        
        print(f"   计算出的 Loss 值: {loss.item()}")
        print(f"   动态权重参数 requires_grad: {criterion.w_params.requires_grad}")
        
        if not torch.isnan(loss):
            print("✅ 损失函数测试通过！")
        else:
            print("❌ Loss 变成了 NaN！")
            return
            
    except Exception as e:
        print(f"❌ 损失函数报错: {e}")
        traceback.print_exc()
        return

    # 5. 测试反向传播 (Mixed Precision)
    print("\n[Step 4] 测试反向传播 (AMP 混合精度)...")
    try:
        params = list(model.parameters()) + list(criterion.parameters())
        optimizer = torch.optim.AdamW(params, lr=0.001)
        scaler = torch.amp.GradScaler('cuda')
        
        optimizer.zero_grad()
        
        # 模拟一次完整的训练步
        with torch.amp.autocast('cuda'):
            pred = model(dummy_aux, dummy_main)
            loss = criterion(pred, dummy_target, input_mosaic_low_res=dummy_main)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("✅ 反向传播测试通过！梯度更新正常。")
        
    except Exception as e:
        print(f"❌ 反向传播报错: {e}")
        traceback.print_exc()
        return

    print("\n========== 🎉 恭喜！全系统自检通过！ ==========")
    print(f"👉 你的 120x120 + Dim={test_dim} 环境已准备就绪。")
    print("👉 可以运行 train.py 开始正式训练了！")

if __name__ == "__main__":
    check_everything()