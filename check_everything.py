import torch
from models.network import DSTCarbonFormer
from models.losses import HybridLoss
import time

def check_everything():
    print("\n========== 🛠️ 全系统自检程序启动 ==========")
    
    # 1. 准备环境
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 检测计算设备: {device}")
    
    # 2. 模拟假数据 (Batch=2, Time=3, H=128, W=128)
    print("\n[Step 1] 生成模拟数据...")
    # 辅助流: 9通道
    dummy_aux = torch.randn(2, 9, 3, 128, 128).to(device)
    # 主流: 1通道
    dummy_main = torch.randn(2, 1, 3, 128, 128).to(device)
    # 标签: 1通道
    dummy_target = torch.randn(2, 1, 3, 128, 128).to(device)
    print("✅ 模拟数据就绪")

    # 3. 测试模型 (含 FFT 硬约束)
    print("\n[Step 2] 测试模型前向传播 (含 FFT 硬约束)...")
    try:
        model = DSTCarbonFormer(aux_c=9, main_c=1).to(device)
        
        # 记录显存
        if torch.cuda.is_available():
            print(f"   显存占用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            
        # 跑一次
        pred = model(dummy_aux, dummy_main)
        
        print(f"   输出形状: {pred.shape}")
        if pred.shape == dummy_target.shape:
            print("✅ 模型结构测试通过！FFT 硬约束层运行正常。")
        else:
            print("❌ 尺寸不匹配！")
            return
            
    except Exception as e:
        print(f"❌ 模型报错: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 测试损失函数 (含物理一致性)
    print("\n[Step 3] 测试混合损失函数 (含物理守恒 Loss)...")
    try:
        # 注意：这里我们故意把 alpha, beta 等设为 1，确保每项都能算出数
        criterion = HybridLoss(alpha=1, beta=1, gamma=1, delta=1).to(device)
        
        # 关键点：一定要传入 input_main
        loss = criterion(pred, dummy_target, input_main=dummy_main)
        
        print(f"   计算出的 Loss 值: {loss.item()}")
        if not torch.isnan(loss):
            print("✅ 损失函数测试通过！物理一致性 Loss 计算正常。")
        else:
            print("❌ Loss 变成了 NaN (非数字)！可能梯度爆炸了。")
            return
            
    except Exception as e:
        print(f"❌ 损失函数报错: {e}")
        print("   💡 提示: 检查一下是否传入了 input_main 参数？")
        import traceback
        traceback.print_exc()
        return

    # 5. 测试反向传播 (Mixed Precision)
    print("\n[Step 4] 测试反向传播 (AMP 混合精度)...")
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler('cuda')
        
        optimizer.zero_grad()
        
        # 模拟一次完整的训练步
        with torch.amp.autocast('cuda'):
            pred = model(dummy_aux, dummy_main)
            loss = criterion(pred, dummy_target, input_main=dummy_main)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("✅ 反向传播测试通过！梯度更新正常。")
        
    except Exception as e:
        print(f"❌ 反向传播报错: {e}")
        return

    print("\n========== 🎉 恭喜！全系统自检通过，可以开始训练！ ==========")

if __name__ == "__main__":
    check_everything()