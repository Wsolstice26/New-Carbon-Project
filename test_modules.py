import torch
from data.dataset import DualStreamDataset
from torch.utils.data import DataLoader
from models.network import DSTCarbonFormer
from config import CONFIG
import time

def test_dataset():
    print("\n========== 🧪 测试 1: 数据加载器 ==========")
    # 1. 初始化数据集 (强制使用 batch_size=2 来测试)
    ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'train')
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    
    print(f"✅ 数据集加载成功，总样本数: {len(ds)}")
    
    # 2. 尝试读取一个 Batch
    start = time.time()
    aux, main, target = next(iter(dl))
    print(f"⏱️ 读取一个Batch耗时: {time.time()-start:.4f}s")
    
    # 3. 检查形状
    # 预期: [2, 9, 3, 128, 128]
    print(f"   Aux Shape  (辅助流): {aux.shape}") 
    # 预期: [2, 1, 3, 128, 128]
    print(f"   Main Shape (主流)  : {main.shape}")
    print(f"   Target Shape (标签): {target.shape}")
    
    if aux.shape[1] == 9 and main.shape[1] == 1:
        print("🎉 数据格式检查通过！")
    else:
        print("❌ 数据通道数不对，请检查 Dataset 代码！")
    
    return aux, main

def test_model(aux_dummy, main_dummy):
    print("\n========== 🧪 测试 2: 模型前向传播 ==========")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 测试设备: {device}")
    
    # 1. 搬运数据
    aux = aux_dummy.to(device)
    main = main_dummy.to(device)
    
    # 2. 初始化模型
    model = DSTCarbonFormer(aux_c=9, main_c=1).to(device)
    
    # 3. 尝试跑一次 Forward
    try:
        output = model(aux, main)
        print(f"✅ 模型运行成功！输出形状: {output.shape}")
        
        # 检查输出尺寸是否和输入一致 (128x128)
        if output.shape == main.shape:
            print("🎉 输入输出尺寸完全匹配！模型结构没问题。")
        else:
            print(f"❌ 尺寸不匹配! 输入: {main.shape}, 输出: {output.shape}")
            
    except Exception as e:
        print(f"❌ 模型报错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 先测数据
    try:
        aux_batch, main_batch = test_dataset()
        # 数据没问题再测模型
        test_model(aux_batch, main_batch)
    except Exception as e:
        print(f"❌ 测试中断: {e}")