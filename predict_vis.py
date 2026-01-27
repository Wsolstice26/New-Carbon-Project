import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader

# 导入你的模块
from data.dataset import DualStreamDataset
from models.network import DSTCarbonFormer
from config import CONFIG

# 设置中文字体 (防止乱码)
# 如果还乱码，可以尝试 'Microsoft YaHei' 或删除这两行
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 使用设备: {device}")

    # 1. 加载最佳模型
    model_path = os.path.join(CONFIG['save_dir'], "best_model.pth")
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}")
        return

    print(f"📦 加载模型权重: {model_path}")
    model = DSTCarbonFormer(aux_c=9, main_c=1).to(device)
    
    # 加载权重 (map_location 确保在单卡或CPU上也能跑)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. 加载验证集数据
    try:
        val_ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'val')
        # shuffle=True 随机抽几张
        val_dl = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=0)
        
        # 获取一个 Batch
        aux, main, target = next(iter(val_dl))
        aux = aux.to(device)
        main = main.to(device)
        target = target.to(device)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 3. 预测
    print("🔮 正在预测...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            pred = model(aux, main)

    # 4. 反归一化 (还原成真实碳排放吨数)
    norm_factor = 11.0
    
    # [Batch, Channel, Time, H, W]
    # 还原到真实物理空间
    pred_real = torch.expm1(pred * norm_factor).clamp(min=0).cpu().numpy()
    target_real = torch.expm1(target * norm_factor).clamp(min=0).cpu().numpy()
    
    # 5. 可视化绘图
    print("🎨 正在绘图...")
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    
    # 选取时间窗口的中间帧 (索引 1) 进行展示
    time_idx = 1 
    
    for i in range(4):
        # 提取中间那一年的数据: [i, Channel=0, Time=1, H, W]
        t_img = target_real[i, 0, time_idx]
        p_img = pred_real[i, 0, time_idx]
        
        # 统一 Colorbar 范围 (以真值为准，防止预测值过大导致全黑)
        vmax = max(np.max(t_img), np.max(p_img), 1.0)
        
        # --- 第一列：真实标签 (Ground Truth) ---
        ax1 = axes[i, 0]
        im1 = ax1.imshow(t_img, cmap='inferno', vmin=0, vmax=vmax)
        ax1.set_title(f"真实标签 (样本{i})")
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # --- 第二列：模型预测 (Prediction) ---
        ax2 = axes[i, 1]
        im2 = ax2.imshow(p_img, cmap='inferno', vmin=0, vmax=vmax)
        
        # 统计指标
        p_max = np.max(p_img)
        p_mean = np.mean(p_img)
        ax2.set_title(f"预测结果 (Max={p_max:.2f})")
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # --- 第三列：误差图 (Diff) ---
        ax3 = axes[i, 2]
        diff = np.abs(t_img - p_img)
        # 误差显示通常不需要太大的量程，取 vmax 的一半或者由数据自动决定
        im3 = ax3.imshow(diff, cmap='coolwarm') 
        ax3.set_title(f"绝对误差 (MAE={np.mean(diff):.2f})")
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("result_preview.png", dpi=300)
    print(f"✅ 结果已保存为 result_preview.png，请打开查看！")
    
    # 🔥 核心诊断
    print("\n========== 🏥 诊断报告 ==========")
    # 检查所有样本中间帧的最大值平均数
    avg_pred_max = np.mean([np.max(pred_real[i, 0, time_idx]) for i in range(4)])
    
    if avg_pred_max < 0.1:
        print(f"❌ 【严重警告】预测值过小 (Avg Max={avg_pred_max:.4f})！")
        print("   模型可能发生了【全零崩塌】(All-Zero Collapse)。")
    else:
        print(f"✅ 模型预测正常 (Avg Max={avg_pred_max:.2f})，未发生崩塌。")
        print("   请检查生成的图片，确认纹理细节是否清晰。")

if __name__ == "__main__":
    visualize()