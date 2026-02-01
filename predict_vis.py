import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from torch.utils.data import DataLoader

# ==========================================
# 🛠️ 导入项目模块
# ==========================================
# 确保能找到当前目录下的模块
sys.path.append(os.getcwd())

try:
    from data.dataset import DualStreamDataset
    from models.network import DSTCarbonFormer
    from config import CONFIG
except ImportError:
    print("❌ 导入失败！请确保你在项目根目录下运行此脚本。")
    print("   例如: /home/wdc/mamba_env/bin/python predict_vis.py")
    sys.exit(1)

# ==========================================
# 🎨 绘图风格设置 (适配 Linux 服务器)
# ==========================================
# Linux 服务器通常没有 SimHei，使用 DejaVu Sans 既通用又美观
plt.rcParams['font.family'] = 'DejaVu Sans' 
plt.rcParams['axes.unicode_minus'] = False

def predict_and_visualize():
    # 1. 环境与配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*50)
    print(f"🔥 预测环境: {device}")
    print(f"📂 实验目录: {CONFIG['save_dir']}")
    print(f"📏 模型配置: Dim={CONFIG['dim']}, Patch={CONFIG['patch_size']}")
    print("="*50)

    # 2. 寻找最佳权重
    # 优先找 best_model.pth，如果没有则找 latest.pth
    ckpt_path = os.path.join(CONFIG['save_dir'], "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"⚠️ 未找到 best_model.pth，尝试使用 latest.pth...")
        ckpt_path = os.path.join(CONFIG['save_dir'], "latest.pth")
    
    if not os.path.exists(ckpt_path):
        print(f"❌ 错误: 在 {CONFIG['save_dir']} 下未找到任何模型权重！")
        return

    # 3. 加载模型
    print(f"📦 正在加载模型: {os.path.basename(ckpt_path)}")
    try:
        # 🔥 关键: 必须传入 dim 参数，确保结构与训练时一致
        model = DSTCarbonFormer(aux_c=9, main_c=1, dim=CONFIG['dim']).to(device)
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        # 兼容处理: 检查是保存了完整 checkpoint 还是只保存了 state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 建议: 检查 config.py 中的 DIM 是否与训练时的设置一致 (48 或 64)。")
        return

    # 4. 加载验证数据
    print("📦 正在加载验证集数据...")
    try:
        # 使用 val 模式，随机打乱抽取 4 张
        val_ds = DualStreamDataset(CONFIG['data_dir'], CONFIG['split_config'], 'val', time_window=CONFIG['time_window'])
        val_dl = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=0)
        
        # 获取一个 Batch
        aux, main, target = next(iter(val_dl))
        aux = aux.to(device)
        main = main.to(device)
        target = target.to(device)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 5. 执行推理
    print("🔮 正在进行超分推理...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            pred = model(aux, main)

    # 6. 数据还原 (反归一化)
    # 从 Log 域还原到真实物理量 (吨)
    norm_factor = CONFIG['norm_factor']
    
    pred_real = torch.expm1(pred * norm_factor).clamp(min=0).cpu().numpy()
    target_real = torch.expm1(target * norm_factor).clamp(min=0).cpu().numpy()
    input_real = torch.expm1(main * norm_factor).clamp(min=0).cpu().numpy() # 低清输入也还原
    
    # 7. 绘图可视化 (4行 x 4列)
    print("🎨 正在绘制对比图...")
    save_path = os.path.join(CONFIG['save_dir'], "result_preview.png")
    
    fig, axes = plt.subplots(4, 4, figsize=(22, 20))
    # 选取时间窗口的中间帧 (例如 T=3 时取 index 1)
    time_idx = CONFIG['time_window'] // 2
    
    for i in range(4):
        # 提取当前样本的中间帧 [Channel=0, Time=mid, H, W]
        in_img = input_real[i, 0, time_idx]
        t_img = target_real[i, 0, time_idx]
        p_img = pred_real[i, 0, time_idx]
        
        # 动态设置显示范围 (以真值为基准，防止过曝或过暗)
        vmax = max(np.max(t_img), np.max(p_img), 1.0)
        
        # --- 第一列: Low Res Input (马赛克输入) ---
        ax1 = axes[i, 0]
        im1 = ax1.imshow(in_img, cmap='inferno', vmin=0, vmax=vmax)
        ax1.set_title(f"Low Res Input\n(Mosaic Data)", fontsize=10)
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # --- 第二列: Prediction (我们的结果) ---
        ax2 = axes[i, 1]
        im2 = ax2.imshow(p_img, cmap='inferno', vmin=0, vmax=vmax)
        p_max = np.max(p_img)
        ax2.set_title(f"Ours Prediction\nMax={p_max:.1f}", fontsize=10, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # --- 第三列: Ground Truth (高清真值) ---
        ax3 = axes[i, 2]
        im3 = ax3.imshow(t_img, cmap='inferno', vmin=0, vmax=vmax)
        ax3.set_title(f"Ground Truth\n(High Res)", fontsize=10)
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # --- 第四列: Error Map (误差热力图) ---
        ax4 = axes[i, 3]
        # 计算绝对误差
        diff = np.abs(t_img - p_img)
        mae = np.mean(diff)
        # 误差图使用 coolwarm 色系 (蓝=低误差, 红=高误差)
        im4 = ax4.imshow(diff, cmap='coolwarm') 
        ax4.set_title(f"Absolute Error\nMAE={mae:.2f}", fontsize=10)
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 绘图完成！图片已保存至:\n   👉 {save_path}")
    
    # 8. 简易诊断报告
    print("\n========== 🏥 模型诊断报告 ==========")
    avg_pred_max = np.mean([np.max(pred_real[i, 0, time_idx]) for i in range(4)])
    avg_gt_max = np.mean([np.max(target_real[i, 0, time_idx]) for i in range(4)])
    
    print(f"📊 抽样统计 (Avg Max Value):")
    print(f"   GT (真值)   : {avg_gt_max:.4f}")
    print(f"   Pred (预测) : {avg_pred_max:.4f}")
    
    if avg_pred_max < 0.1 and avg_gt_max > 1.0:
        print(f"❌ [严重警告] 预测值接近全零，可能发生了模型崩塌！")
    elif avg_pred_max > avg_gt_max * 5:
        print(f"⚠️ [警告] 预测值异常偏大，可能存在梯度爆炸。")
    else:
        print(f"✅ 数值范围正常。请打开 result_preview.png 查看纹理细节。")
    print("="*50)

if __name__ == "__main__":
    predict_and_visualize()