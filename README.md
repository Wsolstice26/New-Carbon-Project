# 🌍 Carbon Emission Super-Resolution (DSTCarbonFormer)

**基于 Mamba + MoE + 自适应混合损失的碳排放时空超分辨率重建框架**

本项目实现了一个深度学习模型，旨在利用多源辅助数据（如夜间灯光、路网、NDVI等）将粗糙的碳排放数据重建为高分辨率的精细分布图。模型在 **v1.7** 版本中采用了 **DSTCarbonFormer** (Dual-Stream Spatio-Temporal Carbon Transformer) 架构，并集成了 **自适应权重损失 (Auto-Weighted Loss)** 和 **Balanced MAE 导向优化** 机制，彻底解决了“数值准确性”与“纹理平滑度”难以平衡的痛点，以及背景像素过多导致的梯度稀释问题。

---

## ✨ 核心特性

* **双流架构 (Dual-Stream) & SFT 融合**:
    * **辅助流 (Aux Stream)**: 处理高分辨率多源辅助数据 (9通道: NTL, Road, Water, NDVI, etc.)。
    * **主流 (Main Stream)**: 处理低分辨率碳排放数据。
    * **SFT 融合**: 使用空间特征变换 (SFT) 层将辅助流的纹理信息动态注入主流中。

* **前沿计算架构 (Mamba & MoE)**:
    * **Mamba (SSM)**: 引入状态空间模型替代传统 Transformer，在捕获超长程时空依赖的同时，保持线性计算复杂度。
    * **混合专家系统 (MoE)**: 针对成都市不同功能区（高度城市化中心 vs 边缘生态区）进行动态计算资源分配。

* **⚖️ 平衡自适应损失 (v1.7 New)**:
    * **平衡掩码机制 (Balanced Masked Loss)**: 强制“城市区域”与“背景区域”在梯度计算中占比 **1:1**，解决背景 0 值过多导致的 Loss 虚低问题。
    * **正数权重自适应**: 采用指数归一化 (Exponential Normalization) 替代对数方差，保证 Loss 恒为正数，训练曲线更直观。
    * **自动加权**: 自动学习 `Pixel`、`SSIM` 和 `TV` Loss 之间的最佳比例。

* **🎯 Balanced MAE 导向优化 (v1.7 New)**:
    * 摒弃传统的 Global MAE，采用 **Balanced MAE (50% City + 50% Bg)** 作为早停 (Early Stopping) 的唯一依据，强迫模型攻克高排放难点区域。

* **频率硬约束 (FFT Constraint)**:
    * 集成 **SEN2SR 频率域约束**，通过 3D FFT 强制模型保留低频数值准确性，并有效缓解 1km 格子边缘的“棋盘效应”。

* **硬件适配与稳定性 (AMD ROCm Optimized)**:
    * 针对 **AMD Radeon RX 9060 XT** 显存管理进行专项优化。
    * **NaN 免疫机制**: 通过梯度裁剪与显存碎块化回收，解决了在大 Batch 下的数值溢出问题。

---

## 📂 目录结构

```text
Carbon_SR_Project/
├── data/               # 数据集处理模块
│   └── dataset.py      # DualStreamDataset 定义 (含 Log 归一化/滑窗逻辑)
├── models/             # 模型定义
│   ├── network.py      # DSTCarbonFormer 主模型 (v1.7: Mamba+MoE)
│   ├── blocks.py       # SFT层, 多尺度块, FFT约束层等
│   └── losses.py       # HybridLoss (v1.7: 平衡掩码 + 正数权重)
├── check_*.py          # 数据自检脚本 (data_stats, granular_stats, everything)
├── config.py           # 全局配置文件 (路径、超参数)
├── create_split.py     # 数据集划分脚本
├── train.py            # 训练脚本 (v1.7: Balanced MAE 早停 + 实时监控)
├── predict_vis.py      # 预测可视化脚本 (生成对比图)
├── inference.py        # 全量推断脚本 (生成最终 .npy 结果)
├── requirements.txt    # 依赖库列表
└── README.md           # 项目说明文档

```

---

## 🛠️ 环境安装

1. **克隆项目**

```bash
git clone [https://github.com/Wsolstice26/Carbon-Emission-Super-Resolution.git](https://github.com/Wsolstice26/Carbon-Emission-Super-Resolution.git)
cd Carbon-Emission-Super-Resolution

```

2. **激活环境**
建议使用 Python 3.12+ 及对应的 ROCm PyTorch 环境。

```bash
source /home/wdc/mamba_env/bin/activate

```

---

## 📦 数据准备

数据应存放于 `config.py` 中 `data_dir` 指定的目录。

### 文件命名格式

* **输入特征 (Aux)**: `X_{Year}.npy` (例如 `X_2014.npy`)
* 形状: `[N, 9, 128, 128]`
* 内容: 9个波段的辅助地理数据。


* **标签/粗糙数据 (Main)**: `Y_{Year}.npy` (例如 `Y_2014.npy`)
* 形状: `[N, 1, 128, 128]`
* 内容: 真实的碳排放数据。



### 数据预处理

运行以下命令生成数据集划分文件 (`split_config.json`)：

```bash
python create_split.py

```

*该脚本会将数据按 8:1:1 划分为训练集、验证集和测试集。*

---

## 🚀 运行步骤

### 1. 系统自检 (可选但推荐)

在开始漫长的训练前，检查显存占用（针对 16GB VRAM）和数据完整性：

```bash
python check_everything.py

```

### 2. 模型训练

启动训练脚本。v1.7 版本会自动平衡城市与背景的权重，并实时显示 MAE。

```bash
python train.py

```

* **配置**: 修改 `config.py` 中的 `batch_size`, `lr`, `epochs` 等参数。
* **输出**: 模型权重会保存在 `Checkpoints/` 目录下。

### 3. 结果可视化

训练完成后，查看模型在验证集上的表现（生成对比图）：

```bash
python predict_vis.py

```

* 生成的 `result_preview.png` 将展示：真实标签 vs 预测结果 vs 误差热力图。

### 4. 全量推断

生成 2014-2023 全年份的高分辨率碳排放数据：

```bash
python inference.py

```

* 结果将保存为 `Pred_{Year}.npy` 文件。

---

## ⚙️ 关键配置 (config.py)

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `batch_size` | **20** | 针对 RX 9060 XT 优化的最佳值 (VRAM ~10GB) |
| `lr` | **5e-5** | 初始学习率 (Loss 参数也会随此更新) |
| `resume` | `True` | 是否开启断点续训 |
| `patience` | 15 | 基于 **Balanced MAE** 的早停耐心值 |
| `save_freq` | 5 | 每隔多少轮保存一次检查点 |

---

## 📊 性能指标 (v1.7 - Chengdu)

模型训练过程会实时监控以下指标：

* **Balanced MAE**: 核心指标 (0.5 * City_MAE + 0.5 * Bg_MAE)，防止被背景 0 值稀释。
* **City MAE (Non-Zero)**: 核心城区/工业区的高排放点预测误差。
* **Weights**: 实时显示的 Loss 权重分配 (Pixel / SSIM / TV)。

---

## 📝 引用与致谢

本项目参考了 SEN2SR 的频率约束思想与 SFT (Spatial Feature Transform) 融合机制。
如果您在研究中使用了本项目，请引用相关代码库。

---

*Last Updated: 2026-01-29 (v1.7 Release)*