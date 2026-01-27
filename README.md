# 🌍 Carbon Emission Super-Resolution (DSTCarbonFormer)

**基于双流时空Transformer的碳排放超分辨率重建框架**

本项目实现了一个深度学习模型，旨在利用多源辅助数据（如夜间灯光、路网、NDVI等）将粗糙的碳排放数据重建为高分辨率的精细分布图。模型采用了 **DSTCarbonFormer** (Dual-Stream Spatio-Temporal Carbon Transformer) 架构，并集成了 **SEN2SR 频率硬约束** 和 **混合物理损失函数**，以确保重建结果在数值上的准确性和空间纹理上的真实性。

---

## ✨ 核心特性

* **双流架构 (Dual-Stream)**:
* **辅助流 (Aux Stream)**: 处理高分辨率的多源辅助数据 (9通道: NTL, Road, Water, NDVI, etc.)。
* **主流 (Main Stream)**: 处理低分辨率/粗糙的碳排放数据。
* **SFT 融合**: 使用空间特征变换 (SFT) 层将辅助流的纹理信息注入到主流中。


* **时空感知 (Spatio-Temporal)**:
* 采用 `T=3` 的滑动窗口输入，捕捉碳排放的时序演变规律。
* 集成 **Efficient Global Context Block** 进行全局信息建模。


* **频率硬约束 (FFT Constraint)**:
* 引入 SEN2SR 的频率域约束，强制模型保留低频的宏观数值准确性，同时恢复高频纹理细节。


* **混合损失函数 (Hybrid Loss)**:
* 组合了 `Pixel Loss` (加权), `SSIM Loss`, `Edge Loss`, `Consistency Loss` (物理守恒) 和 `TV Loss` (平滑去噪)。
* 引入 **Weighted Mask** 机制，重点关注高碳排放区域，解决样本不平衡导致的“全零崩塌”问题。



---

## 📂 目录结构

```text
Carbon_SR_Project/
├── data/               # 数据集处理模块
│   └── dataset.py      # DualStreamDataset 定义 (含 Log 归一化/滑窗逻辑)
├── models/             # 模型定义
│   ├── network.py      # DSTCarbonFormer 主模型
│   ├── blocks.py       # SFT层, 多尺度块, FFT约束层等
│   └── losses.py       # HybridLoss 混合损失函数
├── check_*.py          # 数据自检脚本 (data_stats, granular_stats, everything)
├── config.py           # 全局配置文件 (路径、超参数)
├── create_split.py     # 数据集划分脚本
├── train.py            # 训练脚本 (支持断点续训、早停、混合精度)
├── predict_vis.py      # 预测可视化脚本 (生成对比图)
├── inference.py        # 全量推断脚本 (生成最终 .npy 结果)
├── requirements.txt    # 依赖库列表
└── README.md           # 项目说明文档

```

---

## 🛠️ 环境安装

1. **克隆项目**
```bash
git clone https://github.com/Wsolstice26/Carbon-Emission-Super-Resolution.git
cd Carbon-Emission-Super-Resolution

```


2. **安装依赖**
建议使用 Python 3.8+ 和 CUDA 环境。
```bash
pip install -r requirements.txt

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

在开始漫长的训练前，检查显存占用和数据完整性：

```bash
python check_everything.py

```

### 2. 模型训练

启动训练脚本。支持自动断点续训（如果意外中断，再次运行即可接着跑）。

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
| `batch_size` | 32 | 根据显存调整 (16G显存推荐 16-32) |
| `lr` | 1e-4 | 初始学习率 (配合 CosineAnnealingLR) |
| `resume` | True | 是否开启断点续训 |
| `patience` | 15 | 早停机制的耐心值 (Epochs) |
| `save_freq` | 5 | 每隔多少轮保存一次检查点 |

---

## 📊 性能指标

模型训练过程会实时监控以下指标：

* **Train/Val Loss**: 混合损失值。
* **Real MAE (吨)**: 还原到真实物理空间后的平均绝对误差。

---

## 📝 引用与致谢

本项目参考了 SEN2SR 的频率约束思想与 SFT (Spatial Feature Transform) 融合机制。
如果您在研究中使用了本项目，请引用相关代码库。

---

*Last Updated: 2026-01-26*