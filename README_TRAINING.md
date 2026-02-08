下面是一份**更简单**、够用的 `README.md`（中文），直接复制到项目根目录即可。

---

# Carbon-Emission-Super-Resolution（DSTCarbonFormer）

基于 **低分辨率主变量（Main）+ 多源辅助因子（Aux）** 的碳排放时空超分辨率重建。
核心代码：训练脚本 `train.py`，数据加载 `dataset.py`，模型 `network.py`，损失 `losses.py`，组件 `blocks.py`，配置 `config.py`。

---

## 1. 目录结构（最关键的）

```
Carbon-Emission-Super-Resolution/
├─ train.py                 # 训练/验证/测试主入口（含指标与日志）
├─ config.py                # 实验配置（自动创建 Checkpoints 目录）
├─ data/
│  └─ Train_Data_Yearly_120/   # 示例：按年份存放 X_y.npy / Y_y.npy
├─ data/dataset.py          # DualStreamDataset：加载、缓存、统计量(nz_ratio/cv_log)
├─ models/
│  ├─ network.py            # DSTCarbonFormer 网络主体
│  ├─ losses.py             # HybridLoss：WeightedL1 + MSE + Sparse + Entropy (auto-weight)
│  └─ blocks.py             # 组件：SFT、多尺度、MoE、DualAttention、BiMamba等
└─ Checkpoints/
   └─ Run_YYYYMMDD_HHMM_...  # 训练输出（log、autosave、best_model、配置等）
```

---

## 2. 环境依赖

* Python 3.10+
* PyTorch
* numpy, tqdm
* mamba_ssm（模型中使用 Mamba / BiMamba）

安装示例：

```bash
pip install torch numpy tqdm
pip install mamba-ssm
```

---

## 3. 数据格式（约定）

* 每个年份两份文件：

  * `X_{year}.npy`：Aux（多通道辅助因子）
  * `Y_{year}.npy`：GT（高分辨率目标）
* 代码默认从 `data/Train_Data_Yearly_120/` 读取。

---

## 4. 训练

1）先在 `config.py` 里改参数（如 `BATCH_SIZE / DIM / EPOCHS / LR` 等）。

2）启动训练：

```bash
python train.py
```

输出会写入 `Checkpoints/Run_.../`，包括：

* `training_log.csv`
* `autosave_latest.pth`
* `best_model.pth`
* `experiment_config.json`

---

## 5. Resume（断点续训）

在 `config.py` 中设置：

```python
RESUME = True
```

会自动加载最近一次匹配实验的 `autosave_latest.pth`。

---

## 6. 简要说明

* **模型**：`network.py` 中 `DSTCarbonFormer`。
* **损失**：`losses.py` 中 `HybridLoss`，核心监督在 1km 尺度（把 100m 预测 sum-pool 到 1km 再比对）。

---

如果你希望再“更简单”（比如只保留：一句话简介 + 安装 + 运行三行命令），我也可以给你一版“极简 README”。

