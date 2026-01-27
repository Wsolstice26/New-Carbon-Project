# E:\superResulotion\Code\Carbon_SR_Project\config.py

CONFIG = {
    # ==========================
    # 1. 路径设置
    # ==========================
    "data_dir": r"C:\superResulotion\Train_Data_Yearly_Coords",
    "split_config": r"E:\superResulotion\Configs\split_config.json",
    "save_dir": r"E:\superResulotion\Checkpoints\DST_Experiment_01",
    
    # ==========================
    # 2. 训练超参数
    # ==========================
    # 🔥 修改：Batch Size 改为 32 (如显存不足可回调至 16 或 8)
    "batch_size": 48,
    
    # 学习率
    "lr": 1e-4,
    
    "epochs": 100, # 可以设大一点，反正有早停
    
    # ==========================
    # 3. 高级功能设置 (新增)
    # ==========================
    
    # 🔥 是否断点续训？
    # 如果设为 True，程序会自动去 save_dir 找最新的 epoch_xx.pth 继续练
    # 如果设为 False，每次都从头开始
    "resume": False, 
    
    # 🔥 早停 (Early Stopping) 耐心值
    # 如果验证集 Loss 连续 15 个 Epoch 不下降，就停止训练
    "patience": 15,
    
    # Windows 下必须为 0
    "num_workers": 8,
    "save_freq": 5,
}