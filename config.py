# E:\superResulotion\Code\Carbon_SR_Project\config.py

CONFIG = {
    # ==========================
    # 1. 路径设置
    # ==========================
    "data_dir": r"/workspace/Train_Data_Yearly_Coords",
    "split_config": r"/workspace/Configs/split_config.json",
    "save_dir": r"/workspace/Checkpoints/DST_Experiment_01",
    
    # ==========================
    # 2. 训练超参数
    # ==========================
    # 🔥 16G 显存的黄金甜点配置
    "batch_size": 32,
    
    # 学习率
    "lr": 1e-4,
    
    "epochs": 100,
    
    # ==========================
    # 3. 其他设置
    # ==========================
    "resume": False, 
    "patience": 15,
    
    # 🔥🔥🔥 核心修改：AMD + Windows 必须设为 0，否则速度起不来
    "num_workers": 8,
    
    "save_freq": 5,
}