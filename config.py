CONFIG = {
    # ==========================
    # 1. 路径设置 (针对 Docker 环境)
    # ==========================
    "data_dir": "/train_data",
    "split_config": "/workspace/Configs/split_config.json",
    "save_dir": "/workspace/Checkpoints/DST_Experiment_01",
    
    # ==========================
    # 2. 训练超参数
    # ==========================
    "batch_size": 32, # RX 9060 XT 16G 显存的推荐配置
    "lr": 1e-4,
    "epochs": 100,
    
    # ==========================
    # 3. 其他设置
    # ==========================
    "resume": False, 
    "patience": 15,
    
    "num_workers": 8,
    "save_freq": 5,
}