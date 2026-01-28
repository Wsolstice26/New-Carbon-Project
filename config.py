# config.py

CONFIG = {
    # 路径保持不变
    "data_dir": "/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_Coords",
    "split_config": "/home/wdc/Carbon-Emission-Super-Resolution/Configs/split_config.json",
    "save_dir": "/home/wdc/Carbon-Emission-Super-Resolution/Checkpoints/DST_Experiment_01",
    
    # ==========================
    # 🔴 关键修改 1: 显存救星
    # ==========================
    # 关闭 AMP 后显存占用翻倍，必须降到 8 (甚至 4)
    "batch_size": 12, 
    
    # ==========================
    # 🔴 关键修改 2: 稳定训练
    # ==========================
    # 降低学习率，防止梯度爆炸
    "lr": 5e-5,  
    
    "epochs": 100,
    "resume": True,    
    "patience": 15,
    "num_workers": 8,
    "save_freq": 5,
}