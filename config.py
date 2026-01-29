CONFIG = {
    # 路径不变...
    "data_dir": "/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_Coords",
    "split_config": "/home/wdc/Carbon-Emission-Super-Resolution/Configs/split_config.json",
    "save_dir": "/home/wdc/Carbon-Emission-Super-Resolution/Checkpoints/DST_Experiment_01",
    
    # 保持 Batch Size 24 或 16 (根据显存情况)
    "batch_size": 16, 
    
    # 🔴 降学习率：1e-4 可能太冲了，由 5e-5 开始比较稳
    "lr": 5e-5,
    
    "epochs": 100,
    "resume": True,    
    "patience": 15,
    "num_workers": 8,
    "save_freq": 5,
}