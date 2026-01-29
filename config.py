CONFIG = {
    # 路径不变...
    "data_dir": "/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_Coords",
    "split_config": "/home/wdc/Carbon-Emission-Super-Resolution/Configs/split_config.json",
    "save_dir": "/home/wdc/Carbon-Emission-Super-Resolution/Checkpoints/DST_Experiment_01",
    
    # Batch Size 20  16G显存下的选择
    "batch_size": 20, 
    
    # 🔴 降学习率：1e-4 可能太冲了，由 5e-5 开始比较稳
    "lr": 5e-5,
    
    "epochs": 100,
    "resume": False,    
    "patience": 15,
    "num_workers": 8,
    "save_freq": 5,
}