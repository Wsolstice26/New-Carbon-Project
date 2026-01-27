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
    # 🔥 您要求的修改：Batch Size 改为 8
    # 16G 显存跑 128x128 的图，Batch=8 是完全没问题的
    "batch_size": 8,
    
    # 学习率：Batch变大了，学习率保持 1e-4 是比较稳妥的
    # 如果 Loss 下降太慢，后期可以改成 2e-4
    "lr": 1e-4,
    
    "epochs": 50,
    
    # Windows 下必须为 0，否则容易报错
    "num_workers": 0,
    
    # ==========================
    # 3. 其他设置
    # ==========================
    "save_freq": 5,
}