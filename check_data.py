import numpy as np

y_path = "/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_120/Y_2014.npy"
Y = np.load(y_path)

if Y.ndim == 4:
    Y2 = Y[:, 0]
elif Y.ndim == 3:
    Y2 = Y
else:
    raise ValueError(Y.shape)

N, H, W = Y2.shape
scale = 10
assert H % scale == 0 and W % scale == 0

# 找到第一个非零 patch
flat_max = Y2.reshape(N, -1).max(axis=1)
nz_idx = np.where(flat_max > 0)[0]

if nz_idx.size == 0:
    print("❌ 整个 Y_2014.npy 没有任何非零 patch（全局 max==0）")
    print("Global min/max:", Y2.min(), Y2.max())
    raise SystemExit(0)

i = int(nz_idx[0])
y0 = Y2[i]
print(f"✅ 找到第一个非零 patch: idx={i}, patch_min/max={y0.min()}/{y0.max()}")

block_means = y0.reshape(H//scale, scale, W//scale, scale).mean(axis=(1,3))
uniq = np.unique(block_means)

print("\n12×12 block means:")
print(block_means)
print("\nUnique block mean count:", uniq.shape[0])
print("Min / Max block mean:", block_means.min(), block_means.max())

if uniq.shape[0] > 1:
    print("\n❗结论：12×12 block 之间存在空间差异（block 均值不同）")
else:
    print("\n✅结论：12×12 block 之间没有差异（block 均值完全相同）")


import numpy as np

Y = np.load("/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_120/Y_2014.npy")[:,0]
idx = 4
y0 = Y[idx]

# 取一个非零 block，比如 (row=7,col=11) 这种你刚才明显非零的位置
r, c = 7, 11
blk = y0[r*10:(r+1)*10, c*10:(c+1)*10]
print("block min/max:", blk.min(), blk.max(), "std:", blk.std())
