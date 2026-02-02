import numpy as np

y_path = "/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_120/Y_2014.npy"
Y = np.load(y_path)[:, 0]   # (N,120,120)

N, H, W = Y.shape
s = 10
hg, wg = H // s, W // s

# 每个 patch 的 max，找非零 patch
patch_max = Y.reshape(N, -1).max(axis=1)
nz_patches = np.where(patch_max > 0)[0]
print("nonzero patches:", len(nz_patches), "/", N)

rng = np.random.default_rng(2026)

# 抽样参数（可以改大）
num_patches = min(30, len(nz_patches))
blocks_per_patch = 20

stds = []
means = []
picked = rng.choice(nz_patches, size=num_patches, replace=False)

for idx in picked:
    y = Y[idx]
    # 先算 12×12 block mean，找非零 block
    bm = y.reshape(hg, s, wg, s).mean(axis=(1,3))
    nz_blocks = np.argwhere(bm > 0)
    if nz_blocks.size == 0:
        continue

    pick_blocks = nz_blocks[rng.choice(len(nz_blocks), size=min(blocks_per_patch, len(nz_blocks)), replace=False)]
    for r, c in pick_blocks:
        blk = y[r*s:(r+1)*s, c*s:(c+1)*s]
        stds.append(float(blk.std()))
        means.append(float(blk.mean()))

stds = np.array(stds)
means = np.array(means)

print("sampled blocks:", len(stds))
print("block std:  min/median/mean/max =", stds.min(), np.median(stds), stds.mean(), stds.max())

# 判断“近似常数”的比例：std <= 1e-6
const_ratio = (stds <= 1e-6).mean() if len(stds) else 0
print("const block ratio (std<=1e-6):", const_ratio)

# 给一个更宽松阈值：相对波动 std/mean
rel = stds / (means + 1e-12)
print("relative std (std/mean): min/median/mean/max =",
      rel.min(), np.median(rel), rel.mean(), rel.max())
print("near-constant ratio (std/mean <= 1e-6):", (rel <= 1e-6).mean() if len(rel) else 0)


import numpy as np

Y = np.load("/home/wdc/Carbon-Emission-Super-Resolution/data/Train_Data_Yearly_120/Y_2014.npy")[:,0]
s = 10

# 取一个高值 patch，避免全0
idx = int(np.argmax(Y.reshape(Y.shape[0], -1).max(axis=1)))
y = Y[idx]

# 找一个非零 block
bm = y.reshape(12, s, 12, s).mean(axis=(1,3))
r, c = np.argwhere(bm > 0)[0]

blk = y[r*s:(r+1)*s, c*s:(c+1)*s]
print("patch idx:", idx, "block(rc):", (r,c))
print("block pixel value (approx):", float(blk.mean()))
print("block sum:", float(blk.sum()))
print("block mean:", float(blk.mean()))
print("sum/mean:", float(blk.sum() / (blk.mean() + 1e-12)))
