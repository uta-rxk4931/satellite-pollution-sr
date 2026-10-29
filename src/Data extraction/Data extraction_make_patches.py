import numpy as np
from pathlib import Path

INPUT_DIR = r"path\to\your\dfw_tifs_2023"   # folder with the _dfw.npy files
OUTPUT_NPZ = r"path\to\your\patches_2023.npz"

PATCH = 64
SCALE = 4
MIN_VALID = 0.6
STRIDE = PATCH // 2

npy_files = sorted(Path(INPUT_DIR).glob("*_dfw.npy"))
print(f"Found {len(npy_files)} daily maps")

all_lr, all_hr, all_mask, all_year = [], [], [], []

for f in npy_files:
    aod = np.load(f).astype(np.float32)

    year = int(f.name.split(".A")[1][:4])
    
    valid = np.isfinite(aod) & (aod > -0.1) & (aod < 5)
    aod = np.where(valid, aod, 0.0)
    
    h, w = aod.shape
    for r in range(0, h - PATCH + 1, STRIDE):
        for c in range(0, w - PATCH + 1, STRIDE):
            hr = aod[r:r+PATCH, c:c+PATCH]
            m = valid[r:r+PATCH, c:c+PATCH]
            if m.mean() < MIN_VALID:
                continue
            
           
            hs, ws = PATCH // SCALE, PATCH // SCALE
            hr_b = hr.reshape(hs, SCALE, ws, SCALE)
            m_b = m.reshape(hs, SCALE, ws, SCALE)
            count = m_b.sum(axis=(1, 3))
            lr = (hr_b * m_b).sum(axis=(1, 3)) / np.maximum(count, 1)
            lr[count == 0] = 0
            
            all_hr.append(hr)
            all_mask.append(m.astype(np.float32))
            all_lr.append(lr.astype(np.float32))
            all_year.append(year)

hr = np.stack(all_hr)[:, None]
mask = np.stack(all_mask)[:, None]
lr = np.stack(all_lr)[:, None]
year = np.array(all_year)

print(f"Patches: {len(hr)}")
print(f"Years: {np.unique(year)}")

np.savez_compressed(OUTPUT_NPZ, lr=lr, hr=hr, mask=mask, year=year, scale=SCALE)
print("Saved", OUTPUT_NPZ)