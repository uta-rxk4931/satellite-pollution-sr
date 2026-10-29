import numpy as np
from pathlib import Path

# Folder that contains patches_2018.npz ... patches_2023.npz
PATCH_DIR = r"path\to\your\FINAL patches 2018_2023"  
OUT_DIR = Path(PATCH_DIR) / "split_data"
OUT_DIR.mkdir(exist_ok=True)


years = [2018, 2019, 2020, 2021, 2022, 2023]
data = {}
for y in years:
    path = Path(PATCH_DIR) / f"patches_{y}.npz"
    data[y] = np.load(path)
    print(f"{y}: {len(data[y]['hr'])} patches")

def concat(years_list):
    return {
        "lr":   np.concatenate([data[y]["lr"]   for y in years_list]),
        "hr":   np.concatenate([data[y]["hr"]   for y in years_list]),
        "mask": np.concatenate([data[y]["mask"] for y in years_list]),
        "year": np.concatenate([data[y]["year"] for y in years_list]),
    }

train_years = [2018, 2019, 2020, 2021]
val_years   = [2022]
test_years  = [2023]

train = concat(train_years)
val   = concat(val_years)
test  = concat(test_years)

# Train-only mean / std (on valid HR pixels)
train_hr = train["hr"]
train_mask = train["mask"] > 0.5
mean = train_hr[train_mask].mean()
std  = train_hr[train_mask].std()

print(f"\nTrain patches: {len(train['hr'])}")
print(f"Val patches:   {len(val['hr'])}")
print(f"Test patches:  {len(test['hr'])}")
print(f"Train-only mean: {mean:.4f}")
print(f"Train-only std:  {std:.4f}")

# Save
np.savez_compressed(OUT_DIR / "train.npz", **train, scale=4)
np.savez_compressed(OUT_DIR / "val.npz",   **val,   scale=4)
np.savez_compressed(OUT_DIR / "test.npz",  **test,  scale=4)
np.savez(OUT_DIR / "train_stats.npz", mean=mean, std=std)

print(f"\nSaved to {OUT_DIR}")
print("Files: train.npz, val.npz, test.npz, train_stats.npz")