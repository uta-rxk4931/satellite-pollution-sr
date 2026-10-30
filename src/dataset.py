"""
Data for training the SR model.

Two sources:
  - fake (dummy) AOD maps, for testing the code without real data
  - real 64x64 patches from Rency's Track A code (data/sprint2_patches.npz)

Every sample is 4 tensors:
    lr       low-res input (1, 16, 16)
    lr_mask  1 = real value, 0 = cloud (1, 16, 16)
    hr       high-res target (1, 64, 64)
    hr_mask  1 = real value, 0 = cloud (1, 64, 64)
"""
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset


class AODDataset(Dataset):
    def __init__(self, lr, hr, mean, std):
        # lr, hr: numpy arrays with NaN where there are clouds
        # mean, std: from the TRAINING data, used to normalize everything
        self.mean = mean
        self.std = std

        # Masks: 1 where we have data, 0 where it's NaN (cloud)
        self.lr_mask = torch.tensor(np.isfinite(lr), dtype=torch.float32)
        self.hr_mask = torch.tensor(np.isfinite(hr), dtype=torch.float32)

        # Normalize (so values are around 0), then replace NaN with 0.
        # The masks tell the model and the loss which pixels are fake zeros.
        self.lr = torch.tensor(np.nan_to_num((lr - mean) / std), dtype=torch.float32)
        self.hr = torch.tensor(np.nan_to_num((hr - mean) / std), dtype=torch.float32)

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, i):
        return self.lr[i], self.lr_mask[i], self.hr[i], self.hr_mask[i]

    def to_aod(self, x):
        """Undo the normalization -> real AOD values."""
        return x * self.std + self.mean


def make_low_res(hr, scale):
    """
    Make the blurry input by averaging scale x scale blocks.
    (Dummy-data version only; the real one is part of Track A.)
    """
    n, c, h, w = hr.shape
    blocks = hr.reshape(n, c, h // scale, scale, w // scale, scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # numpy warns about all-NaN blocks
        lr = np.nanmean(blocks, axis=(3, 5))
    # If more than half the block is cloud, call the low-res pixel cloud too
    cloud_fraction = np.isnan(blocks).mean(axis=(3, 5))
    lr[cloud_fraction > 0.5] = np.nan
    return lr.astype(np.float32)


def make_fake_aod(num_maps, size=64, seed=0):
    """
    Fake AOD maps that look roughly like the real ones:
      - a background haze level
      - a few big smooth blobs (regional pollution)
      - a few thin streaks (like a highway or smoke plume)
      - sometimes a round cloud hole (NaN)
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    maps = np.zeros((num_maps, 1, size, size), dtype=np.float32)

    for i in range(num_maps):
        m = np.full((size, size), rng.uniform(0.05, 0.25))

        # Big smooth blobs
        for _ in range(rng.integers(2, 6)):
            cy, cx = rng.uniform(0, size, 2)
            width = rng.uniform(8, 25)
            height = rng.uniform(0.02, 0.3)
            m += height * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * width ** 2))

        # Thin streaks: long in one direction, about 1 pixel wide in the other.
        # These are smaller than a low-res pixel, so the model has to "recover" them.
        for _ in range(rng.integers(0, 4)):
            cy, cx = rng.uniform(0, size, 2)
            angle = rng.uniform(0, np.pi)
            along = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
            across = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
            m += rng.uniform(0.05, 0.2) * np.exp(-along ** 2 / (2 * 15 ** 2) - across ** 2 / (2 * 1.2 ** 2))

        # A little noise
        m += rng.normal(0, 0.005, m.shape)

        # Cloud hole in about 60% of the maps
        if rng.random() < 0.6:
            cy, cx = rng.uniform(0, size, 2)
            radius = rng.uniform(5, 15)
            m[(y - cy) ** 2 + (x - cx) ** 2 < radius ** 2] = np.nan

        maps[i, 0] = m
    return maps


def make_dummy_datasets(scale=4, num_train=2000, num_val=200):
    hr_train = make_fake_aod(num_train, seed=0)
    hr_val = make_fake_aod(num_val, seed=1)   # different seed = different maps

    # Normalization stats come from the training set only
    mean = float(np.nanmean(hr_train))
    std = float(np.nanstd(hr_train))

    train = AODDataset(make_low_res(hr_train, scale), hr_train, mean, std)
    val = AODDataset(make_low_res(hr_val, scale), hr_val, mean, std)
    return train, val

# Sprint 2 ======================================================================================
# Note a missing = 0 is dangerous, we need to normalize and make it sit around 0 with spread 1 to help NN learn 
def read_real_patches(path="data/sprint2_patches.npz"):

    data = np.load(path) # loads the patches
    scale = int(data["scale"]) # reads the size ratio between high-res and low-res (4, so 64 -> 16)
    hr_mask = data["mask"] > 0 # turns the 0/1 masks into true/false shape is now (n, 1, 64, 64) Checks if pixel does exist 
    n, c, h, w = hr_mask.shape 
    lr_mask = hr_mask.reshape(n, c, h // scale, scale, w // scale, scale).any(axis=(3, 5)) # scale the high res to a low resolution image

    hr = np.where(hr_mask, data["hr"], np.nan).astype(np.float32)
    lr = np.where(lr_mask, data["lr"], np.nan).astype(np.float32)
    return lr, hr, scale


# Packages all the real patches into one Pytorch-style dataset, in order to run a one-patch sanity test
def load_real_patches(path="data/sprint2_patches.npz"):
    """All real patches in one dataset (used by the one-patch overfit test)."""
    lr, hr, scale = read_real_patches(path)
    # Normalization only for training; to_aod() undoes it before scoring
    mean = float(np.nanmean(hr))
    std = float(np.nanstd(hr))
    return AODDataset(lr, hr, mean, std), scale
# ===============================================================================================

def overlaps(a, b):
    """
    True if patches a and b share a half- or quarter-patch of identical pixels.
    Track A cuts patches every half patch, so neighbours from the same day overlap.
    """
    a, b = a[0], b[0]
    half = a.shape[-1] // 2
    pairs = [(a[:, half:], b[:, :half]), (b[:, half:], a[:, :half]),     # side by side
             (a[half:], b[:half]), (b[half:], a[:half]),                   # above / below
             (a[half:, half:], b[:half, :half]), (b[half:, half:], a[:half, :half]),  # diagonal
             (a[half:, :half], b[:half, half:]), (b[half:, :half], a[:half, half:])]
    return any(np.array_equal(x, y, equal_nan=True) and np.isfinite(x).mean() > 0.5 for x, y in pairs)

# Sprint 3 train on real patches and compare with bicubic
def make_real_datasets(path="data/sprint2_patches.npz", val_fraction=0.2):
    """
    Temporary train/val split of the real patches, until Track A's split by year is ready.
    """
    lr, hr, scale = read_real_patches(path) # 144 patches, in DATE order

    cut = int(len(hr) * (1 - val_fraction)) # uses only 80% 115 patches

    window = 40  # overlapping patches are always close together in the file

    while cut > 0 and any(overlaps(hr[i], hr[j])
                          for i in range(max(0, cut - window), cut)
                          for j in range(cut, min(len(hr), cut + window))):
        cut -= 1

    # Normalization stats come from the training set only
    mean = float(np.nanmean(hr[:cut]))
    std = float(np.nanstd(hr[:cut]))

    train = AODDataset(lr[:cut], hr[:cut], mean, std) # 112 patches
    val = AODDataset(lr[cut:], hr[cut:], mean, std) # 32 patches
    return train, val, scale


# Year split: the real train / val / test sets (Track A, combine_and_split.py)
def make_year_datasets(folder="data/split_data"):
    """
    Train 2018-21, val 2022, test 2023. Different years never share pixels, so no
    overlap check is needed. Normalization uses Track A's train-only mean/std.
    """
    lr_train, hr_train, scale = read_real_patches(f"{folder}/train.npz")
    lr_val, hr_val, _ = read_real_patches(f"{folder}/val.npz")
    lr_test, hr_test, _ = read_real_patches(f"{folder}/test.npz")

    stats = np.load(f"{folder}/train_stats.npz")
    mean, std = float(stats["mean"]), float(stats["std"])

    train = AODDataset(lr_train, hr_train, mean, std)
    val = AODDataset(lr_val, hr_val, mean, std)
    test = AODDataset(lr_test, hr_test, mean, std)
    return train, val, test, scale
