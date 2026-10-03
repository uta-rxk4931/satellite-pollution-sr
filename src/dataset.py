"""
Data for training the SR model.

Right now we use fake (dummy) AOD maps so the model can be tested before
the real 64x64 patches exist (those come from Rency's Track A code).

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
