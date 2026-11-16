"""
RQ4: does sharpened 1 km AOD correlate more strongly with EPA PM2.5
than the original coarse 4 km AOD? (year-split, full 2023 test year)

Compares six AOD versions at each monitor:
4 km coarse, bicubic, EDSR, U-Net, SwinIR, true 1 km.
Reports all-days R2 and within-day R2.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import rasterio
import torch

from src import metrics
from src.models.sr_net import build_model
from src.rq4_first_look import nanmean, r2, within_day

SCALE = 4
PATCH = 64          # high-res patch size the models were trained on (low-res = 16)
MIN_VALID = 0.6     # keep a patch if at least 60% of its pixels are real
MODELS = ["edsr", "unet", "swinir"]


def read_valid(path):
    """One daily map, cleaned the same way as the patch notebook. Returns aod (0 = missing) and valid."""
    with rasterio.open(path) as src:
        aod = src.read(1).astype(np.float32)
        transform_index = src.index
    valid = np.isfinite(aod) & (aod > -0.1) & (aod < 5)
    return np.where(valid, aod, 0.0), valid, transform_index


def low_res(aod, valid):
    """Masked 4x4 block average. Returns lr and lr_mask (block has any real pixel)."""
    h, w = (aod.shape[0] // SCALE) * SCALE, (aod.shape[1] // SCALE) * SCALE
    a = aod[:h, :w].reshape(h // SCALE, SCALE, w // SCALE, SCALE)
    m = valid[:h, :w].reshape(h // SCALE, SCALE, w // SCALE, SCALE)
    count = m.sum(axis=(1, 3))
    lr = (a * m).sum(axis=(1, 3)) / np.maximum(count, 1)
    return lr, count > 0


def tile_around(lr, lr_mask, row, col):
    """
    A 16x16 low-res tile containing the monitor (the size the models trained on),
    kept inside the map. Returns the tile, its mask, and the monitor's position in
    the 64x64 high-res output.
    """
    n = PATCH // SCALE
    lr_row, lr_col = row // SCALE, col // SCALE
    top = min(max(lr_row - n // 2, 0), lr.shape[0] - n)
    left = min(max(lr_col - n // 2, 0), lr.shape[1] - n)
    tile = lr[top:top + n, left:left + n]
    tile_mask = lr_mask[top:top + n, left:left + n]
    return tile, tile_mask, row - top * SCALE, col - left * SCALE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tifs", default="data/raw data/raw_tifs_2023/dfw_aod_*.tif")
    parser.add_argument("--epa", default="data/epa/dfw_pm25_midday_2023-01-01_2023-12-31.csv")
    parser.add_argument("--out", default="data/epa/rq4_sr_matched.csv")
    parser.add_argument("--stats", default="data/split_data/train_stats.npz")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tifs = sorted(glob.glob(args.tifs))
    print(f"Found {len(tifs)} tifs")

    # Normalization stats from the year-split training set
    stats = np.load(args.stats)
    mean, std = float(stats["mean"]), float(stats["std"])
    print(f"Using train stats: mean={mean:.4f}, std={std:.4f}")

    # Year-split checkpoints
    models = {}
    for name in MODELS:
        path = f"checkpoints/{name}_years.pt"
        if not os.path.exists(path):
            print(f"skipping {name}: {path} not found")
            continue
        model = build_model(name, SCALE, use_consistency=False)
        model.load_state_dict(torch.load(path, map_location=device))
        models[name] = model.to(device).eval()
    print(f"Loaded models: {list(models)}")

    # EPA midday PM2.5, one value per site per day
    epa = pd.read_csv(args.epa)
    # Handle possible column name differences
    if "site_id" in epa.columns:
        epa["site"] = epa["site_id"].str.rsplit("-", n=1).str[0]
    elif "Site Num" in epa.columns:
        epa["site"] = epa["Site Num"].astype(str)
    epa = epa.groupby(["site", "Date Local"], as_index=False).agg(
        pm25=("pm25", "mean") if "pm25" in epa.columns else ("Sample Measurement", "mean"),
        Latitude=("Latitude", "first"),
        Longitude=("Longitude", "first")
    )
    # Normalize date column name
    if "Date Local" in epa.columns:
        epa = epa.rename(columns={"Date Local": "date"})

    # For each day, every AOD version at every monitor
    rows = []
    for path in tifs:
        date = os.path.basename(path)[-14:-4]   # dfw_aod_2023-01-01.tif -> 2023-01-01
        aod, valid, index = read_valid(path)
        lr, lr_mask = low_res(aod, valid)

        day_epa = epa[epa["date"] == date]
        for _, m in day_epa.iterrows():
            try:
                row, col = index(m["Longitude"], m["Latitude"])
            except Exception:
                continue
            if not (0 <= row < aod.shape[0] and 0 <= col < aod.shape[1]):
                continue
            if not valid[row, col]:
                continue  # cloudy at the monitor

            tile, tile_mask, r, c = tile_around(lr, lr_mask, row, col)

            # Model input: normalized like training, missing pixels = 0
            x = torch.tensor(
                np.where(tile_mask, (tile - mean) / std, 0.0),
                dtype=torch.float32
            )[None, None]
            x_mask = torch.tensor(tile_mask, dtype=torch.float32)[None, None]

            values = {
                "date": date,
                "site": m["site"],
                "pm25": m["pm25"],
                "aod_4km": lr[row // SCALE, col // SCALE],
                "aod_1km": aod[row, col],
                "bicubic": (metrics.bicubic_upscale(x, SCALE)[0, 0, r, c] * std + mean).item(),
            }

            with torch.no_grad():
                for name, model in models.items():
                    sharp, _ = model(x.to(device), x_mask.to(device))
                    values[name] = (sharp[0, 0, r, c] * std + mean).item()

            # 3x3 average of the real 1 km map (less noisy)
            window = np.where(valid, aod, np.nan)[
                max(row - 1, 0):row + 2,
                max(col - 1, 0):col + 2
            ]
            values["aod_1km_3x3"] = nanmean(window)
            rows.append(values)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"days used: {sorted(set(df['date']))}")
    print(f"{len(df)} cloud-free monitor-days ({df['site'].nunique()} sites)")

    # R² table
    cols = ["aod_4km", "bicubic"] + list(models) + ["aod_1km", "aod_1km_3x3"]
    wd = within_day(df, cols + ["pm25"])
    print(f"within-day: {len(wd)} monitor-days on {wd['date'].nunique()} days\n")
    print(f"{'':14s}{'all days':>10s}{'within-day':>12s}")
    for c in cols:
        w = f"{r2(wd[c], wd['pm25']):12.3f}" if len(wd) >= 3 else f"{'too few':>12s}"
        print(f"{c:14s}{r2(df[c], df['pm25']):10.3f}{w}")

    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()