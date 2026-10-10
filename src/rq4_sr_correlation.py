"""
RQ4 (Sprint 5, Track A): does SHARPENED AOD track ground PM2.5 better than blurry AOD?

Same idea as src/rq4_first_look.py, but with the SR models' output added:
    aod_4km      blurry 4 km AOD at the monitor          (what SR starts from)
    bicubic      4 km map enlarged with bicubic           (the simple baseline)
    edsr / unet / swinir   4 km map sharpened by each trained model
    aod_1km      the real 1 km AOD                        (what SR is aiming for)
Each one is compared with EPA midday PM2.5 (R^2), for all days and within-day
(see rq4_first_look.py for what "within-day" means).

IMPORTANT, fair days only: the models trained on the 1 km maps of early June. On those
days a model has already seen the right answer, so its output would look too good.
We rebuild Rency's patch list (same rules as src/sprint2_patches.ipynb) to find out which
day each training patch came from, and only score days with NO training patches.

This is a CODE CHECK, not a result: one month leaves very few fair monitor-days.
The real RQ4 numbers need the 2018-2023 data and the year split (test on 2023).

Needs: src/epa_pm25.py run first, and trained models in checkpoints/
(python -m src.train --data real --model <name> --epochs 300).

How to run:
    python -m src.rq4_sr_correlation
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import rasterio
import torch

from src import metrics
from src.dataset import make_real_datasets
from src.models.sr_net import build_model
from src.rq4_first_look import nanmean, r2, within_day

SCALE = 4
PATCH = 64          # high-res patch size the models were trained on (low-res = 16)
MIN_VALID = 0.6     # Rency's rule: keep a patch if at least 60% of its pixels are real
MODELS = ["edsr", "unet", "swinir"]


def read_valid(path):
    """One daily map, cleaned the same way as the patch notebook. Returns aod (0 = missing) and valid."""
    with rasterio.open(path) as src:
        aod = src.read(1).astype(np.float32)
        transform_index = src.index
    valid = np.isfinite(aod) & (aod > -0.1) & (aod < 5)
    return np.where(valid, aod, 0.0), valid, transform_index


def training_days(tifs, n_train):
    """
    Rebuild the patch list in the same order as the notebook and return the dates
    of the first n_train patches (the training set).
    """
    dates = []
    for path in tifs:
        _, valid, _ = read_valid(path)
        h, w = valid.shape
        for r in range(0, h - PATCH + 1, PATCH // 2):
            for c in range(0, w - PATCH + 1, PATCH // 2):
                if valid[r:r + PATCH, c:c + PATCH].mean() >= MIN_VALID:
                    dates.append(os.path.basename(path)[-14:-4])
    return set(dates[:n_train]), len(dates)


def low_res(aod, valid):
    """Masked 4x4 block average, exactly like the notebook. Returns lr and lr_mask (block has any real pixel)."""
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
    parser.add_argument("--tifs", default="data/raw_tifs/dfw_aod_*.tif")
    parser.add_argument("--epa", default="data/epa/dfw_pm25_midday_2023-06-01_2023-06-30.csv")
    parser.add_argument("--out", default="data/epa/rq4_sr_matched.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tifs = sorted(glob.glob(args.tifs))

    # 1. Which days did the models train on? Normalization must also match training.
    train_set, _, _ = make_real_datasets()
    seen, n_patches = training_days(tifs, len(train_set))
    print(f"rebuilt {n_patches} patches; training used {len(train_set)} of them, "
          f"from {len(seen)} days ({min(seen)} to {max(seen)})")
    mean, std = train_set.mean, train_set.std

    # 2. The trained models
    models = {}
    for name in MODELS:
        path = f"checkpoints/{name}_real.pt"
        if not os.path.exists(path):
            print(f"skipping {name}: {path} not found")
            continue
        model = build_model(name, SCALE, use_consistency=False)
        model.load_state_dict(torch.load(path, map_location=device))
        models[name] = model.to(device).eval()

    # 3. EPA midday PM2.5, one value per site per day (instruments at the same site averaged)
    epa = pd.read_csv(args.epa)
    epa["site"] = epa["site_id"].str.rsplit("-", n=1).str[0]
    epa = epa.groupby(["site", "Date Local"], as_index=False).agg(
        pm25=("pm25", "mean"), Latitude=("Latitude", "first"), Longitude=("Longitude", "first"))

    # 4. For each FAIR day, every AOD version at every monitor
    rows = []
    for path in tifs:
        date = os.path.basename(path)[-14:-4]
        if date in seen:
            continue
        aod, valid, index = read_valid(path)
        lr, lr_mask = low_res(aod, valid)
        for _, m in epa[epa["Date Local"] == date].iterrows():
            row, col = index(m["Longitude"], m["Latitude"])
            if not (0 <= row < lr.shape[0] * SCALE and 0 <= col < lr.shape[1] * SCALE):
                continue
            if not valid[row, col]:
                continue  # cloudy at the monitor: no true 1 km value to compare against
            tile, tile_mask, r, c = tile_around(lr, lr_mask, row, col)

            # Model input: normalized like training, missing pixels = 0 (as in AODDataset)
            x = torch.tensor(np.where(tile_mask, (tile - mean) / std, 0.0), dtype=torch.float32)[None, None]
            x_mask = torch.tensor(tile_mask, dtype=torch.float32)[None, None]
            values = {"date": date, "site": m["site"], "pm25": m["pm25"],
                      "aod_4km": lr[row // SCALE, col // SCALE],
                      "aod_1km": aod[row, col],
                      "bicubic": (metrics.bicubic_upscale(x, SCALE)[0, 0, r, c] * std + mean).item()}
            with torch.no_grad():
                for name, model in models.items():
                    sharp, _ = model(x.to(device), x_mask.to(device))
                    values[name] = (sharp[0, 0, r, c] * std + mean).item()
            # Also the 3x3 average of the real 1 km map (less noisy), clouds skipped
            window = np.where(valid, aod, np.nan)[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2]
            values["aod_1km_3x3"] = nanmean(window)
            rows.append(values)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"fair days (no training patches): {sorted(set(df['date']))}")
    print(f"{len(df)} cloud-free monitor-days on fair days ({df['site'].nunique()} sites)")

    # 5. R^2 table
    cols = ["aod_4km", "bicubic"] + list(models) + ["aod_1km", "aod_1km_3x3"]
    wd = within_day(df, cols + ["pm25"])
    print(f"within-day: {len(wd)} monitor-days on {wd['date'].nunique()} days\n")
    print(f"{'':14s}{'all days':>10s}{'within-day':>12s}")
    for c in cols:
        w = f"{r2(wd[c], wd['pm25']):12.3f}" if len(wd) >= 3 else f"{'too few':>12s}"
        print(f"{c:14s}{r2(df[c], df['pm25']):10.3f}{w}")
    print(f"\nsaved {args.out}")
    print("Reminder: this many points is a code check, not a result.")


if __name__ == "__main__":
    main()
