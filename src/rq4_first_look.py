"""
RQ4 first look: does satellite AOD track ground PM2.5 at the EPA monitors?

For every monitor and every day, we read:
    aod_1km   the 1 km AOD pixel the monitor sits in        (the sharp map SR is aiming for)
    aod_4km   the 4x4-block average around that pixel        (the blurry map SR starts from)
    pm25      the EPA midday PM2.5 (10:00-14:00 local)       (from src/epa_pm25.py)
and check how well each AOD explains PM2.5 (R^2). If 1 km beats 4 km here, sharper
AOD is worth having, which is the reason for RQ4. The SR model's output gets added later.

Needs src/epa_pm25.py to have been run first (for data/epa/dfw_pm25_midday_*.csv).

How to run:
    python -m src.rq4_first_look
"""
import argparse
import glob
import os
import warnings

import matplotlib
matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

SCALE = 4  # 4x4 blocks of 1 km pixels = one 4 km pixel, same as the training patches

# Chart colours (one series per panel, so one colour)
BLUE = "#2a78d6"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"


def read_aod(path):
    """Read one daily AOD map. Returns the 1 km map, the 4 km map (same size), and the file handle."""
    src = rasterio.open(path)
    hr = src.read(1).astype(np.float64)  # already real AOD, NaN = missing (cloud)

    # 4 km map: average each 4x4 block of real pixels, then copy that average back to all
    # 16 pixels, so we can look it up at the same (row, col) as the 1 km map.
    # Blocks start at the top-left corner, like Rency's patches. Leftover edge pixels are dropped.
    h, w = (hr.shape[0] // SCALE) * SCALE, (hr.shape[1] // SCALE) * SCALE
    blocks = hr[:h, :w].reshape(h // SCALE, SCALE, w // SCALE, SCALE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # all-NaN (fully cloudy) blocks give a harmless warning
        lr = np.nanmean(blocks, axis=(1, 3))  # a 4 km pixel is real if any of its 16 pixels is
    lr_big = np.full_like(hr, np.nan)
    lr_big[:h, :w] = np.repeat(np.repeat(lr, SCALE, axis=0), SCALE, axis=1)
    return hr, lr_big, src


def r2(x, y):
    """R^2 of a straight-line fit = squared correlation."""
    return np.corrcoef(x, y)[0, 1] ** 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tifs", default="data/raw_tifs/dfw_aod_*.tif")
    parser.add_argument("--epa", default="data/epa/dfw_pm25_midday_2023-06-01_2023-06-30.csv")
    parser.add_argument("--out", default="data/epa/rq4_matched.csv")
    parser.add_argument("--figure", default="docs/diagrams/rq4_first_scatter.png")
    args = parser.parse_args()

    # 1. EPA midday PM2.5. A site can have several instruments (the 4th part of site_id);
    # average them so each site counts once per day.
    epa = pd.read_csv(args.epa)
    epa["site"] = epa["site_id"].str.rsplit("-", n=1).str[0]
    epa = epa.groupby(["site", "Date Local"], as_index=False).agg(
        pm25=("pm25", "mean"), Latitude=("Latitude", "first"), Longitude=("Longitude", "first"))

    # 2. For each day with an AOD map, look up the AOD at each monitor
    rows = []
    for path in sorted(glob.glob(args.tifs)):
        date = os.path.basename(path)[-14:-4]  # 'dfw_aod_2023-06-01.tif' -> '2023-06-01'
        hr, lr, src = read_aod(path)
        for _, m in epa[epa["Date Local"] == date].iterrows():
            row, col = src.index(m["Longitude"], m["Latitude"])  # lat/lon -> pixel position
            if not (0 <= row < hr.shape[0] and 0 <= col < hr.shape[1]):
                continue
            rows.append({"date": date, "site": m["site"], "pm25": m["pm25"],
                         "aod_1km": hr[row, col], "aod_4km": lr[row, col]})
        src.close()
    matched = pd.DataFrame(rows)

    # Keep monitor-days where both AODs exist (cloud-free at the monitor)
    both = matched.dropna(subset=["aod_1km", "aod_4km"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    both.to_csv(args.out, index=False)

    print(f"{len(matched)} monitor-days with PM2.5 and an AOD map, "
          f"{len(both)} cloud-free at the monitor "
          f"({both['site'].nunique()} sites, {both['date'].nunique()} days)")
    r2_4, r2_1 = r2(both["aod_4km"], both["pm25"]), r2(both["aod_1km"], both["pm25"])
    print(f"R^2 with PM2.5:  4 km AOD {r2_4:.3f}   1 km AOD {r2_1:.3f}")

    # 3. Two scatter plots side by side, same axes, so they can be compared directly
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True,
                             layout="constrained")
    for ax, col, label, r in [(axes[0], "aod_4km", "4 km AOD (blurry)", r2_4),
                              (axes[1], "aod_1km", "1 km AOD (sharp)", r2_1)]:
        x, y = both[col], both["pm25"]
        ax.scatter(x, y, s=28, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.8)
        slope, intercept = np.polyfit(x, y, 1)  # straight-line fit
        xs = np.linspace(x.min(), x.max(), 2)
        ax.plot(xs, slope * xs + intercept, color=INK_SOFT, linewidth=2)
        ax.set_title(f"{label}\nR² = {r:.2f}", color=INK)
        ax.set_xlabel(label, color=INK_SOFT)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_SOFT)
    axes[0].set_ylabel("EPA PM2.5, 10:00–14:00 (µg/m³)", color=INK_SOFT)
    fig.suptitle(f"RQ4 first look: AOD vs ground PM2.5, DFW June 2023 "
                 f"({len(both)} monitor-days, {both['site'].nunique()} sites)", color=INK)
    os.makedirs(os.path.dirname(args.figure), exist_ok=True)
    fig.savefig(args.figure, dpi=120, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"saved {args.out} and {args.figure}")


if __name__ == "__main__":
    main()
