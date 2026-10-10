"""
RQ4 first look: does satellite AOD track ground PM2.5 at the EPA monitors?

For every monitor and every day, we read:
    aod_4km      the 4x4-block average around the monitor's pixel   (the blurry map SR starts from)
    aod_1km      the 1 km AOD pixel the monitor sits in              (the sharp map SR is aiming for)
    aod_1km_3x3  average of the 3x3 pixels around the monitor        (1 km, but less noisy)
    pm25         the EPA midday PM2.5 (10:00-14:00 local)            (from src/epa_pm25.py)
and check how well each AOD explains PM2.5 (R^2), in two ways:

    All days     the plain comparison. Mostly measures whether AOD catches the
                 day-to-day haze level over the whole region, which 4 km already sees.
    Within-day   from each value we subtract that day's average over all monitors.
                 What's left is "this place vs the rest of DFW on the same day", which
                 is exactly what extra resolution should help with.

If 1 km beats 4 km (especially within-day), sharper AOD is worth having, which is the
reason for RQ4. The SR model's output gets added later.

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

SCALE = 4             # 4x4 blocks of 1 km pixels = one 4 km pixel, same as the training patches
MIN_SITES_PER_DAY = 3  # within-day comparison needs at least this many monitors on the same day

# The three AOD versions we compare: (column, label)
AODS = [("aod_4km", "4 km AOD (blurry)"),
        ("aod_1km", "1 km AOD (one pixel)"),
        ("aod_1km_3x3", "1 km AOD (3×3 average)")]

# Chart colours (one series per panel, so one colour)
BLUE = "#2a78d6"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"


def nanmean(x, **kwargs):
    """np.nanmean without the harmless 'Mean of empty slice' warning for all-cloud areas."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(x, **kwargs)


def read_aod(path):
    """Read one daily AOD map. Returns the 1 km map, the 4 km map (same size), and the file handle."""
    src = rasterio.open(path)
    hr = src.read(1).astype(np.float64)  # already real AOD, NaN = missing (cloud)

    # 4 km map: average each 4x4 block of real pixels, then copy that average back to all
    # 16 pixels, so we can look it up at the same (row, col) as the 1 km map.
    # Blocks start at the top-left corner, like Rency's patches. Leftover edge pixels are dropped.
    h, w = (hr.shape[0] // SCALE) * SCALE, (hr.shape[1] // SCALE) * SCALE
    blocks = hr[:h, :w].reshape(h // SCALE, SCALE, w // SCALE, SCALE)
    lr = nanmean(blocks, axis=(1, 3))  # a 4 km pixel is real if any of its 16 pixels is
    lr_big = np.full_like(hr, np.nan)
    lr_big[:h, :w] = np.repeat(np.repeat(lr, SCALE, axis=0), SCALE, axis=1)
    return hr, lr_big, src


def r2(x, y):
    """R^2 of a straight-line fit = squared correlation."""
    return np.corrcoef(x, y)[0, 1] ** 2


def within_day(df, cols):
    """
    Subtract each day's average (over all monitors that day) from every value.
    Only keeps days with at least MIN_SITES_PER_DAY monitors, otherwise the
    'day average' is too shaky to mean anything.
    """
    df = df.groupby("date").filter(lambda g: len(g) >= MIN_SITES_PER_DAY).copy()
    for c in cols:
        df[c] = df[c] - df.groupby("date")[c].transform("mean")
    return df


def style(ax):
    """Light grid, no top/right box lines."""
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_SOFT)


def scatter(ax, x, y, title):
    """One scatter panel with a straight-line fit."""
    ax.scatter(x, y, s=28, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.8)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 2)
    ax.plot(xs, slope * xs + intercept, color=INK_SOFT, linewidth=2)
    ax.set_title(title, color=INK, fontsize=10.5)
    style(ax)


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
            # 3x3 window around the monitor (smaller at the map edge); clouds are skipped
            window = hr[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2]
            rows.append({"date": date, "site": m["site"], "pm25": m["pm25"],
                         "aod_4km": lr[row, col], "aod_1km": hr[row, col],
                         "aod_1km_3x3": nanmean(window)})
        src.close()
    matched = pd.DataFrame(rows)

    # Keep monitor-days where ALL three AODs exist, so every version is scored on the same points
    cols = [c for c, _ in AODS]
    both = matched.dropna(subset=cols)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    both.to_csv(args.out, index=False)
    print(f"{len(matched)} monitor-days with PM2.5 and an AOD map, "
          f"{len(both)} cloud-free at the monitor "
          f"({both['site'].nunique()} sites, {both['date'].nunique()} days)")

    # 3. Within-day version: subtract each day's average from AOD and PM2.5
    wd = within_day(both, cols + ["pm25"])
    print(f"within-day: {len(wd)} monitor-days on {wd['date'].nunique()} days "
          f"with at least {MIN_SITES_PER_DAY} monitors")

    # 4. R^2 table
    print(f"\n{'':26s}{'all days':>10s}{'within-day':>12s}")
    scores = {}
    for c, label in AODS:
        scores[c] = (r2(both[c], both["pm25"]), r2(wd[c], wd["pm25"]))
        print(f"{label:26s}{scores[c][0]:10.3f}{scores[c][1]:12.3f}")

    # 5. Figure: top row = all days, bottom row = within-day. Same axes along each row.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), sharey="row", layout="constrained")
    for j, (c, label) in enumerate(AODS):
        scatter(axes[0, j], both[c], both["pm25"], f"{label}\nR² = {scores[c][0]:.2f}")
        scatter(axes[1, j], wd[c], wd["pm25"], f"{label}, within-day\nR² = {scores[c][1]:.2f}")
        axes[0, j].set_xlabel("AOD", color=INK_SOFT)
        axes[1, j].set_xlabel("AOD minus that day's DFW average", color=INK_SOFT)
    # Same x range across each row, so the panels compare fairly
    for i in range(2):
        lo = min(ax.get_xlim()[0] for ax in axes[i])
        hi = max(ax.get_xlim()[1] for ax in axes[i])
        for ax in axes[i]:
            ax.set_xlim(lo, hi)
    axes[0, 0].set_ylabel("EPA PM2.5, 10:00–14:00 (µg/m³)", color=INK_SOFT)
    axes[1, 0].set_ylabel("PM2.5 minus that day's DFW average (µg/m³)", color=INK_SOFT)
    fig.suptitle(f"RQ4 first look: AOD vs ground PM2.5, DFW June 2023\n"
                 f"top: all {len(both)} monitor-days ({both['site'].nunique()} sites)   ·   "
                 f"bottom: within-day, {len(wd)} monitor-days on {wd['date'].nunique()} days",
                 color=INK)
    os.makedirs(os.path.dirname(args.figure), exist_ok=True)
    fig.savefig(args.figure, dpi=120, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"\nsaved {args.out} and {args.figure}")


if __name__ == "__main__":
    main()
