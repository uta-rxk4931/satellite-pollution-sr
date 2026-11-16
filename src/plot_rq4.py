"""Make RQ4 figures from the matched CSV."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/epa/rq4_sr_matched.csv")
out_dir = Path("data/epa/figures")
out_dir.mkdir(exist_ok=True)

# Figure 1: All-days scatter (4 km vs best SR vs true 1 km)
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
versions = [
    ("aod_4km", "4 km coarse"),
    ("edsr", "EDSR (best SR)"),
    ("aod_1km", "True 1 km"),
]
for ax, (col, title) in zip(axes, versions):
    ax.scatter(df[col], df["pm25"], s=8, alpha=0.4)
    ax.set_xlabel(f"AOD ({title})")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("EPA PM2.5 (µg/m³)")
fig.suptitle("RQ4: AOD vs midday EPA PM2.5 (all days, 2023)")
fig.tight_layout()
fig.savefig(out_dir / "rq4_scatter_all_days.png", dpi=150)
print("Saved", out_dir / "rq4_scatter_all_days.png")

# Figure 2: Within-day anomalies
df["date"] = pd.to_datetime(df["date"])
day_mean = df.groupby("date")[["pm25", "aod_4km", "edsr", "aod_1km"]].transform("mean")
anom = df[["pm25", "aod_4km", "edsr", "aod_1km"]] - day_mean

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, (col, title) in zip(axes, versions):
    ax.scatter(anom[col], anom["pm25"], s=8, alpha=0.4)
    ax.set_xlabel(f"AOD anomaly ({title})")
    ax.set_title(title)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("PM2.5 anomaly")
fig.suptitle("RQ4: Within-day anomalies (2023)")
fig.tight_layout()
fig.savefig(out_dir / "rq4_scatter_within_day.png", dpi=150)
print("Saved", out_dir / "rq4_scatter_within_day.png")

plt.show()