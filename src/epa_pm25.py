"""
Sprint 4 (Track A): (Swap Ryan) ground-truth PM2.5 from EPA monitors in the DFW box.

EPA measures PM2.5 (tiny particles, the pollution that harms lungs) at fixed ground
monitors. We will compare it with the satellite AOD later (RQ4: does sharpened AOD
match ground PM2.5 better than blurry AOD?). EPA data is for EVALUATION ONLY, never training.

Where the data comes from:
    EPA's prebuilt yearly files at https://aqs.epa.gov/aqsweb/airdata/download_files.html
    Parameter 88101 = PM2.5, measured with the official (FRM/FEM) methods.
    No login or API key needed.

What it saves (in data/epa/, which is never committed):
    dfw_pm25_hourly_<start>_<end>.csv   every hourly reading inside the box
    dfw_pm25_daily_<start>_<end>.csv    EPA's daily (24-hour) averages inside the box
    dfw_pm25_midday_<start>_<end>.csv   per monitor per day: average of 10:00-14:00 local,
                                        which covers both satellite passes
                                        (Terra ~10:30, Aqua ~13:30)
    dfw_pm25_monitors.csv               one row per monitor: id, name, lat/lon

How to run:
    python -m src.epa_pm25                                  # June 2023 (matches sprint2 patches)
    python -m src.epa_pm25 --start 2018-01-01 --end 2023-12-31   # all years for the year split
"""
import argparse
import os
import urllib.request

import pandas as pd

# Same box as the AOD data (src/sprint1_dfw_aod.js): west, south, east, north
DFW_BOX = (-97.8, 32.3, -96.3, 33.5)

URL = "https://aqs.epa.gov/aqsweb/airdata/{kind}_88101_{year}.zip"
OUT_DIR = "data/epa"
RAW_DIR = "data/epa/raw"

# Satellite passes over DFW are around 10:30 (Terra) and 13:30 (Aqua) local time
MIDDAY_HOURS = range(10, 15)  # 10:00, 11:00, 12:00, 13:00, 14:00


def download(kind, year):
    
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{kind}_88101_{year}.zip")
    if not os.path.exists(path):
        print(f"downloading {kind} {year} ...")
        urllib.request.urlretrieve(URL.format(kind=kind, year=year), path)
    return path


def in_box(df):
    # Keep only rows whose monitor is inside the DFW box
    west, south, east, north = DFW_BOX
    return df[df["Longitude"].between(west, east) & df["Latitude"].between(south, north)]


def site_id(df):
    """
    One id per monitor, e.g. '48-113-0069-1' (state-county-site-POC).
    POC = 'parameter occurrence code': a site can have more than one PM2.5 instrument.
    """
    return (df["State Code"].astype(str).str.zfill(2) + "-" +
            df["County Code"].astype(str).str.zfill(3) + "-" +
            df["Site Num"].astype(str).str.zfill(4) + "-" +
            df["POC"].astype(str))


def load(kind, years, start, end, date_col):
    """Read the yearly files, keep the DFW box and the date range."""
    parts = []
    for year in years:
        # pandas can read a CSV straight out of a .zip file
        df = pd.read_csv(download(kind, year), low_memory=False)
        df = in_box(df)
        df = df[df[date_col].between(start, end)]  # dates are 'YYYY-MM-DD' strings, so this works
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "site_id", site_id(df))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-06-01")
    parser.add_argument("--end", default="2023-06-30")
    args = parser.parse_args()

    years = range(int(args.start[:4]), int(args.end[:4]) + 1)
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{args.start}_{args.end}"

    # 1. Hourly readings ("Sample Measurement" is PM2.5 in micrograms per cubic metre)
    hourly = load("hourly", years, args.start, args.end, "Date Local")
    hourly = hourly[["site_id", "Latitude", "Longitude", "Date Local", "Time Local",
                     "Sample Measurement", "Units of Measure", "County Name"]]
    hourly = hourly.rename(columns={"Sample Measurement": "pm25"})
    hourly.to_csv(f"{OUT_DIR}/dfw_pm25_hourly_{tag}.csv", index=False)

    # 2. Midday average: the hours around the satellite passes, per monitor per day.
    # "Time Local" looks like '13:00', so the first two characters are the hour.
    hour = hourly["Time Local"].str[:2].astype(int)
    midday = (hourly[hour.isin(MIDDAY_HOURS)]
              .groupby(["site_id", "Date Local"], as_index=False)
              .agg(pm25=("pm25", "mean"), n_hours=("pm25", "size"),
                   Latitude=("Latitude", "first"), Longitude=("Longitude", "first")))
    midday = midday[midday["n_hours"] >= 3]  # need at least 3 of the 5 hours to trust the average
    midday.to_csv(f"{OUT_DIR}/dfw_pm25_midday_{tag}.csv", index=False)

    # 3. EPA's own daily averages. Some monitors only report daily (one filter per day),
    # so this file can have monitors the hourly file doesn't.
    daily = load("daily", years, args.start, args.end, "Date Local")
    daily = daily[daily["Sample Duration"].str.contains("24")]  # 24-hour averages only
    daily = daily[["site_id", "Latitude", "Longitude", "Date Local", "Arithmetic Mean",
                   "Sample Duration", "Local Site Name", "County Name"]]
    daily = daily.rename(columns={"Arithmetic Mean": "pm25"})
    daily.to_csv(f"{OUT_DIR}/dfw_pm25_daily_{tag}.csv", index=False)

    # 4. Monitor list (from both files)
    names = daily.groupby("site_id")["Local Site Name"].first()
    monitors = (pd.concat([hourly, daily])
                .groupby("site_id", as_index=False)
                .agg(Latitude=("Latitude", "first"), Longitude=("Longitude", "first"),
                     county=("County Name", "first")))
    monitors["name"] = monitors["site_id"].map(names)
    monitors["hourly"] = monitors["site_id"].isin(hourly["site_id"])
    monitors["daily"] = monitors["site_id"].isin(daily["site_id"])
    monitors.to_csv(f"{OUT_DIR}/dfw_pm25_monitors.csv", index=False)

    # Quick summary
    print(f"\n{len(monitors)} PM2.5 monitors in the DFW box "
          f"({monitors['hourly'].sum()} hourly, {monitors['daily'].sum()} daily)")
    print(f"hourly readings: {len(hourly):,}   midday averages: {len(midday):,}   "
          f"daily averages: {len(daily):,}")
    print(f"midday PM2.5: median {midday['pm25'].median():.1f}, "
          f"min {midday['pm25'].min():.1f}, max {midday['pm25'].max():.1f} ug/m3")
    print(f"saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
