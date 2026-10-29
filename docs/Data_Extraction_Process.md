# Data Extraction Process — DFW MAIAC AOD (2018–2023)

## Overview
This document describes how we obtained daily 1 km MAIAC AOD (MCD19A2.061) for the Dallas–Fort Worth region, clipped it, and converted it into patches for the super-resolution project.

**Final products:**
- Per-year patch files: `patches_YYYY.npz`
- Ready for year-based split: 2018–2021 train / 2022 val / 2023 test

## 1. Download the raw data
1. Go to NASA Earthdata Search: https://search.earthdata.nasa.gov/
2. Search for **MCD19A2** and select  
   **MODIS/Terra+Aqua Land Aerosol Optical Depth Daily L2G Global 1km SIN Grid V061**
3. Apply filters:
   - Temporal: one year at a time (e.g. 2018-01-01 to 2018-12-31)
   - Spatial: upload `dfw_box.geojson` or draw the box  
     West -97.8, South 32.3, East -96.3, North 33.5
4. Download All → choose **Download Script** or **Earthdata Download** tool.
5. Save files into a folder such as `maiac_2018/`, `maiac_2019/`, etc.
6. Expect ~365 granules (~3 GB) per year. 

## 2. Clip to DFW box
Use `Data_extraction_dfw_clip.py` (pyhdf version).

```bash
pip install pyhdf numpy
python Data_extraction_dfw_clip.py
```

- Input: folder of `.hdf` files
- Output: folder of small `_dfw.npy` files (200×200)

## 3. Create patches
Use `Data_extraction_make_patches.py`.

```bash
python Data_extraction_make_patches.py
```

- Input: folder of `_dfw.npy` files
- Output: `patches_YYYY.npz` containing:
  - `lr` (low-resolution, scale=4, box average)
  - `hr` (high-resolution target)
  - `mask` (valid pixels)
  - `year`
  - `scale`

## 4. Year split and train-only statistics
Use `combine_and_split.py`.

```bash
python combine_and_split.py
```

This script:
- Loads all `patches_YYYY.npz` files
- Creates the year split:
  - Train: 2018–2021
  - Val: 2022
  - Test: 2023
- Computes mean and standard deviation **only on the training patches**
- Saves:
  - `train.npz`
  - `val.npz`
  - `test.npz`
  - `train_stats.npz`

## Notes
- Raw HDF tiles are **not** uploaded to GitHub (too large).
- Only the final `.npz` patch files (and train statistics) go into the repository.


## Files
- `Data_extraction_dfw_clip.py` — clips HDF tiles to DFW
- `Data_extraction_make_patches.py` — builds the per-year patch `.npz` files
- `combine_and_split.py` — creates the final year split + train-only stats
- `dfw_box.geojson` — area of interest used for Earthdata filtering
