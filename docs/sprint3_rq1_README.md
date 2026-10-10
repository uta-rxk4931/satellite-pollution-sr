# Sprint 3: first real training run (RQ1)

## What this is in one sentence
We train the SR models on many real DFW patches, then test them on **different** patches they have never seen, and check whether they beat plain bicubic. That is the first answer to **RQ1: can deep SR sharpen AOD maps better than simple interpolation?**

These are **preliminary** numbers. They use a temporary split of the June 2023 data. The final numbers will use Rency's split by year (2018–21 train / 2022 validation / 2023 test).

---

## Setup (once per computer)

### 1. Install the packages
```
python -m pip install numpy matplotlib scikit-image timm
```

### 2. Install the GPU version of PyTorch (if you have an NVIDIA card)
```
python -m pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

> **Watch out:** installing `timm` also installs `torchvision`, which can quietly replace the GPU PyTorch with a CPU-only one. Run step 2 **after** step 1. Without a GPU, skip step 2. Everything still works, just slower.

### 3. Check it worked
```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
`True` means the GPU will be used. Every script also prints `device: cuda` (GPU) or `device: cpu` at the start.

---

## All the commands

Open a terminal in the project folder. Always use `python -m src.<name>`, not `python src/<name>.py`, or Python can't find the other project files. `-W ignore` hides harmless library warnings.

### Step 1: Sanity check, learn one patch (Sprint 2 test)
```
python -W ignore -m src.overfit_one_patch                  # EDSR (default)
python -W ignore -m src.overfit_one_patch --model unet --figure docs/diagrams/sprint2_overfit_unet.png
python -W ignore -m src.overfit_one_patch --model swinir --figure docs/diagrams/sprint2_overfit_swinir.png
```
Each run should end with `PASS`. Give each model its own `--figure` name, or it overwrites the EDSR picture. See `docs/sprint2_overfit_README.md` for details.

### Step 2: Train on real patches and compare with bicubic (Sprint 3, RQ1)
```
python -W ignore -m src.train --data real --model edsr   --epochs 300 --eval_every 25
python -W ignore -m src.train --data real --model unet   --epochs 300 --eval_every 25
python -W ignore -m src.train --data real --model swinir --epochs 300 --eval_every 25
```

To also save the output to a text file (PowerShell):
```
python -W ignore -m src.train --data real --model edsr --epochs 300 --eval_every 25 *> train_edsr.log
```

### Options for `src.train` (all optional)
| Option | What it does | Default |
|---|---|---|
| `--data real` | Use Rency's real patches instead of fake maps | `dummy` (fake maps) |
| `--data_path file.npz` | Which real-data file to load | `data/sprint2_patches.npz` |
| `--model unet` | `edsr`, `unet` or `swinir` | `edsr` |
| `--epochs 300` | How many times to go through all the training patches | `10` |
| `--eval_every 25` | Score the validation set every N epochs (it always scores the last one) | `1` |
| `--batch_size 32` | How many patches per training step | `32` |
| `--lr 2e-4` | Learning rate (how big each learning step is) | `2e-4` |
| `--consistency` | Turn on the average-back rule (that's Sprint 4 / RQ2, so leave it off for now) | off |

The trained model is saved to `checkpoints/<model>_real.pt` (e.g. `checkpoints/swinir_real.pt`). The `checkpoints/` folder is in `.gitignore`, so it's never committed.

### How long it takes (RTX 3080)
| Command | Time |
|---|---|
| One-patch test, EDSR or U-Net | ~10–15 s |
| One-patch test, SwinIR | ~2 min |
| Real training, 300 epochs, EDSR or U-Net | ~20 s |
| Real training, 300 epochs, SwinIR | ~1 min |

On a CPU, expect several times longer.

---

## The data split (temporary)
`data/sprint2_patches.npz` has 144 patches from 29 days in June 2023. `make_real_datasets()` in `src/dataset.py` splits them **by date**:

| Set | Patches | What it's for |
|---|---|---|
| Train | 112 (early June) | The model learns from these |
| Validation | 32 (last days of June) | Scored only, never learned from |

Why not a random split? Rency cuts the patches so neighbours from the same day **overlap** by half a patch. A random split would put almost the same pixels in both sets, and the scores would look better than they really are. The code moves the cut earlier until no validation patch overlaps a training patch.

The normalization (mean and std) is computed from the **training patches only**, so no information leaks in from the validation set.

---

## How to read the output
Every `--eval_every` epochs, `src.train` prints two lines. In each pair, the first number is the **model** and the number in brackets is **bicubic**.

```
epoch 300 | loss -1.365 | RMSE 0.0140 (bicubic 0.0147) | PSNR 37.10 (bicubic 36.68) | SSIM 0.905 (bicubic 0.891)
          | consistency 1.0e-03 | detail 0.54 (bicubic 0.32) | coverage90 0.83 | cal_err 0.041 | unc 0.0112
```

| Number | Meaning | Good is | Which question |
|---|---|---|---|
| **loss** | What the model is minimizing. It can be **negative**; that's normal for this loss (see below). | going down | – |
| **RMSE** | Average error, in AOD units | lower | RQ1 |
| **PSNR** | Image quality in dB | higher | RQ1 |
| **SSIM** | How similar the shapes look (0–1) | higher | RQ1 |
| **consistency** | How far the sharp map is from averaging back to the blurry input | lower | RQ2 |
| **detail** | Sharpness compared with the real map. 1.0 is right, below 1 is too blurry, above 1 is made-up detail | close to 1.0 | RQ2 |
| **coverage90** | How often the true value falls inside the model's "90% sure" range | close to 0.90 | RQ3 |
| **cal_err** | How far off the confidence is, averaged over 10 levels | lower (0 = perfect) | RQ3 |
| **unc** | Average uncertainty the model reports (AOD) | small, given good coverage | RQ3 |

**Why the loss is negative:** `src.train` uses `uncertainty_loss`, which trains the sharp map **and** the confidence map together. That loss includes a `log(variance)` term, which goes below zero when the model is confident. Only the trend matters: down is good.

---

## Results

### 1. One-patch test (patch 40, 2000 steps, learning rate 2e-4)
| Model | Parameters | Final RMSE | vs bicubic (0.0235) | PSNR gain | Result |
|---|---|---|---|---|---|
| EDSR | 1.52M | 0.00017 | 0.7% of bicubic's error | +42.7 dB | PASS |
| U-Net | 1.93M | 0.00033 | 1.4% | +37.2 dB | PASS |
| SwinIR-light | 0.92M | 0.00040 | 1.7% | +35.3 dB | PASS |

Figures: `docs/diagrams/sprint2_overfit.png` (EDSR), `sprint2_overfit_unet.png`, `sprint2_overfit_swinir.png`.

All three can learn real AOD through the shared model wrapper. U-Net's error goes up and down during training, so its final number changes the most between runs (0.6%–1.4% of bicubic's error); it passes every time. SwinIR learned more slowly but steadily, and was still improving at step 2000.

### 2. Real training (112 train / 32 validation patches, 300 epochs)
Scores on the **validation** patches, which the model never trained on:

| | RMSE ↓ | PSNR ↑ | SSIM ↑ | Detail (1.0 = right) |
|---|---|---|---|---|
| Bicubic | 0.0147 | 36.68 | 0.891 | 0.32 |
| EDSR | 0.0142 | 36.92 | 0.902 | 0.53 |
| U-Net | 0.0142 | 36.93 | 0.900 | 0.40 |
| **SwinIR-light** | **0.0140** | **37.10** | **0.905** | 0.54 |

### What the results mean
- **All three models beat bicubic on every score**, on patches they never saw: about 4–5% lower RMSE, +0.25 to +0.42 dB PSNR, +0.01 SSIM. The gain is small, but it holds across the whole run, not at one lucky epoch.
- **The models add real detail.** Detail goes from 0.32 (bicubic) to about 0.5, and the error drops at the same time, so the new detail is real, not made up. They're still too smooth: the real map has about twice as much fine detail.
- **SwinIR is slightly ahead and the most stable**, with the fewest parameters. But 32 validation patches from a few days are too few to call a winner: 0.0140 vs 0.0142 is within noise.
- **No severe overfitting.** The scores level off around epoch 150–200 and don't get worse after.

### Why the gain is small
112 patches from one month is very little data. The one-patch test proves the models **can** learn the fine detail. To do it on **new** patches, they need many more examples. Rency's 2018–21 training set will give them that.

### Suggested sentence for the midterm
> On a preliminary June 2023 split, all three SR models beat bicubic (best: RMSE 0.0140 vs 0.0147, +0.4 dB PSNR, SSIM 0.905 vs 0.891). Final numbers will use the 2018–21 / 2022 / 2023 year split.

---

## Things to know
- **Confidence-map numbers are a bonus.** RQ3 isn't due until Sprint 5. They came for free because `src.train` uses `uncertainty_loss`. Coverage is about 0.83–0.89 against the 0.90 target, so the model is slightly overconfident.
- **Results can vary a little between runs** on a GPU, even with the same seed (e.g. the last digit of RMSE).
- **The average-back rule is off** in every run here. It's Sprint 4 / RQ2.

## Still to do in Sprint 3
- [ ] Rency: split by year (2018–21 / 2022 / 2023) and the training-only normalization or log transform
- [ ] Rerun Step 2 on the year split, for the real RQ1 numbers
- [ ] Decide with Rency: train RQ1 with plain masked MSE (like the one-patch test) or keep `uncertainty_loss`
- [ ] Pick a main model, or keep all three for the RQ1 table
- [ ] Midterm text (written together): four RQs, the average-back plan, out of scope (no PDE, no PM2.5 retrieval)

---
---

# ⏩ Sprint 4 (started early): EPA PM2.5 and the first RQ4 look

> **This is Sprint 4 work, done ahead of schedule.** After the midterm, Ryan leads **Track A**, which includes pulling the EPA PM2.5 data and preparing the RQ4 comparison. It doesn't depend on Rency's year split, so it could start now.

## What this is in one sentence
We download PM2.5 measured at EPA ground monitors in DFW and check whether satellite AOD tracks it, and whether the **sharp 1 km** AOD tracks it better than the **blurry 4 km** AOD. That's **RQ4: does sharpened AOD correlate more strongly with ground PM2.5 than the blurry version?**

EPA data is for **evaluation only**, never training.

## Extra setup
One more package, to read the georeferenced AOD maps in `data/raw_tifs/`:
```
python -m pip install rasterio pandas
```

## Commands
```
python -m src.epa_pm25            # 1. download EPA PM2.5 for the DFW box, June 2023
python -m src.rq4_first_look      # 2. match AOD to each monitor, compute R², save the figure
```
For all years, once the 2018–2022 AOD maps exist:
```
python -m src.epa_pm25 --start 2018-01-01 --end 2023-12-31
```
Step 1 downloads about 87 MB from EPA the first time (cached in `data/epa/raw/` after that). Both steps take under a minute.

## The files involved
```
src/epa_pm25.py                        <- downloads and filters the EPA data
src/rq4_first_look.py                  <- matches AOD to monitors, R², figure
data/raw_tifs/dfw_aod_2023-06-*.tif    <- full 1 km AOD maps, one per day (have lat/lon)
data/epa/                              <- EPA outputs (never committed)
docs/diagrams/rq4_first_scatter.png    <- the figure
```

### What `epa_pm25.py` saves (in `data/epa/`)
| File | What's in it |
|---|---|
| `dfw_pm25_hourly_<dates>.csv` | Every hourly PM2.5 reading inside the box |
| `dfw_pm25_daily_<dates>.csv` | EPA's 24-hour averages |
| `dfw_pm25_midday_<dates>.csv` | Per monitor per day: average of **10:00–14:00 local**, which covers both satellite passes (Terra ~10:30, Aqua ~13:30). Needs at least 3 of the 5 hours. |
| `dfw_pm25_monitors.csv` | One row per instrument: id, name, lat/lon, county |

- **Source:** EPA's prebuilt files, parameter **88101** (PM2.5, official methods). No login or API key.
- **Area:** the same box as the AOD data: lon −97.8 to −96.3, lat 32.3 to 33.5.

### The monitors (June 2023)
10 instruments at **7 sites**: Dallas Convention Center, Dallas Hinton (3 instruments), Denton Airport South, Kaufman, Fort Worth Northwest, Haws Athletic Center, Fort Worth California Parkway North (2 instruments). Sites with more than one instrument are averaged, so each site counts once per day.

## How the matching works (`rq4_first_look.py`)
For each monitor on each day, it reads three AOD values from that day's map:

| Column | What it is |
|---|---|
| `aod_4km` | Average of the 4×4 block the monitor sits in: the **blurry** map SR starts from (same block averaging as the training patches) |
| `aod_1km` | The single 1 km pixel the monitor sits in: the **sharp** map SR aims for |
| `aod_1km_3x3` | Average of the 3×3 pixels around the monitor: 1 km, but less noisy |

Only monitor-days where all three exist (not cloudy at the monitor) are kept, so every version is scored on the same points. The matched table is saved to `data/epa/rq4_matched.csv`.

Then it scores each version in two ways:
- **All days:** plain R² between AOD and PM2.5. This mostly measures whether AOD catches the **day-to-day haze level** over all of DFW.
- **Within-day:** from every value, subtract that day's average over all monitors, then compute R². What's left is **"this place vs the rest of DFW on the same day"**, the thing extra resolution should help with. Only days with at least 3 monitors are used.

**R²** = how much of the up-and-down in PM2.5 the AOD explains, from 0 (nothing) to 1 (everything).

## Results (June 2023)
88 monitor-days were cloud-free at the monitor (7 sites, 20 days); 81 of them fall on the 14 days with at least 3 monitors.

| R² with EPA PM2.5 | All days | Within-day |
|---|---|---|
| 4 km AOD (blurry) | 0.31 | 0.04 |
| 1 km AOD (one pixel) | 0.30 | 0.01 |
| 1 km AOD (3×3 average) | 0.31 | 0.02 |

![RQ4 first look](diagrams/rq4_first_scatter.png)

### What the results mean
- **AOD does track PM2.5** (R² ≈ 0.3). That's normal for raw AOD with no weather adjustments.
- **1 km is no better than 4 km.** At the monitors the two differ by only 0.016 AOD on average, while AOD varies by 0.068 across days. Most of the signal is the regional haze level, which 4 km already sees. The single 1 km pixel is a bit noisier; the 3×3 average fixes that, but only brings it level with 4 km.
- **Within a day, AOD explains almost nothing** (R² 0.01–0.04) at either resolution. All of the AOD–PM2.5 link in June 2023 is "hazier day = more PM2.5". Within a day, the monitors differ by only about ±4 µg/m³, which is small next to the noise, and 7 sites give few places to compare.

### What this means for RQ4
With one month and 7 sites, the data **can't show** that sharper AOD tracks PM2.5 better. Discuss with Rency **before the midterm locks the RQ wording**. Options:
- **More data:** rerun on 2018–2023 (about 6× more, more seasons, smoke events where the differences between places are bigger). This is the first thing to try. It needs the 2018–2022 AOD maps from Track A.
- **Frame RQ4 as an open question:** "Does SR change the AOD–PM2.5 relationship?" A finding of "not measurably" is still a valid result.
- **Local slices:** look near highways and specific monitors, where sharp local differences are most likely.

## Things to know
- The daily AOD maps don't record which satellite pass they came from (Terra, Aqua or a blend). That's why PM2.5 uses the whole 10:00–14:00 window. **Ask Rency**; if it's known, narrow the window.
- The `.npz` patches have **no coordinates**, so matching uses the full maps in `data/raw_tifs/`. **Ask Rency** to save each patch's corner position in future patch files, so the SR model's output can be matched to monitors too.
- **Kaufman** (lon −96.318) is right at the east edge of the box.
- On cloudy days, many monitors have no AOD: 199 monitor-days had PM2.5 and a map, but only 88 were cloud-free at the monitor.

## ⏩⏩ Sprint 5 preview: sharpened AOD vs EPA (code check)

> **Sprint 5 Track A task, built early:** "run the correlation (sharpened AOD vs EPA monitors, R²)". The code works; **the numbers are not a result yet** (see below).

```
python -W ignore -m src.rq4_sr_correlation
```
Needs `src.epa_pm25` run first, and the trained models in `checkpoints/` (from Step 2 above).

### What it does (`src/rq4_sr_correlation.py`)
1. **Finds the fair days.** The models trained on the 1 km maps of early June, so on those days they've already seen the right answer. The script rebuilds Rency's patch list with the same rules as `src/sprint2_patches.ipynb` (64×64 windows every 32 pixels, at least 60% valid) and checks which day each of the 112 training patches came from. The rebuilt patches are **identical** to `data/sprint2_patches.npz`, so the list is right. Only days with **no** training patches are scored.
2. **Sharpens the AOD around each monitor.** It block-averages the day's map to 4 km, takes the 16×16 low-res tile around the monitor (the same size the models trained on), and runs each trained model on it: EDSR, U-Net and SwinIR-light. Bicubic is added as the simple baseline.
3. **Scores every version against EPA PM2.5** (R², all days and within-day), on the same monitor-days. The matched table goes to `data/epa/rq4_sr_matched.csv`.

### Results (June 2023, fair days only)
8 fair days (June 1, 2, 12, 15, 16, 28, 29, 30), **21 cloud-free monitor-days** at 7 sites. Within-day: only 2 days with at least 3 monitors (14 points).

| R² with EPA PM2.5 | All days | Within-day |
|---|---|---|
| 4 km AOD (blurry) | 0.74 | 0.43 |
| Bicubic | 0.69 | 0.32 |
| EDSR | 0.70 | 0.29 |
| U-Net | 0.69 | 0.32 |
| SwinIR-light | 0.69 | 0.31 |
| 1 km AOD (real) | 0.69 | 0.30 |
| 1 km AOD (3×3 average) | 0.71 | 0.37 |

### What the results mean
- **The pipeline works end to end:** EPA data → AOD maps → trained models → R² table.
- **Don't read anything into the numbers.** 21 points (and 14 within-day) is far too few: one or two monitor-days can move R² by 0.1 or more. The R² values are higher than in the 88-point table above just because these 8 days happen to span a wide range of haze. All versions sit within noise of each other.
- **For a real answer:** AOD maps for 2018–2023 → year split → retrain → rerun this one command, scoring the 2023 test year.

## Still to do in Sprint 4 (Track A)
- [ ] Rerun on 2018–2023 once the AOD maps for those years exist
- [x] Add the **SR model's** sharpened AOD (done early: `src/rq4_sr_correlation.py`, code check only)
- [ ] Ask Rency which satellite pass each daily map comes from
- [ ] Highway / monitor slices
- [ ] Add `rasterio` and `pandas` to `requirements.txt`
