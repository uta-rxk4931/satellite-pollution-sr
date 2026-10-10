# Sprint 2: the "learn one patch" test

## What this is in one sentence
We give the model **one** real DFW patch and train it on that same patch over and over. If it can't learn to copy even one patch, something in the code is broken.

This is a **sanity check**, not the real training. The real training script is `src/train.py` (`--data real` for the real patches); see `docs/sprint3_rq1_README.md`.

---

## How to run it
Open a terminal in the project folder and type:

```
python -W ignore -m src.overfit_one_patch
```

- Use `-m src.overfit_one_patch`, not `src/overfit_one_patch.py`. Otherwise Python can't find the other project files.
- `-W ignore` hides harmless library warnings.
- It takes about 2 minutes on CPU, or about 10 seconds on a GPU (SwinIR is slower: about 2 minutes on a GPU).

### Options (all optional)
| Option | What it does | Default |
|---|---|---|
| `--patch 43` | Train on a specific patch (0–143) | the most detailed, mostly cloud-free patch (40) |
| `--model unet` | Which model to use: `edsr`, `unet` or `swinir` | `edsr` |
| `--steps 500` | How many training steps | `2000` |
| `--lr 5e-4` | Learning rate (how big each learning step is) | `2e-4` |
| `--figure path.png` | Where to save the picture | `docs/diagrams/sprint2_overfit.png` |

---

## Key words
| Word | Meaning |
|---|---|
| **AOD** | Aerosol Optical Depth: how much haze or pollution is in the air. Around 0.15 in DFW. |
| **Patch** | A small square of the map. Here it is 64×64 pixels at 1 km. |
| **LR (low-res)** | The blurry input: 16×16 pixels, where each pixel is the average of a 4×4 block. |
| **HR (high-res)** | The sharp 1 km answer we want the model to produce (64×64). |
| **Mask** | A yes/no map: 1 = real pixel, 0 = missing (cloud or bad reading). |
| **Bicubic** | The simple "smooth stretch" way to enlarge an image. It's the baseline we must beat. |
| **RMSE** | Average error. **Lower is better.** |
| **PSNR** | Image quality in dB. **Higher is better.** |
| **SSIM** | How similar the shapes look, from 0 to 1. **Higher is better.** |
| **Detail ratio** | How sharp the output is compared with the real map. 1.0 is right; below 1 is too blurry. |

---

## The files involved

```
data/sprint2_patches.npz        <- Rency's data (144 patches, June 2023, DFW)
src/dataset.py                  <- load_real_patches(): reads that file
src/models/sr_net.py            <- build_model(): makes the model (unchanged)
src/metrics.py                  <- RMSE / PSNR / SSIM / bicubic (unchanged)
src/overfit_one_patch.py        <- THE TEST SCRIPT (this README explains it)
docs/diagrams/sprint2_overfit.png  <- the picture it saves
```

### What's inside Rency's file
| Name | Shape | Meaning |
|---|---|---|
| `lr` | 144 × 1 × 16 × 16 | blurry inputs |
| `hr` | 144 × 1 × 64 × 64 | sharp 1 km answers |
| `mask` | 144 × 1 × 64 × 64 | 1 = real pixel, 0 = missing |
| `scale` | 4 | each low-res pixel = a 4×4 block |

The values are **already real AOD**, so we never multiply them by MAIAC's 0.001 scale factor again.

---

## How the code works, step by step

### Step 0: Loading the data (`load_real_patches` in `src/dataset.py`)
Rency stores missing pixels as **0**, but our code expects them to be **blank (NaN)**. The loader:
1. Turns every pixel where `mask = 0` into blank.
2. Makes a low-res mask: a low-res pixel counts as real if **any** of its 4×4 block is real.
3. Hands everything to the existing `AODDataset`, which:
   - shifts and scales the numbers to sit around 0, because models learn better that way (this is called normalizing),
   - can undo that with `to_aod()`, so every score is reported in real AOD.

### Step 1: Pick one patch (`pick_patch`)
Out of the patches that are **at least 95% cloud-free**, it picks the one with the **most variation**. A flat, boring patch would be too easy: bicubic would already look perfect. You can override this with `--patch`.

### Step 2: Build the model (in `main`)
```python
model = build_model(args.model, scale, use_consistency=False)
```
- The default model is **EDSR**, about 1.5 million settings to learn.
- The model starts from the bicubic image and only learns the **missing detail** on top.
- `use_consistency=False` means the **average-back rule is off**, as Rency asked.
- The model also outputs a confidence map, which we **ignore** for now (`sharp, _ = model(...)`; the `_` throws it away).

### Step 3: Train on that one patch, 2000 times
Each step:
1. The model looks at the blurry input and guesses the sharp map.
2. We measure how wrong it is (the **loss**):
   ```python
   loss = ((sharp - hr) ** 2 * hr_mask).sum() / hr_mask.sum()
   ```
   This is the squared difference, averaged, and **only on real pixels**: `* hr_mask` zeroes out the missing ones.
3. `loss.backward()` and `optimizer.step()` nudge the model to be a little less wrong next time.

Every 100 steps it prints a line like:
```
step   500 | loss 4.09e-06 | RMSE 0.00011 (bicubic 0.02353) | PSNR 79.45 (bicubic 32.57) | ...
```
In each pair, the first number is the **model** and the number in brackets is **bicubic**. Watch RMSE go down and PSNR go up.

### Step 4: Score it (`score`)
It converts everything back to real AOD and computes RMSE, PSNR, SSIM and detail ratio, for **both** the model and bicubic. Missing pixels are skipped. These functions come from `src/metrics.py`.

### Step 5: Pass or fail
```python
verdict = "PASS" if model_RMSE / bicubic_RMSE < 0.1 else "FAIL"
```
**PASS** means the model's error is less than 10% of bicubic's error.

### Step 6: Save the picture (`save_figure`)
It makes a 2-row figure and saves it with `fig.savefig(...)`. Each run overwrites the old picture.

---

## How to read the picture

```
 Top row:     low-res input | bicubic | model  | 1 km target   [AOD colour bar]
 Bottom row:                | bicubic | model  |               [error colour bar]
                              error     error
```

- **Top row:** all four use the same colours. The model should look like the target, and bicubic should look blurry.
- **Bottom row:** the error maps, on the same scale. Bright = big error, black = no error.
- **RMSE** is written in the titles.
- **White spots** = missing pixels (clouds or bad readings). They're skipped in the picture and the scores.

---

## Our results (patch 40, 2000 steps, learning rate 2e-4)

All three models pass:

| | Bicubic | EDSR | U-Net | SwinIR-light |
|---|---|---|---|---|
| Parameters | – | 1.52M | 1.93M | 0.92M |
| RMSE (lower = better) | 0.0235 | 0.00017 | 0.00033 | 0.00040 |
| Error vs bicubic | – | 0.7% | 1.4% | 1.7% |
| PSNR gain over bicubic | – | +42.7 dB | +37.2 dB | +35.3 dB |
| SSIM (higher = better) | 0.798 | 1.000 | 1.000 | 1.000 |
| Detail (1.0 = right) | 0.24 (too blurry) | 1.00 | 1.00 | 1.00 |
| Result | – | **PASS** | **PASS** | **PASS** |
| Figure | – | `sprint2_overfit.png` | `sprint2_overfit_unet.png` | `sprint2_overfit_swinir.png` |

The figures are in `docs/diagrams/`. To make them again:
```
python -W ignore -m src.overfit_one_patch
python -W ignore -m src.overfit_one_patch --model unet --figure docs/diagrams/sprint2_overfit_unet.png
python -W ignore -m src.overfit_one_patch --model swinir --figure docs/diagrams/sprint2_overfit_swinir.png
```

- **EDSR** learns fastest and most steadily.
- **U-Net**'s error goes up and down during training, so its final number changes the most between runs (we've seen anywhere from 0.6% to 1.4% of bicubic's error). It passes every time.
- **SwinIR** learns slowly but steadily. It was still improving at step 2000, and it's about 10× slower per step.

Patch 43 (100% cloud-free) also passed with EDSR: 0.4% of bicubic's error.

Learning rate is 2e-4. With 1e-3, EDSR got close to perfect, then blew up around step 1200 and failed.

The script uses the same random seed every run, so results are **nearly** the same each time. On a GPU the last digits can change a little (e.g. RMSE 0.00016 vs 0.00018).

---

## What this does and doesn't prove
- ✅ The data loads correctly.
- ✅ The model trains on real data.
- ✅ The scoring works and the masks are respected.
- ❌ It does **not** show the model works on patches it has never seen. That's **Sprint 3**: train `src/train.py` on many real patches and test on different ones.

---

## Things we skipped on purpose (per Rency)
- **Confidence map** (the model's "how sure am I" output): ignored.
- **Average-back rule** (the sharp map must average back to the blurry input): turned off.

Both come back in later sprints.
