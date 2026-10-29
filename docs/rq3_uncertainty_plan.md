# To-do on the better machine: proving the uncertainty map (RQ3)

## Why
The model outputs two maps per patch: the sharp AOD **ŷ** and its uncertainty **σ̂** (`log_var` in `src/models/sr_net.py`). The loss (`uncertainty_loss`) only *teaches* σ̂. To show σ̂ can be trusted, test it on data the model never saw (the 2023 test year) against the true 1 km values.

So far it has only been checked on 3 June days (`docs/rq3_uncertainty_note.md`). Those checks found it overconfident at cloud edges and on its most-confident pixels, and unstable between epochs.

## Step 0: Get set up
- [ ] `git pull` on `ryan-dev` (it includes the merge with main).
- [ ] Check that `data/split_data/` contains `train.npz` (2018–21, 4543 patches), `val.npz` (2022, 1142), `test.npz` (2023, 976) and `train_stats.npz`.
- [ ] Check that the GPU works: `python -c "import torch; print(torch.cuda.is_available())"`.

## Step 1: Train on the split by year
- [ ] In `src/dataset.py`, add a loader for `data/split_data/`:
  - reuse `read_real_patches()` for train, val and test
  - normalise with `mean`/`std` from `train_stats.npz`
- [ ] In `src/train.py`, add a `--data years` option. Pick the epoch using **2022 val only**, then score **2023 test** once at the end.
- [ ] Train EDSR, U-Net and SwinIR, each with and without `--consistency`.
- [ ] Sanity check: the model should beat bicubic on 2023 test (RMSE/PSNR).

## Step 2: Error map vs uncertainty map (images)
For a few 2023 test patches, plot side by side:

```
LR input | sharp ŷ | true 1 km y | |ŷ − y| error | σ̂ uncertainty
```

- [ ] Draw the error and σ̂ panels with the same colour scale, and show cloud pixels as white.
- [ ] The plotting code in `src/overfit_one_patch.py` already has this layout. Add the σ̂ panel.
- [ ] Pick examples: a clear day, a cloudy day, and a high-AOD (smoky) day.
- [ ] **Look for:** bright spots in σ̂ should match bright spots in the error map. A spot that's bright in the error map but dark in σ̂ is overconfident.

## Step 3: Prove it with every pixel in 2023 test
Use `src/rq3_uncertainty_check.py`, which already computes coverage and calibration:

- [ ] **Calibration curve:** at confidence levels 10%…95%, check how often the true value falls inside ŷ ± z·σ̂. Plot promised vs actual; the diagonal means honest. This is `docs/diagrams/rq3_calibration.png` redone on 2023.
- [ ] **Error grows with σ̂:** group pixels by σ̂ (low → high) and plot the average real error per group. It should rise. Also report the rank correlation (`spearman()` already exists).
- [ ] **Throw out the unsure pixels:** remove the highest-σ̂ pixels first and recompute RMSE. It should drop faster than when removing random pixels.
- [ ] **Where it fails:** compare coverage at cloud edges vs open sky, and on high-AOD vs normal patches.

## Step 4: Fix it if it's off
- [ ] Find one number k so that k·σ̂ gives 90% coverage on **2022 val**.
- [ ] Apply k to 2023 test and report coverage before and after.

## Step 5 (if time): Ensemble
- [ ] Train 5 EDSRs with seeds 0–4.
- [ ] Where they disagree is model uncertainty ("never seen this before"), which σ̂ alone can't capture. Show it on the smoky-day example.

## Needs from Rency
- [ ] The patch files store only the year. To bootstrap by day and label examples by date or season, ask her to also save `date` and patch `(row, col)` in `src/Data extraction/Data extraction_make_patches.py`. Her script already parses the date from the filename.

## Done when
- [ ] 5-panel image figure (Step 2)
- [ ] Calibration curve, error-vs-σ̂ plot and remove-pixels plot on 2023 (Step 3)
- [ ] Coverage before and after the k fix (Step 4)
- [ ] One paragraph of limits: σ̂ does **not** include MAIAC's own measurement error (see `rq3_uncertainty_note.md`, Evidence 5)
