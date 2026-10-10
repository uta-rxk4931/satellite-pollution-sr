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
