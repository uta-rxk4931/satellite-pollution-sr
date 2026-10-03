# SR Architecture Choices

Sprint 1, Track B: which super-resolution (SR) architecture to use for sharpening MAIAC AOD maps from 4 km to 1 km (s=4).

## What drives the choice

1. **The input is tiny.** A low-res patch is only 16×16 pixels, so very large models mostly add overfitting risk.
2. **AOD maps are smooth fields, not photos.** There's no texture, text or sharp edges, which is where photo-SR models gain the most.
3. **Clouds leave holes.** The model needs to use context from around a gap.
4. **Our contribution is the confidence map and the average-back rule.** Both attach to *any* network, so the base architecture matters less than getting those right.
5. **Hardware:** an RTX 3080 (10 GB) for training. A laptop CPU is enough for quick code tests.

## The top 3

### 1. EDSR (CNN), the main model
**What it is:** a stack of residual blocks (conv → ReLU → conv, plus a skip connection), then an upsampling step at the end (Lim et al., 2017). This is what [sr_net.py](sr_net.py) implements.

**Why:**
- **Simple and stable.** It removed batch norm from earlier SR models, which also suits AOD, because batch norm changes the scale of the values and AOD values matter as numbers.
- **Fast to train.** About 1–2M parameters, so a run takes minutes on a 3080.
- **Easy to extend.** The uncertainty head and the average-back rule are already working on it.
- **The standard baseline in SR papers.**

**Weakness:** each pixel only "sees" a small neighborhood, which makes it weaker at filling in around large cloud gaps.

### 2. SwinIR-light (transformer), the comparison model
**What it is:** a transformer built for image restoration (Liang et al., 2021). It splits the image into small windows and uses *attention*, so each pixel can directly look at every other pixel in its window. The windows shift between layers so information spreads across the patch.

**Why:**
- **It can look far across the patch.** Around a cloud hole it can pull information from the far side of the patch.
- **It covers the transformer side of RQ1.** The proposal cites HAT as the transformer reference, and SwinIR is HAT's direct predecessor with the same design, but about 20× smaller (about 0.9M parameters in the light version against about 20M for HAT).
- **It fits on a 3080** at 64×64 patches.

**Weakness:** slower and pickier to train than EDSR, and it needs more data to beat a CNN.

### 3. U-Net (encoder–decoder), the geoscience standard
**What it is:** it shrinks the image step by step to capture big-picture patterns, then expands back up (Ronneberger et al., 2015). Skip connections carry fine detail directly across from each shrinking step to the matching expanding step. For SR, the input is upsampled first (bicubic), then the U-Net corrects it.

**Why:**
- **It's the most-used model in weather and climate downscaling papers**, the closest field to ours.
- **It sees context at several sizes at once.** The shrunken levels capture regional haze and smoke plumes, and the full-size levels capture fine streaks.
- **It handles holes well**, because its big-picture view helps bridge cloud gaps.

**Weakness:** it's built for same-size input and output, so the image has to be upsampled first. That costs more computation than EDSR.

## Summary

| | EDSR | SwinIR-light | U-Net |
|---|---|---|---|
| Family | CNN | Transformer | Encoder–decoder CNN |
| Parameters (as built) | 1.52M | 0.92M | 1.93M |
| Training speed | Fastest | Slowest | Medium |
| Cloud gaps | Weakest | Strongest | Good |
| Role in project | Main model (Sprints 1–3) | Comparison (Sprint 6) | Comparison or backup |

## Code

| File | What's in it |
|---|---|
| [edsr.py](edsr.py) | EDSR-baseline: 64 channels, 16 residual blocks, two ×2 PixelShuffle steps. Checked equal to the official code (github.com/sanghyun-son/EDSR-PyTorch, MIT). |
| [unet.py](unet.py) | U-Net body: bicubic first, 3 levels down (64→32→16→8), widths 32/64/128/256 |
| [swinir.py](swinir.py) | SwinIR-light: a small wrapper around the official model with the paper's light settings (60 dims, 4 blocks × 6 Swin layers, 6 heads, 8×8 windows) |
| [swinir_official.py](swinir_official.py) | The official SwinIR code, unchanged ([github.com/JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR), Apache-2.0). Needs `timm`. |
| [sr_net.py](sr_net.py) | Shared shell used by all three: `sharp = bicubic + detail`, optional average-back rule, uncertainty loss, `build_model(name)` |

Every body takes `(lr, lr_mask)` and returns `(detail, log_var)`, so they're interchangeable:

```
python -m src.train --model edsr
python -m src.train --model unet
python -m src.train --model swinir
```

## Considered but not chosen

| Model | Why not |
|---|---|
| SRCNN (2014) | Only 3 layers; often barely beats bicubic |
| RCAN (2018) | EDSR plus channel attention; deeper and slower, and the gain on smooth AOD maps is unclear |
| HAT (2023) | About 20M parameters and built for photos; overkill for 16×16 inputs |
| GANs (SRGAN / ESRGAN) | Trained to *invent* realistic-looking detail, the opposite of our goal of avoiding hallucination (RQ2) |
| Diffusion (SR3, BUFF) | Uncertainty comes naturally, but slow to run and complex for this timeline |

## Plan

1. **Sprints 1–3:** build everything on EDSR, because it's the fastest way to working results for the midterm.
2. **Sprint 6:** add SwinIR-light and U-Net with the same uncertainty head and average-back rule, train all three on the same real patches, and compare:
   - RMSE / PSNR / SSIM against bicubic (RQ1)
   - coverage90, which checks whether the confidence map is honest
   - results on smoke days (e.g. Sep 19, 2020)
3. **Optional, using the GPU:** a *deep ensemble* of about 5 EDSRs with different random starts. Where they disagree is the model's own uncertainty, which complements the data-noise uncertainty from the second output head.
