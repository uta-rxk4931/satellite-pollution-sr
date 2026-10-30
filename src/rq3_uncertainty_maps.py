"""
RQ3 as pictures: what the confidence map is, and whether it can be trusted pixel by pixel.

The model outputs TWO maps from the blurry input alone, without ever seeing the true 1 km map:
    prediction  y_hat      - its best guess of the sharp AOD
    uncertainty sigma_hat  - how far off it thinks it might be, at each pixel

One row per 2023 test patch (year split), in three parts:

    1. MODEL OUTPUT (no truth used):   low-res input | prediction | uncertainty sigma_hat
    2. CHECK vs TRUTH:                 true 1 km | actual error |y - y_hat| | outside 90% range
    3. ONE LINE, PIXEL BY PIXEL:       the prediction along one row of the patch, with its
                                       90% range (y_hat +- 1.645 sigma_hat) as a band, and the
                                       true values as dots. Red dots fall outside the band.

An honest uncertainty map has about 10% of true values outside its 90% range, and its
bright spots (high sigma_hat) sit where the actual error is big.

The three patches are picked automatically from 2023 (test):
    clear day  - fully cloud-free, typical AOD
    cloudy day - the most cloud edges (where the model has the least information)
    high AOD   - the haziest patch (smoke / dust)

How to run (needs the _years checkpoints from `python -m src.train --data years ...`):
    python -W ignore -m src.rq3_uncertainty_maps                         # SwinIR-light
    python -W ignore -m src.rq3_uncertainty_maps --model edsr
    python -W ignore -m src.rq3_uncertainty_maps --model edsr --consistency
    python -W ignore -m src.rq3_uncertainty_maps --patches 12 340 901    # your own patches
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap

from src.dataset import make_year_datasets
from src.models.sr_net import build_model
from src.rq3_uncertainty_check import spearman

LABELS = {"edsr": "EDSR", "unet": "U-Net", "swinir": "SwinIR-light"}
Z90 = 1.645  # 90% of a normal distribution is within 1.645 standard deviations
BAND, LINE, INSIDE, OUTSIDE = "#9ecae1", "#2a78d6", "#0b0b0b", "#d62728"


def predict(model, dataset, device, batch=64):
    """Prediction and predicted std (AOD units) for every patch, in batches to save GPU memory."""
    preds, stds = [], []
    with torch.no_grad():
        for i in range(0, len(dataset), batch):
            lr = dataset.lr[i:i + batch].to(device)
            lr_mask = dataset.lr_mask[i:i + batch].to(device)
            sharp, log_var = model(lr, lr_mask)
            preds.append(dataset.to_aod(sharp).cpu())
            stds.append((torch.exp(0.5 * log_var) * dataset.std).cpu())
    return torch.cat(preds), torch.cat(stds)


def pick_patches(test):
    """Indices of a clear-day, a cloudy-day and a high-AOD patch from the test year."""
    mask = test.hr_mask
    valid = mask.mean(dim=(1, 2, 3))
    aod = test.to_aod(test.hr)
    mean_aod = (aod * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))

    # clear day: fully cloud-free, AOD closest to the typical (median) clear patch
    clear = torch.where(valid == 1)[0]
    clear_idx = clear[(mean_aod[clear] - mean_aod[clear].median()).abs().argmin()].item()

    # cloudy day: the most valid pixels sitting next to a cloud
    near_gap = (F.max_pool2d(1 - mask, 3, stride=1, padding=1) > 0).float() * mask
    cloudy_idx = near_gap.sum(dim=(1, 2, 3)).argmax().item()

    # high AOD: the haziest patch
    hazy_idx = mean_aod.argmax().item()
    return [(clear_idx, "clear day"), (cloudy_idx, "cloudy day"), (hazy_idx, "high AOD")]


def pick_line(std, mask, border=8):
    """
    Row for the profile plot: among rows that are at least 75% cloud-free, the most uncertain one.
    Rows near the patch border are skipped: sigma_hat is always high there (no neighbours on one side).
    """
    valid_frac = mask.mean(dim=1)
    mean_std = (std * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    mean_std[valid_frac < 0.75] = -1
    mean_std[:border] = -1
    mean_std[-border:] = -1
    return int(mean_std.argmax())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="swinir", choices=list(LABELS))
    parser.add_argument("--consistency", action="store_true", help="use the model trained with the average-back rule")
    parser.add_argument("--patches", type=int, nargs="+", help="test patch numbers (0-975) instead of the automatic three")
    parser.add_argument("--figure", help="where to save the picture")
    parser.add_argument("--calibrated", action="store_true",
                        help="use the guaranteed range from src.rq3_guaranteed_range (runs/rq3b_lambda.json)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    _, _, test, scale = make_year_datasets()

    tag = f"{args.model}{'_consistency' if args.consistency else ''}"
    label = f"{LABELS[args.model]}{' + average-back rule' if args.consistency else ''}"

    # Raw 90% range (1.645 sigma), or the guaranteed range calibrated on 2022 (lambda_hat sigma)
    z, range_name = Z90, "90% range ŷ ± 1.645σ̂"
    if args.calibrated:
        with open("runs/rq3b_lambda.json") as f:
            z = json.load(f)[tag]["lambda"]
        range_name = f"guaranteed range ŷ ± {z:.2f}σ̂"
    model = build_model(args.model, scale, use_consistency=args.consistency)
    model.load_state_dict(torch.load(f"checkpoints/{tag}_years.pt", map_location=device))
    pred, std = predict(model.to(device).eval(), test, device)
    target, mask = test.to_aod(test.hr), test.hr_mask
    lr, lr_mask = test.to_aod(test.lr), test.lr_mask

    rows = [(i, f"patch {i}") for i in args.patches] if args.patches else pick_patches(test)

    # whole 2023 test year, for the title
    m = mask.bool()
    outside_all = ((target - pred).abs() > z * std)[m].float().mean().item()
    print(f"{label}: {outside_all:.1%} of all 2023 pixels outside the {range_name} (target <= 10%)")

    blank = lambda x, mk: np.where(mk.numpy() > 0, x.numpy(), np.nan)
    outside_cmap = ListedColormap(["#d9d8d3", OUTSIDE])  # grey = inside range, red = outside
    # Blues, but starting at light blue, so low uncertainty can't be confused with white cloud pixels
    sigma_cmap = ListedColormap(plt.cm.Blues(np.linspace(0.2, 1, 256)))

    n = len(rows)
    fig, axes = plt.subplots(n, 7, figsize=(26, 3.7 * n), layout="constrained", squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 1.8]})

    for r, (i, name) in enumerate(rows):
        mk = mask[i, 0]
        valid = mk.bool()
        y, y_hat, s = target[i, 0], pred[i, 0], std[i, 0]
        err = (y - y_hat).abs()
        outside = (err > z * s).float()
        line = pick_line(s, mk)

        # same AOD scale for input / prediction / truth
        vmin, vmax = y[valid].min().item(), y[valid].max().item()
        err_max = torch.quantile(err[valid], 0.99).item()

        # (image, title, colour map, min, max, colour-bar label)
        panels = [
            (blank(lr[i, 0], lr_mask[i, 0]), "low-res input (4 km)", "viridis", vmin, vmax, None),
            (blank(y_hat, mk), "prediction ŷ (1 km)", "viridis", vmin, vmax, None),
            (blank(s, mk), "UNCERTAINTY σ̂\n(model output, no truth used)", sigma_cmap, 0,
             torch.quantile(s[valid], 0.99).item(), "σ̂ (AOD)"),
            (blank(y, mk), "true 1 km y", "viridis", vmin, vmax, "AOD"),
            (blank(err, mk), "actual error |y − ŷ|", "magma", 0, err_max, "|error| (AOD)"),
            (blank(outside, mk), f"truth outside {range_name}\n{outside[valid].mean():.0%} of pixels "
                                 f"(target ≤ 10%)", outside_cmap, 0, 1, None),
        ]
        for c, (img, title, cmap, lo, hi, bar) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(title, fontsize=10, fontweight="bold" if c == 2 else "normal")
            ax.set_xticks([])
            ax.set_yticks([])
            if bar:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=bar)
        # dashed line = the row used in the profile plot
        for c in (1, 2, 3):
            axes[r, c].axhline(line, color="white" if c != 2 else "#0b0b0b", ls="--", lw=1)
        axes[r, 0].set_ylabel(f"{name}\n(patch {i})", fontsize=11)

        # Profile: one row of pixels, prediction +- 90% range, true values as dots
        ax = axes[r, 6]
        x = np.arange(y.shape[-1])
        ok = valid[line].numpy()
        yh, sl, yt = y_hat[line].numpy(), s[line].numpy(), y[line].numpy()
        ax.fill_between(x, yh - z * sl, yh + z * sl, color=BAND, alpha=0.7, lw=0,
                        label=f"model's range {range_name}")
        ax.plot(x, yh, color=LINE, lw=1.5, label="prediction ŷ")
        out_line = ok & (np.abs(yt - yh) > z * sl)
        ax.scatter(x[ok & ~out_line], yt[ok & ~out_line], s=12, color=INSIDE, zorder=3, label="true value, inside")
        ax.scatter(x[out_line], yt[out_line], s=16, color=OUTSIDE, zorder=3, label="true value, outside")
        rho = spearman(s[valid], err[valid])
        ax.set_title(f"along the dashed line (row {line}): {out_line.sum()} of {ok.sum()} outside\n"
                     f"rank corr σ̂ vs error, whole patch: {rho:.2f}", fontsize=10)
        ax.set_xlabel("pixel (1 km) along the line")
        ax.set_ylabel("AOD")
        ax.spines[["top", "right"]].set_visible(False)
        if r == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper left")

    fig.suptitle(f"{label} on the 2023 test year.  Left: what the model outputs.  "
                 f"Middle: checked against the truth.  Right: pixel by pixel along one line.   "
                 f"Whole year: {outside_all:.1%} of pixels outside the {range_name} (target ≤ 10%)", fontsize=12)
    path = args.figure or f"docs/diagrams/rq3_maps_{tag}{'_calibrated' if args.calibrated else ''}.png"
    fig.savefig(path, dpi=110)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
