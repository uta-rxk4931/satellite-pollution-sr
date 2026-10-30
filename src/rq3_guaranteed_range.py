"""
RQ3(b): turn the confidence map into a range with a GUARANTEE.

Question: can every 1 km pixel get a range  y_hat +- lambda * sigma_hat  that contains the true value
for at least 90% of pixels in an unseen year, and is that range narrower than one that ignores sigma_hat?

Method (conformal risk control, as in im2im-uq / PUQ, Belhasin et al.):
    1. Calibration year (2022): for each patch, the "coverage loss" L(lambda) = fraction of its real
       pixels whose true value falls OUTSIDE the range. Average over patches = risk.
    2. Add a safety margin for having only n calibration patches (Hoeffding): sqrt(ln(1/delta) / 2n).
    3. Pick the SMALLEST lambda whose risk + margin <= alpha (alpha = 10% misses).
       Then, with probability >= 1 - delta, the average miss rate on new data is <= alpha.
    4. Check it on 2023 (never used for anything else).

Baseline: the same procedure for a CONSTANT range  y_hat +- c  that ignores sigma_hat. Both get the same
guarantee on average. If sigma_hat is real uncertainty, its range puts the width where it is needed:
the misses are spread evenly, instead of piling up on the hard pixels (cloud edges, hazy days).

How to run (needs the _years checkpoints from `python -m src.train --data years ...`):
    python -W ignore -m src.rq3_guaranteed_range
Saves runs/rq3b_lambda.json and two figures in docs/diagrams/.
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import make_year_datasets
from src.models.sr_net import build_model
from src.rq3_uncertainty_check import COLORS, GRID, INK, INK_SOFT
from src.rq3_uncertainty_maps import predict

ALPHA = 0.10   # allowed miss rate (10%)
DELTA = 0.10   # chance the guarantee itself fails (it holds with 90% confidence)
LAMBDAS = torch.arange(0.0, 8.0, 0.005)     # multipliers for sigma_hat
WIDTHS = torch.arange(0.0, 0.25, 0.0001)    # half-widths (AOD) for the constant range

ARCHS = [("edsr", "EDSR"), ("unet", "U-Net"), ("swinir", "SwinIR-light")]
RUNS = [(a, rule) for a, _ in ARCHS for rule in (False, True)]
LABELS = dict(ARCHS)


def tag_of(arch, rule):
    return f"{arch}{'_consistency' if rule else ''}"


def label_of(arch, rule):
    return f"{LABELS[arch]}{' + rule' if rule else ''}"


def patch_risk(abs_err, scale, mask, grid):
    """
    Coverage loss for every patch and every grid value:
    fraction of real pixels with abs_err > g * scale. Returns (patches, grid).
    """
    n = abs_err.shape[0]
    ratio = (abs_err / scale).reshape(n, -1)
    valid = mask.reshape(n, -1).bool()
    ratio = torch.where(valid, ratio, torch.full_like(ratio, -1.0))  # cloud pixels never count as a miss
    ratio, _ = ratio.sort(dim=1)
    inside_or_cloud = torch.searchsorted(ratio, grid.expand(n, -1).contiguous(), right=True)
    outside = ratio.shape[1] - inside_or_cloud
    return outside / valid.sum(dim=1, keepdim=True)


def calibrate(risk_per_patch, grid):
    """Smallest grid value whose average risk + Hoeffding margin <= ALPHA."""
    n = risk_per_patch.shape[0]
    bound = risk_per_patch.mean(dim=0) + math.sqrt(math.log(1 / DELTA) / (2 * n))
    ok = torch.where(bound <= ALPHA)[0]
    return grid[ok[0]].item(), bound


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    _, val, test, scale = make_year_datasets()
    margin = math.sqrt(math.log(1 / DELTA) / (2 * len(val)))
    print(f"calibration 2022: {len(val)} patches, test 2023: {len(test)} patches")
    print(f"target: <= {ALPHA:.0%} misses, guaranteed with {1 - DELTA:.0%} confidence "
          f"(safety margin {margin:.3f}, so 2022 must be <= {ALPHA - margin:.1%})\n")

    # 2023 groups for "where does it fail?"
    t_mask = test.hr_mask
    t_aod = test.to_aod(test.hr)
    mean_aod = (t_aod * t_mask).sum(dim=(1, 2, 3)) / t_mask.sum(dim=(1, 2, 3))
    groups = {
        "clear patches": (t_mask.mean(dim=(1, 2, 3)) == 1)[:, None, None, None] * t_mask,
        "cloud edges": (F.max_pool2d(1 - t_mask, 3, stride=1, padding=1) > 0).float() * t_mask,
        "high-AOD patches (top 10%)": (mean_aod >= torch.quantile(mean_aod, 0.9))[:, None, None, None] * t_mask,
    }

    results = {}
    for arch, rule in RUNS:
        tag = tag_of(arch, rule)
        model = build_model(arch, scale, use_consistency=rule)
        model.load_state_dict(torch.load(f"checkpoints/{tag}_years.pt", map_location=device))
        model = model.to(device).eval()

        r = {}
        for name, ds in (("cal", val), ("test", test)):
            pred, std = predict(model, ds, device)
            r[name] = ((pred - ds.to_aod(ds.hr)).abs(), std, ds.hr_mask)

        # Calibrate both ranges on 2022
        err, std, mask = r["cal"]
        lam, lam_bound = calibrate(patch_risk(err, std, mask, LAMBDAS), LAMBDAS)
        c, c_bound = calibrate(patch_risk(err, torch.ones_like(std), mask, WIDTHS), WIDTHS)

        # Check on 2023
        err, std, mask = r["test"]
        valid = mask.bool()
        miss = patch_risk(err, std, mask, torch.tensor([lam]))[:, 0].mean().item()
        miss_c = patch_risk(err, torch.ones_like(std), mask, torch.tensor([c]))[:, 0].mean().item()
        miss_raw = patch_risk(err, std, mask, torch.tensor([1.645]))[:, 0].mean().item()
        width = (2 * lam * std)[valid].mean().item()
        by_group = {}
        for g, gm in groups.items():
            gv = gm.bool()
            by_group[g] = dict(miss=(err[gv] > lam * std[gv]).float().mean().item(),
                               miss_c=(err[gv] > c).float().mean().item(),
                               width=(2 * lam * std[gv]).mean().item())

        results[tag] = dict(label=label_of(arch, rule), arch=arch, rule=rule, lam=lam, c=c,
                            miss=miss, miss_c=miss_c, miss_raw=miss_raw, width=width, width_c=2 * c,
                            groups=by_group, lam_bound=lam_bound, c_bound=c_bound)

    # Print
    print(f"{'model':<22} {'lambda':>7} {'raw 90% miss':>13} {'2023 miss':>10} {'width':>8} "
          f"{'| constant c':>12} {'2023 miss':>10} {'width':>8} {'sigma saves':>12}")
    for r in results.values():
        print(f"{r['label']:<22} {r['lam']:>7.3f} {r['miss_raw']:>13.1%} {r['miss']:>10.1%} {r['width']:>8.4f} "
              f"{'| ' + format(r['c'], '.4f'):>12} {r['miss_c']:>10.1%} {r['width_c']:>8.4f} "
              f"{1 - r['width'] / r['width_c']:>12.0%}")
    print("\n2023 by group (sigma range: miss / mean width | constant range: miss)")
    for r in results.values():
        print(f"  {r['label']}")
        for g, v in r["groups"].items():
            print(f"    {g:<28} miss {v['miss']:>6.1%}  width {v['width']:.4f}   | constant miss {v['miss_c']:>6.1%}")

    os.makedirs("runs", exist_ok=True)
    with open("runs/rq3b_lambda.json", "w") as f:
        json.dump({t: {"lambda": r["lam"], "c": r["c"]} for t, r in results.items()}, f, indent=2)
    print("\nsaved runs/rq3b_lambda.json")

    color = {a: col for (a, _), col in zip(ARCHS, COLORS)}

    # Figure 1: 2022 risk (+ margin) vs lambda
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    for r in results.values():
        ls = "--" if r["rule"] else "-"
        ax.plot(LAMBDAS, r["lam_bound"], color=color[r["arch"]], ls=ls, lw=1.8, label=r["label"])
        ax.plot(r["lam"], ALPHA, "o", color=color[r["arch"]], markeredgecolor="white", markersize=8)
    ax.axhline(ALPHA, color=INK_SOFT, ls=":", lw=1.2)
    ax.axvline(1.645, color=INK_SOFT, ls=":", lw=1)
    ax.text(1.66, 0.42, "raw 90% range\n(λ = 1.645)", color=INK_SOFT, fontsize=8)
    ax.text(4.95, ALPHA + 0.01, "allowed: 10% misses", color=INK_SOFT, fontsize=8, ha="right")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("λ  (range = ŷ ± λ·σ̂)", color=INK_SOFT)
    ax.set_ylabel("2022 miss rate + safety margin", color=INK_SOFT)
    ax.set_title("Calibrating on 2022: the dot is the smallest λ that keeps misses ≤ 10% (with 90% confidence)",
                 color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for a in (ax,):
        a.grid(color=GRID, lw=0.8)
        a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig("docs/diagrams/rq3b_risk_vs_lambda.png", dpi=120, facecolor="#fcfcfb")

    # Figure 2: 2023, sigma range vs constant range, both with the same guarantee.
    # Left: how wide. Right: WHERE the misses land.
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 4.8), layout="constrained",
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    rs = list(results.values())
    y = np.arange(len(rs))
    ax.barh(y + 0.2, [r["width"] for r in rs], height=0.38, color=[color[r["arch"]] for r in rs],
            label="range from σ̂  (ŷ ± λ̂σ̂)")
    ax.barh(y - 0.2, [r["width_c"] for r in rs], height=0.38, color="#c9c8c3", label="constant range  (ŷ ± c)")
    for i, r in enumerate(rs):
        ax.text(r["width"] + 0.0005, i + 0.2, f"{r['width']:.4f}  (2023 miss {r['miss']:.1%})", va="center",
                fontsize=8, color=INK)
        ax.text(r["width_c"] + 0.0005, i - 0.2, f"{r['width_c']:.4f}  (2023 miss {r['miss_c']:.1%})", va="center",
                fontsize=8, color=INK_SOFT)
    ax.set_yticks(y, [r["label"] for r in rs])
    ax.invert_yaxis()
    ax.set_xlim(0, max(r["width_c"] for r in rs) * 1.45)
    ax.set_xlabel("average width of the guaranteed 90% range on 2023 (AOD)  —  narrower is better", color=INK_SOFT)
    ax.set_title("Same guarantee (≤10% misses, 90% confidence): how wide on average?", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)

    # Right: miss rate by group. Each model = one coloured dot (sigma range) and one grey dot (constant).
    names = list(groups)
    for i, r in enumerate(rs):
        dx = (i - (len(rs) - 1) / 2) * 0.08
        for k, g in enumerate(names):
            v = r["groups"][g]
            ax2.plot([k + dx, k + dx], [v["miss"], v["miss_c"]], color=GRID, lw=1, zorder=1)
            ax2.plot(k + dx, v["miss_c"], "o", color="#b5b4ae", markersize=6, zorder=2,
                     label="constant range" if i == 0 and k == 0 else None)
            ax2.plot(k + dx, v["miss"], "o", color=color[r["arch"]], markersize=6, zorder=3,
                     markerfacecolor="white" if r["rule"] else color[r["arch"]],
                     label=r["label"] if k == 0 else None)
    ax2.axhline(ALPHA, color=INK_SOFT, ls=":", lw=1.2)
    ax2.text(2.45, ALPHA + 0.004, "10%", color=INK_SOFT, fontsize=8, ha="right")
    ax2.set_xticks(range(len(names)), names)
    ax2.set_ylim(0, 0.23)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.set_ylabel("2023 miss rate (truth outside the range)", color=INK_SOFT)
    ax2.set_title("Where do the misses land? σ̂ moves width to the hard pixels", color=INK, fontsize=10)
    ax2.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")

    for a in (ax, ax2):
        a.grid(axis="x" if a is ax else "y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig("docs/diagrams/rq3b_width_vs_coverage.png", dpi=120, facecolor="#fcfcfb")
    print("saved docs/diagrams/rq3b_risk_vs_lambda.png, docs/diagrams/rq3b_width_vs_coverage.png")


if __name__ == "__main__":
    main()
