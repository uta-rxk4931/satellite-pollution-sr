"""
RQ3 check: how trustworthy is the model's confidence map?

Uses the trained models in checkpoints/ and the validation patches (temporary June 2023 split).
For each model it measures:
    1. Calibration curve: for each promised level (10% ... 95%), how often the true
       1 km value actually falls inside the model's range. Perfect = on the diagonal.
    2. How uncertain the coverage number ITSELF is: resample the validation patches
       (bootstrap) and see how much coverage90 moves.
    3. Does the confidence map point at the right pixels? Coverage separately for the
       pixels the model is most / least sure about, and the rank correlation between
       predicted uncertainty and actual error.
    4. Cloud edges: coverage next to missing pixels vs away from them.

How to run:
    python -W ignore -m src.rq3_uncertainty_check
"""
import os

import matplotlib
matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src import metrics
from src.dataset import make_real_datasets
from src.models.sr_net import build_model

MODELS = [("edsr", "EDSR"), ("unet", "U-Net"), ("swinir", "SwinIR-light")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed categorical order
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"
N_BOOT = 2000


def predict(model, dataset, device):
    """Run the model on every validation patch. Returns pred, std, target, mask in AOD units."""
    lr, lr_mask = dataset.lr.to(device), dataset.lr_mask.to(device)
    with torch.no_grad():
        sharp, log_var = model(lr, lr_mask)
    pred = dataset.to_aod(sharp).cpu()
    std = (torch.exp(0.5 * log_var) * dataset.std).cpu()  # predicted std in AOD units
    return pred, std, dataset.to_aod(dataset.hr), dataset.hr_mask


def spearman(a, b):
    """Rank correlation: 1 = bigger predicted uncertainty always means bigger error."""
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    _, val, scale = make_real_datasets()
    rng = np.random.default_rng(0)

    # Pixels next to a missing pixel (cloud edge): a valid pixel with any missing neighbour
    missing = 1 - val.hr_mask
    near_gap = (F.max_pool2d(missing, 3, stride=1, padding=1) > 0).float() * val.hr_mask

    results = {}
    for name, label in MODELS:
        model = build_model(name, scale, use_consistency=False)
        model.load_state_dict(torch.load(f"checkpoints/{name}_real.pt", map_location=device))
        pred, std, target, mask = predict(model.to(device).eval(), val, device)

        # 1. Calibration curve
        curve = [metrics.coverage(pred, std, target, mask, lvl) for lvl in metrics.CALIBRATION_LEVELS]

        # 2. Bootstrap coverage90 over PATCHES (pixels in a patch aren't independent)
        n = len(pred)
        boots = []
        for _ in range(N_BOOT):
            idx = torch.tensor(rng.integers(0, n, n))
            boots.append(metrics.coverage90(pred[idx], std[idx], target[idx], mask[idx]))
        lo, hi = np.percentile(boots, [5, 95])

        # 3. Sharpness: coverage90 in the most-sure / least-sure thirds of pixels
        valid = mask.bool()
        s, err = std[valid], (pred - target).abs()[valid]
        q1, q2 = torch.quantile(s, torch.tensor([1 / 3, 2 / 3]))
        z90 = 1.6449
        inside = (err <= z90 * s).float()
        by_third = [inside[s <= q1].mean().item(),
                    inside[(s > q1) & (s <= q2)].mean().item(),
                    inside[s > q2].mean().item()]

        # 4. Cloud edges vs open sky
        edge = near_gap.bool()
        open_sky = valid & ~edge
        cov_edge = metrics.coverage90(pred[edge], std[edge], target[edge], torch.ones(int(edge.sum())))
        cov_open = metrics.coverage90(pred[open_sky], std[open_sky], target[open_sky],
                                      torch.ones(int(open_sky.sum())))

        results[name] = dict(label=label, curve=curve, cov90=curve[8], ci=(lo, hi),
                             cal_err=metrics.calibration_error(pred, std, target, mask),
                             rho=spearman(s, err), thirds=by_third,
                             edge=cov_edge, open=cov_open, n_edge=int(edge.sum()),
                             mean_std=s.mean().item(), rmse=metrics.rmse(pred, target, mask))

    # Print everything
    print(f"validation: {len(val)} patches, {int(val.hr_mask.sum()):,} valid pixels\n")
    for name, r in results.items():
        print(f"== {r['label']}")
        print(f"  coverage90 {r['cov90']:.3f}   bootstrap 90% range {r['ci'][0]:.3f}-{r['ci'][1]:.3f}   "
              f"cal_err {r['cal_err']:.3f}")
        print("  calibration curve: " + "  ".join(
            f"{lvl:.0%}->{c:.2f}" for lvl, c in zip(metrics.CALIBRATION_LEVELS, r["curve"])))
        print(f"  mean predicted std {r['mean_std']:.4f}   RMSE {r['rmse']:.4f}")
        print(f"  rank corr(uncertainty, |error|) {r['rho']:.2f}")
        print(f"  coverage90 by predicted-std third (most sure / middle / least sure): "
              + " / ".join(f"{x:.2f}" for x in r["thirds"]))
        print(f"  coverage90 near cloud edges {r['edge']:.2f} ({r['n_edge']:,} px)   open sky {r['open']:.2f}\n")

    # Figure: calibration curves (left) and coverage90 with its bootstrap range (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained",
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    levels = metrics.CALIBRATION_LEVELS
    ax1.plot([0, 1], [0, 1], color=INK_SOFT, linewidth=1.2, linestyle="--")
    for (name, _), color in zip(MODELS, COLORS):
        r = results[name]
        ax1.plot(levels, r["curve"], color=color, linewidth=2, marker="o", markersize=5, label=r["label"])
    ax1.set_xlabel("promised coverage (model says: \"x% sure\")", color=INK_SOFT)
    ax1.set_ylabel("actual coverage (true value inside the range)", color=INK_SOFT)
    ax1.set_title("Calibration curve (dashed = perfect; below it = overconfident)", color=INK, fontsize=11)
    ax1.legend(frameon=False)

    for i, ((name, _), color) in enumerate(zip(MODELS, COLORS)):
        r = results[name]
        ax2.plot([r["ci"][0], r["ci"][1]], [i, i], color=color, linewidth=3, solid_capstyle="round")
        ax2.plot(r["cov90"], i, "o", color=color, markersize=9, markeredgecolor="white")
    ax2.axvline(0.9, color=INK_SOFT, linestyle="--", linewidth=1.2)
    ax2.text(0.9, len(MODELS) - 0.45, " target 0.90", color=INK_SOFT, fontsize=9)
    ax2.set_yticks(range(len(MODELS)), [r["label"] for r in results.values()])
    ax2.set_ylim(-0.6, len(MODELS) - 0.3)
    ax2.set_xlabel("coverage at 90% (dot) and its bootstrap 90% range (bar)", color=INK_SOFT)
    ax2.set_title(f"How sure are we of coverage90? ({len(val)} val patches)", color=INK, fontsize=11)

    for ax in (ax1, ax2):
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_SOFT)
    fig.suptitle("RQ3 check: is the confidence map honest? (June 2023 validation patches)", color=INK)
    os.makedirs("docs/diagrams", exist_ok=True)
    fig.savefig("docs/diagrams/rq3_calibration.png", dpi=120, facecolor="#fcfcfb")
    print("saved docs/diagrams/rq3_calibration.png")


if __name__ == "__main__":
    main()
