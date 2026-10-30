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

With --data years (2023 test year, _years checkpoints) it also answers RQ3(a), "does sigma_hat point at
the pixels where the model is actually wrong?" (following Zheng et al. 2026):
    5. Error vs sigma_hat: group pixels by predicted sigma_hat, and compare with the RMS error each group
       really has. On the diagonal = sigma_hat is the size of the real error.
    6. V-RMSE (sigma_hat^2 vs actual squared error) and sharpness (average 90% width).
    7. The 16 positions inside each 4x4 low-res block: does sigma_hat follow the error there too?

How to run:
    python -W ignore -m src.rq3_uncertainty_check                # June 2023 split (original note)
    python -W ignore -m src.rq3_uncertainty_check --data years   # 2023 test year
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src import metrics
from src.dataset import make_real_datasets, make_year_datasets
from src.models.sr_net import build_model

MODELS = [("edsr", "EDSR"), ("unet", "U-Net"), ("swinir", "SwinIR-light")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed categorical order
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"
N_BOOT = 2000


def predict(model, dataset, device, batch=64):
    """Run the model on every patch (in batches). Returns pred, std, target, mask in AOD units."""
    preds, stds = [], []
    with torch.no_grad():
        for i in range(0, len(dataset), batch):
            sharp, log_var = model(dataset.lr[i:i + batch].to(device), dataset.lr_mask[i:i + batch].to(device))
            preds.append(dataset.to_aod(sharp).cpu())
            stds.append((torch.exp(0.5 * log_var) * dataset.std).cpu())  # predicted std in AOD units
    return torch.cat(preds), torch.cat(stds), dataset.to_aod(dataset.hr), dataset.hr_mask


def error_vs_sigma(err, s, bins=10):
    """Group pixels into equal-size bins by predicted sigma. Returns (mean sigma, RMS error) per bin."""
    order = s.argsort()
    return ([s[i].mean().item() for i in order.chunk(bins)],
            [err[i].pow(2).mean().sqrt().item() for i in order.chunk(bins)])


def by_position(err, s, mask, scale):
    """RMS error and mean sigma for each of the scale x scale positions inside a low-res block."""
    n, c, h, w = mask.shape
    blocks = lambda x: x.reshape(n, c, h // scale, scale, w // scale, scale)
    m = blocks(mask)
    rms = ((blocks(err) ** 2 * m).sum(dim=(0, 1, 2, 4)) / m.sum(dim=(0, 1, 2, 4))).sqrt()
    sig = (blocks(s) * m).sum(dim=(0, 1, 2, 4)) / m.sum(dim=(0, 1, 2, 4))
    return rms, sig


def spearman(a, b):
    """Rank correlation: 1 = bigger predicted uncertainty always means bigger error."""
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="real", choices=["real", "years"],
                        help="June 2023 validation patches, or the 2023 test year of the year split")
    args = parser.parse_args()
    years = args.data == "years"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    if years:
        _, _, val, scale = make_year_datasets()  # "val" below = the 2023 TEST year here
    else:
        _, val, scale = make_real_datasets()
    rng = np.random.default_rng(0)

    # Pixels next to a missing pixel (cloud edge): a valid pixel with any missing neighbour
    missing = 1 - val.hr_mask
    near_gap = (F.max_pool2d(missing, 3, stride=1, padding=1) > 0).float() * val.hr_mask

    results = {}
    for name, label in MODELS:
        model = build_model(name, scale, use_consistency=False)
        model.load_state_dict(torch.load(f"checkpoints/{name}_{args.data}.pt", map_location=device))
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
                             mean_std=s.mean().item(), rmse=metrics.rmse(pred, target, mask),
                             v_rmse=metrics.v_rmse(pred, std, target, mask),
                             sharp=metrics.sharpness(std, mask),
                             bins=error_vs_sigma(err, s),
                             pos=by_position((pred - target).abs(), std, mask, scale))

    # Print everything
    print(f"{'2023 test' if years else 'validation'}: {len(val)} patches, {int(val.hr_mask.sum()):,} valid pixels\n")
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
        print(f"  coverage90 near cloud edges {r['edge']:.2f} ({r['n_edge']:,} px)   open sky {r['open']:.2f}")
        print(f"  V-RMSE {r['v_rmse']:.2e}   sharpness (mean 90% width) {r['sharp']:.4f}")
        sig, rms = r["bins"]
        print("  error vs sigma (10 groups, most sure -> least sure), sigma / real RMS error: "
              + "  ".join(f"{a:.4f}/{b:.4f}" for a, b in zip(sig, rms)))
        pos_rms, pos_sig = r["pos"]
        print(f"  4x4 block positions: rank corr(sigma, RMS error) {spearman(pos_sig.flatten(), pos_rms.flatten()):.2f}")
        print("    RMS error:  " + " | ".join(" ".join(f"{x:.4f}" for x in row) for row in pos_rms.tolist()))
        print("    mean sigma: " + " | ".join(" ".join(f"{x:.4f}" for x in row) for row in pos_sig.tolist()) + "\n")

    # Figure: calibration curves (left) and coverage90 with its bootstrap range (right)
    if years:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained",
                                            gridspec_kw={"width_ratios": [1.3, 1, 1.2]})
    else:
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
    ax2.set_title(f"How sure are we of coverage90? ({len(val)} {'test' if years else 'val'} patches)",
                  color=INK, fontsize=11)
    axes = [ax1, ax2]

    if years:
        # RQ3(a): does sigma follow the real error? Each dot = 10% of pixels, grouped by sigma.
        top = 0
        for (name, _), color in zip(MODELS, COLORS):
            sig, rms = results[name]["bins"]
            ax3.plot(sig, rms, color=color, lw=2, marker="o", markersize=5, label=results[name]["label"])
            top = max(top, max(sig), max(rms))
        ax3.plot([0, top * 1.05], [0, top * 1.05], color=INK_SOFT, lw=1.2, ls="--")
        ax3.set_xlim(0, top * 1.05)
        ax3.set_ylim(0, top * 1.05)
        ax3.set_xlabel("predicted σ̂ (group average, AOD)", color=INK_SOFT)
        ax3.set_ylabel("real RMS error of that group (AOD)", color=INK_SOFT)
        ax3.set_title("Does σ̂ follow the real error? (dashed = exactly)", color=INK, fontsize=11)
        ax3.legend(frameon=False)
        axes.append(ax3)

    for ax in axes:
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_SOFT)
    fig.suptitle("RQ3 check: is the confidence map honest? "
                 + ("(2023 test year, year split)" if years else "(June 2023 validation patches)"), color=INK)
    os.makedirs("docs/diagrams", exist_ok=True)
    path = f"docs/diagrams/rq3_calibration{'_years' if years else ''}.png"
    fig.savefig(path, dpi=120, facecolor="#fcfcfb")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
