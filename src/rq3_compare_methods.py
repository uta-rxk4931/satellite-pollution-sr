"""
RQ3: which way of training the confidence map works best, and which prevents overconfidence?

Methods (trained by runs/methods/run_all.ps1; B0 is the original run, kept for comparison):
    B0  Gaussian NLL, epoch picked by 2022 RMSE           checkpoints/<model>_years.pt
    M1  Gaussian NLL, epoch picked by 2022 NLL, clipping  checkpoints/methods/<model>_m1.pt
    M2  beta-NLL (beta 0.5) + M1's settings               checkpoints/methods/<model>_m2.pt
    M3  M1 + weight decay 1e-4 (U-Net only)               checkpoints/methods/unet_m3.pt
    M4  deep ensemble: 5 seeds of M1 (U-Net, EDSR)        <model>_m1 + <model>_m1_seed1..4
        prediction = mean of the 5; variance = mean of their sigma^2 + spread of their predictions

Everything is scored on 2023 (test). 2022 is used only to calibrate the guaranteed range (lambda).

How to run (after the experiments):
    python -W ignore -m src.rq3_compare_methods
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import metrics
from src.dataset import make_year_datasets
from src.models.sr_net import build_model
from src.rq3_guaranteed_range import LAMBDAS, calibrate, hard_groups, patch_risk
from src.rq3_uncertainty_check import GRID, INK, INK_SOFT, predict, spearman

ARCHS = [("unet", "U-Net"), ("edsr", "EDSR"), ("swinir", "SwinIR-light")]
METHODS = [
    ("B0", "baseline: NLL, epoch by RMSE"),
    ("M1", "NLL, epoch by NLL + clipping"),
    ("M2", "β-NLL (β=0.5)"),
    ("M3", "M1 + weight decay"),
    ("M4", "ensemble of 5 (M1 seeds)"),
]
METHOD_COLORS = {"B0": "#9a9893", "M1": "#2a78d6", "M2": "#eb6834", "M3": "#8e6bd6", "M4": "#1baf7a"}


def checkpoints(arch, method):
    if method == "B0":
        return [f"checkpoints/{arch}_years.pt"]
    if method == "M1":
        return [f"checkpoints/methods/{arch}_m1.pt"]
    if method == "M2":
        return [f"checkpoints/methods/{arch}_m2.pt"]
    if method == "M3":
        return [f"checkpoints/methods/{arch}_m3.pt"]
    return [f"checkpoints/methods/{arch}_m1.pt"] + [f"checkpoints/methods/{arch}_m1_seed{k}.pt" for k in range(1, 5)]


def run(paths, arch, scale, ds, device):
    """Prediction and std (AOD) for one model, or the ensemble of several."""
    preds, variances = [], []
    for p in paths:
        model = build_model(arch, scale, use_consistency=False)
        model.load_state_dict(torch.load(p, map_location=device))
        pred, std, target, mask = predict(model.to(device).eval(), ds, device)
        preds.append(pred)
        variances.append(std ** 2)
    preds = torch.stack(preds)
    # total variance = average "data" uncertainty + disagreement between the models ("model" uncertainty)
    var = torch.stack(variances).mean(0) + preds.var(0, unbiased=False)
    return preds.mean(0), var.sqrt(), target, mask


def parse_log(path):
    """(epochs, val RMSE, val coverage90) from a training log."""
    text = open(path, encoding="utf-8", errors="ignore").read()
    epochs = [int(x) for x in re.findall(r"^epoch\s+(\d+)", text, flags=re.M)]
    rmse = [float(x) for x in re.findall(r"^epoch.*?RMSE ([0-9.]+) \(bic", text, flags=re.M)]
    cov = [float(x) for x in re.findall(r"coverage90 ([0-9.]+)", text)][:len(epochs)]
    return epochs, rmse, cov


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    _, val, test, scale = make_year_datasets()
    groups = hard_groups(test)

    results = {}
    for arch, arch_label in ARCHS:
        for method, method_label in METHODS:
            paths = checkpoints(arch, method)
            if not all(os.path.exists(p) for p in paths):
                continue
            # 2022: calibrate the guaranteed range (lambda)
            pred, std, target, mask = run(paths, arch, scale, val, device)
            lam, _ = calibrate(patch_risk((pred - target).abs(), std, mask, LAMBDAS), LAMBDAS)
            # 2023: score
            pred, std, target, mask = run(paths, arch, scale, test, device)
            err = (pred - target).abs()
            valid = mask.bool()
            r = dict(arch=arch, arch_label=arch_label, method=method, method_label=method_label,
                     n_models=len(paths),
                     rmse=metrics.rmse(pred, target, mask),
                     cov90=metrics.coverage90(pred, std, target, mask),
                     curve=[metrics.coverage(pred, std, target, mask, l) for l in metrics.CALIBRATION_LEVELS],
                     cal_err=metrics.calibration_error(pred, std, target, mask),
                     nll=metrics.gaussian_nll(pred, std, target, mask),
                     v_rmse=metrics.v_rmse(pred, std, target, mask),
                     rho=spearman(std[valid], err[valid]),
                     width_raw=metrics.sharpness(std, mask),
                     lam=lam,
                     miss=patch_risk(err, std, mask, torch.tensor([lam]))[:, 0].mean().item(),
                     width=(2 * lam * std)[valid].mean().item())
            for g, gm in groups.items():
                gv = gm.bool()
                r[f"miss_{g.split()[0]}"] = (err[gv] > lam * std[gv]).float().mean().item()
            results[f"{arch}_{method}"] = r
            print(f"{arch_label:<13} {method}  RMSE {r['rmse']:.4f}  coverage90 {r['cov90']:.3f}  "
                  f"cal_err {r['cal_err']:.3f}  NLL {r['nll']:.3f}  rho {r['rho']:.2f}  lambda {lam:.2f}  "
                  f"guaranteed: miss {r['miss']:.1%}  width {r['width']:.4f}  "
                  f"cloud edges {r['miss_cloud']:.1%}  high-AOD {r['miss_high-AOD']:.1%}")

    os.makedirs("runs/methods", exist_ok=True)
    with open("runs/methods/compare.json", "w") as f:
        json.dump(results, f, indent=2)

    # Markdown table (pasted into docs/rq3_methods_comparison.md)
    rows = ["| Model | Method | RMSE ↓ | Raw coverage90 (0.90) | Cal. error ↓ | NLL ↓ | Rank corr σ̂ vs error ↑ | λ̂ (honest = 1.645) "
            "| Guaranteed width ↓ | Misses: cloud edges ↓ | Misses: high AOD ↓ |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results.values():
        rows.append(f"| {r['arch_label']} | {r['method']}: {r['method_label']} | {r['rmse']:.4f} | {r['cov90']:.3f} | "
                    f"{r['cal_err']:.3f} | {r['nll']:.3f} | {r['rho']:.2f} | {r['lam']:.2f} | {r['width']:.4f} | "
                    f"{r['miss_cloud']:.1%} | {r['miss_high-AOD']:.1%} |")
    with open("runs/methods/compare_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print("\n" + "\n".join(rows))

    def style(ax):
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK_SOFT)

    # Figure 1: calibration curves per model, one line per method
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    for ax, (arch, arch_label) in zip(axes, ARCHS):
        ax.plot([0, 1], [0, 1], color=INK_SOFT, ls="--", lw=1)
        for method, method_label in METHODS:
            r = results.get(f"{arch}_{method}")
            if r:
                ax.plot(metrics.CALIBRATION_LEVELS, r["curve"], color=METHOD_COLORS[method], lw=2, marker="o",
                        markersize=4, label=f"{method}: {method_label} ({r['cov90']:.2f})")
        ax.set_title(f"{arch_label}: calibration on 2023", color=INK, fontsize=11)
        ax.set_xlabel("promised coverage", color=INK_SOFT)
        ax.set_ylabel("actual coverage", color=INK_SOFT)
        ax.legend(frameon=False, fontsize=7.5, title="method (coverage at 90%)", title_fontsize=8)
        style(ax)
    fig.savefig("docs/diagrams/rq3_methods_calibration.png", dpi=120, facecolor="#fcfcfb")

    # Figure 2: summary bars per method
    keys = list(results)
    labels = [f"{results[k]['arch_label']} {results[k]['method']}" for k in keys]
    colors = [METHOD_COLORS[results[k]["method"]] for k in keys]
    panels = [("cov90", "raw coverage at 90%  (target 0.90)", 0.90),
              ("lam", "λ̂ needed for the guarantee  (honest = 1.645)", 1.645),
              ("miss_cloud", "misses at cloud edges (guaranteed range)", 0.10),
              ("rmse", "RMSE on 2023 (lower = better)", None)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 0.32 * len(keys) + 1.6), layout="constrained", sharey=True)
    y = np.arange(len(keys))
    for ax, (key, title, ref) in zip(axes, panels):
        vals = [results[k][key] for k in keys]
        ax.barh(y, vals, color=colors)
        if ref is not None:
            ax.axvline(ref, color=INK_SOFT, ls="--", lw=1.2)
        lo = min(vals) * 0.9 if key in ("cov90", "rmse") else 0
        ax.set_xlim(lo, max(vals + ([ref] if ref else [])) * 1.08)
        ax.set_title(title, color=INK, fontsize=10)
        style(ax)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.savefig("docs/diagrams/rq3_methods_summary.png", dpi=120, facecolor="#fcfcfb")

    # Figure 3: why B0 is overconfident: 2022 coverage keeps falling while RMSE has stopped improving
    fig, axes = plt.subplots(2, 3, figsize=(16, 6.4), layout="constrained", sharex="col")
    for c, (arch, arch_label) in enumerate(ARCHS):
        for method, path in [("B0", f"runs/years_{arch}.log"), ("M1", f"runs/methods/{arch}_m1.log"),
                             ("M2", f"runs/methods/{arch}_m2.log")]:
            if not os.path.exists(path):
                continue
            ep, rmse, cov = parse_log(path)
            n = min(len(ep), len(rmse), len(cov))
            ep, rmse, cov = ep[:n], rmse[:n], cov[:n]
            keep = [i for i in range(n) if rmse[i] < 0.05]  # drop the blown-up epochs (EDSR B0)
            label = dict(METHODS)[method]
            axes[0, c].plot([ep[i] for i in keep], [cov[i] for i in keep], color=METHOD_COLORS[method], lw=2,
                            label=f"{method}: {label}")
            axes[1, c].plot([ep[i] for i in keep], [rmse[i] for i in keep], color=METHOD_COLORS[method], lw=2)
        axes[0, c].axhline(0.9, color=INK_SOFT, ls="--", lw=1)
        axes[0, c].set_title(f"{arch_label}: 2022 coverage at 90% vs epoch", color=INK, fontsize=10)
        axes[1, c].set_title(f"{arch_label}: 2022 RMSE vs epoch", color=INK, fontsize=10)
        axes[1, c].set_xlabel("epoch", color=INK_SOFT)
        axes[0, c].legend(frameon=False, fontsize=7.5)
        for ax in axes[:, c]:
            style(ax)
    fig.savefig("docs/diagrams/rq3_overconfidence_epochs.png", dpi=120, facecolor="#fcfcfb")
    print("\nsaved runs/methods/compare.json, runs/methods/compare_table.md, docs/diagrams/rq3_methods_*.png, "
          "docs/diagrams/rq3_overconfidence_epochs.png")


if __name__ == "__main__":
    main()
