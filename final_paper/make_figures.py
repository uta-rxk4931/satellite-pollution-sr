"""Publication-style single-column (3.5 in) figures as vector PDFs in final_paper/. Run from the repo root."""
import sys

sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.ticker import PercentFormatter

from src import metrics
from src.dataset import make_year_datasets
from src.models.sr_net import build_model
from src.rq3_guaranteed_range import LAMBDAS, WIDTHS, calibrate, hard_groups, patch_risk
from src.rq3_uncertainty_check import COLORS, MODELS, error_vs_sigma, predict
from src.rq3_uncertainty_maps import pick_patches

# IEEE look: Times-like serif text, LaTeX-style math, 8 pt labels, thin lines, fonts embedded in the PDF
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "savefig.dpi": 400,
})
W = 3.5  # IEEE column width (inches): include with width=\columnwidth so nothing is rescaled
CKPT = r"C:\Users\rgsei\Downloads\checkpoints"
INK_SOFT = "#52514e"

device = "cuda" if torch.cuda.is_available() else "cpu"
_, val, test, scale = make_year_datasets()
groups = hard_groups(test)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def panel_label(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontweight="bold", va="bottom", ha="right")


runs = {}
for arch, label in MODELS:
    model = build_model(arch, scale, use_consistency=False)
    model.load_state_dict(torch.load(rf"{CKPT}\{arch}_years.pt", map_location=device))
    model = model.to(device).eval()
    pv, sv, tv, mv = predict(model, val, device)
    pt, st, tt, mt = predict(model, test, device)
    ev, et = (pv - tv).abs(), (pt - tt).abs()
    lam, _ = calibrate(patch_risk(ev, sv, mv, LAMBDAS), LAMBDAS)
    c, _ = calibrate(patch_risk(ev, torch.ones_like(sv), mv, WIDTHS), WIDTHS)
    runs[arch] = dict(label=label, pred=pt, std=st, target=tt, mask=mt, err=et, lam=lam, c=c)
    print(label, "lambda", round(lam, 3), "c", round(c, 4))

# ---------- Fig A: examples (SwinIR-light), 3 rows x 5 panels, one colorbar per quantity ----------
r = runs["swinir"]
lr, lr_mask = test.to_aod(test.lr), test.lr_mask
blank = lambda x, mk: np.where(mk.numpy() > 0, x.numpy(), np.nan)
sigma_cmap = ListedColormap(plt.cm.Blues(np.linspace(0.15, 1, 256)))
rows = pick_patches(test)
titles = ["4 km input", r"prediction $\hat{y}$", r"truth $y$", r"uncertainty $\hat{\sigma}$", r"error $|y-\hat{y}|$"]
fig, axes = plt.subplots(3, 5, figsize=(W, 2.2), layout="constrained")
fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.01)
for k, (i, name) in enumerate(rows):
    mk = r["mask"][i, 0]
    valid = mk.bool()
    y, yh, s, e = r["target"][i, 0], r["pred"][i, 0], r["std"][i, 0], r["err"][i, 0]
    vmin, vmax = y[valid].min().item(), y[valid].max().item()
    outside = (e > r["lam"] * s) & valid
    panels = [(blank(lr[i, 0], lr_mask[i, 0]), "viridis", vmin, vmax),
              (blank(yh, mk), "viridis", vmin, vmax),
              (blank(y, mk), "viridis", vmin, vmax),
              (blank(s, mk), sigma_cmap, 0, torch.quantile(s[valid], 0.99).item()),
              (blank(e, mk), "magma", 0, torch.quantile(e[valid], 0.99).item())]
    ims = []
    for j, (img, cmap, lo, hi) in enumerate(panels):
        ax = axes[k, j]
        ims.append(ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest"))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
        if k == 0:
            ax.set_title(titles[j], fontsize=7, pad=2)
    yy, xx = np.where(outside.numpy())
    axes[k, 4].scatter(xx, yy, s=0.35, color="#35e0d0", marker="s", linewidths=0, rasterized=True)
    axes[k, 0].set_ylabel(f"{name}\n{outside[valid].float().mean():.0%} outside", fontsize=7, labelpad=2)
fig.savefig("final_paper/fig_examples_swinir.pdf")
fig.savefig("final_paper/fig_examples_swinir.png", dpi=300)
plt.close(fig)

# ---------- Fig B: RQ3(a) calibration curve + error vs sigma ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 1.65), layout="constrained")
a1.plot([0, 1], [0, 1], color=INK_SOFT, ls=(0, (3, 2)), lw=0.6)
hi = 0
for (arch, label), col in zip(MODELS, COLORS):
    q = runs[arch]
    curve = [metrics.coverage(q["pred"], q["std"], q["target"], q["mask"], p) for p in metrics.CALIBRATION_LEVELS]
    a1.plot(metrics.CALIBRATION_LEVELS, curve, color=col, marker="o", ms=2, label=label)
    v = q["mask"].bool()
    sg, rms = error_vs_sigma(q["err"][v], q["std"][v])
    a2.plot(np.array(sg) * 1e3, np.array(rms) * 1e3, color=col, marker="o", ms=2, label=label)
    hi = max(hi, max(sg), max(rms))
a2.plot([0, hi * 1e3], [0, hi * 1e3], color=INK_SOFT, ls=(0, (3, 2)), lw=0.6)
a1.set(xlabel="nominal coverage $p$", ylabel="observed coverage $C_p$", xlim=(0, 1), ylim=(0, 1))
a1.set_xticks([0, 0.5, 1]); a1.set_yticks([0, 0.5, 1])
a2.set(xlabel=r"predicted $\hat{\sigma}$ ($\times10^{-3}$)", ylabel=r"RMS error ($\times10^{-3}$)",
       xlim=(0, None), ylim=(0, None))
a1.legend(frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.1)
for ax, s in ((a1, "(a)"), (a2, "(b)")):
    clean(ax)
    panel_label(ax, s)
fig.savefig("final_paper/fig_rq3a_calibration.pdf")
fig.savefig("final_paper/fig_rq3a_calibration.png", dpi=300)
plt.close(fig)

# ---------- Fig C: RQ3(b) misses by pixel group, sigma range vs constant range ----------
names = list(groups)
short = ["clear patches", "cloud edges", "high AOD (top 10%)"]
fig, ax = plt.subplots(figsize=(W, 1.7), layout="constrained")
for g, gname in enumerate(names):
    gm = groups[gname].bool()
    for k, ((arch, label), col) in enumerate(zip(MODELS, COLORS)):
        q = runs[arch]
        x = g + (k - 1) * 0.2
        m_sig = (q["err"][gm] > q["lam"] * q["std"][gm]).float().mean().item()
        m_c = (q["err"][gm] > q["c"]).float().mean().item()
        ax.annotate("", xy=(x, m_sig), xytext=(x, m_c),
                    arrowprops=dict(arrowstyle="-|>", color="#bdbab3", lw=0.6, mutation_scale=5,
                                    shrinkA=2.5, shrinkB=2.5))
        ax.scatter([x], [m_c], s=9, facecolor="white", edgecolor="#8f8d87", lw=0.7, zorder=3,
                   label=r"constant $\hat{y}\pm c$" if g == 0 and k == 0 else None)
        ax.scatter([x], [m_sig], s=11, color=col, zorder=4, label=label if g == 0 else None)
ax.axhline(0.10, color=INK_SOFT, ls=(0, (1, 1.5)), lw=0.6)
ax.text(0.0, 0.103, r"target $\alpha$ = 10%", fontsize=6.5, color=INK_SOFT, ha="center", va="bottom")
ax.set_xticks(range(3), short)
ax.tick_params(axis="x", length=0)
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_ylabel("miss rate on 2023")
ax.set_ylim(0, 0.235)
ax.set_xlim(-0.5, 2.5)
ax.legend(frameon=False, loc="upper left", ncol=2, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1)
clean(ax)
fig.savefig("final_paper/fig_rq3b_range.pdf")
fig.savefig("final_paper/fig_rq3b_range.png", dpi=300)
print("saved final_paper/*.pdf (+ .png previews)")
