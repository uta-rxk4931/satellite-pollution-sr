"""
Sprint 2 check: Can the Model learn (copy) ONE real DFW patch?

If the training is correct, the model should be able to fit a single patch almost
perfectly and beat bicubic by a alot. No confidence map and no average-back rule yet. 

How to run:
    python -m src.overfit_one_patch                # EDSR, most detailed patch
    python -m src.overfit_one_patch --patch 43     # pick a patch yourself
    python -m src.overfit_one_patch --model unet   # or: edsr, unet, swinir
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import torch

from src import metrics
from src.dataset import load_real_patches
from src.models.sr_net import MODELS, build_model


def pick_patch(dataset, min_valid=0.95):
    """The patch with the most variation (std) among the nearly cloud-free ones."""
    best, best_std = None, -1.0
    for i in range(len(dataset)):
        _, _, hr, hr_mask = dataset[i]
        if hr_mask.mean() < min_valid:
            continue
        std = hr[hr_mask.bool()].std().item()
        if std > best_std:
            best, best_std = i, std
    return best


def score(sharp, lr, hr, hr_mask, dataset, scale):
    """RMSE / PSNR / SSIM / detail for the model and for bicubic, in AOD units."""
    pred = dataset.to_aod(sharp)
    bicubic = dataset.to_aod(metrics.bicubic_upscale(lr, scale))
    target = dataset.to_aod(hr)
    s = {}
    for name, x in [("model", pred), ("bicubic", bicubic)]:
        s[name] = {
            "rmse": metrics.rmse(x, target, hr_mask),
            "psnr": metrics.psnr(x, target, hr_mask),
            "ssim": metrics.ssim(x, target, hr_mask),
            "detail": metrics.detail_ratio(x, target, hr_mask),
        }
    return s, pred, bicubic, target


def save_figure(path, idx, s, lr, lr_mask, bicubic, pred, target, hr_mask, dataset):
    """
    Top row: the maps (one shared AOD colour scale).
    Bottom row: bicubic error vs model error (one shared error scale).
    Missing pixels are white.
    """
    nan = torch.tensor(float("nan"))
    blank = lambda x, m: torch.where(m.bool(), x, nan)[0, 0].numpy()
    bic_err = (bicubic - target).abs()
    model_err = (pred - target).abs()

    valid = target[hr_mask.bool()]
    vmin, vmax = valid.min().item(), valid.max().item()
    # 99th percentile so one bad pixel doesn't wash out the rest of the error map
    err_max = torch.quantile(bic_err[hr_mask.bool()], 0.99).item()

    fig, axes = plt.subplot_mosaic(
        [["lr", "bicubic", "model", "target"],
         [".", "bicubic_err", "model_err", "err_bar"]],
        figsize=(13, 6.6), layout="constrained",
    )
    # Tiny RMSEs would round to 0.0000, so switch to 1.2e-05 style for those
    fmt = lambda x: f"{x:.4f}" if x >= 1e-3 else f"{x:.1e}"
    maps = [
        ("lr", "low-res input (4 km)", blank(dataset.to_aod(lr), lr_mask)),
        ("bicubic", f"bicubic\nRMSE {fmt(s['bicubic']['rmse'])}", blank(bicubic, hr_mask)),
        ("model", f"model\nRMSE {fmt(s['model']['rmse'])}", blank(pred, hr_mask)),
        ("target", "1 km target", blank(target, hr_mask)),
    ]
    for key, title, img in maps:
        im = axes[key].imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[key].set_title(title)
    fig.colorbar(im, ax=[axes[k] for k, _, _ in maps], label="AOD", shrink=0.9)

    errors = [
        ("bicubic_err", "bicubic error", blank(bic_err, hr_mask)),
        ("model_err", "model error", blank(model_err, hr_mask)),
    ]
    for key, title, img in errors:
        im = axes[key].imshow(img, cmap="magma", vmin=0, vmax=err_max)
        axes[key].set_title(title)
    # Error colour bar goes in the empty bottom-right cell, so it doesn't push the top row around
    axes["err_bar"].axis("off")
    fig.colorbar(im, cax=axes["err_bar"].inset_axes([0.02, 0, 0.06, 1]), label="|error| (AOD)")

    for ax in axes.values():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Sprint 2: model trained on one patch (patch {idx}) vs bicubic", fontsize=13)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sprint2_patches.npz")
    parser.add_argument("--patch", type=int, default=None, help="patch index (default: most detailed)")
    parser.add_argument("--model", default="edsr", choices=list(MODELS))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--figure", default="docs/diagrams/sprint2_overfit.png")
    args = parser.parse_args()

    torch.manual_seed(0)  # same results every run
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. One real patch
    dataset, scale = load_real_patches(args.data)
    idx = args.patch if args.patch is not None else pick_patch(dataset)
    lr, lr_mask, hr, hr_mask = (t[None].to(device) for t in dataset[idx])  # batch of 1
    print(f"{len(dataset)} patches in file, using patch {idx} "
          f"({hr_mask.mean().item():.0%} valid pixels)")

    # 2. Model (no average-back rule; the uncertainty output is ignored for now)
    model = build_model(args.model, scale, use_consistency=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model}, {num_params:,} parameters")

    # 3. Train on the same patch over and over
    for step in range(1, args.steps + 1):
        model.train()
        sharp, _ = model(lr, lr_mask)
        loss = ((sharp - hr) ** 2 * hr_mask).sum() / hr_mask.sum()  # masked MSE

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                sharp, _ = model(lr, lr_mask)
            s, *_ = score(sharp.cpu(), lr.cpu(), hr.cpu(), hr_mask.cpu(), dataset, scale)
            m, b = s["model"], s["bicubic"]
            print(f"step {step:5d} | loss {loss.item():.2e} | "
                  f"RMSE {m['rmse']:.5f} (bicubic {b['rmse']:.5f}) | "
                  f"PSNR {m['psnr']:.2f} (bicubic {b['psnr']:.2f}) | "
                  f"SSIM {m['ssim']:.3f} (bicubic {b['ssim']:.3f}) | "
                  f"detail {m['detail']:.2f} (bicubic {b['detail']:.2f})")

    # 4. Final check: did it learn the patch?
    model.eval()
    with torch.no_grad():
        sharp, _ = model(lr, lr_mask)
    s, pred, bicubic, target = score(sharp.cpu(), lr.cpu(), hr.cpu(), hr_mask.cpu(), dataset, scale)
    ratio = s["model"]["rmse"] / s["bicubic"]["rmse"]
    verdict = "PASS" if ratio < 0.1 else "FAIL"
    print(f"\n{verdict}: model RMSE is {ratio:.1%} of bicubic RMSE "
          f"({s['model']['psnr'] - s['bicubic']['psnr']:+.1f} dB PSNR)")

    save_figure(args.figure, idx, s, lr.cpu(), lr_mask.cpu(), bicubic, pred, target, hr_mask.cpu(), dataset)
    print(f"saved figure to {args.figure}")


if __name__ == "__main__":
    main()
