"""
Sprint 2 check: Can the Model learn (copy) ONE real DFW patch?

If the training is correct, the model should be able to fit a single patch almost
perfectly and beat bicubic by a alot. No confidence map and no average-back rule yet.

Why do this? "Overfitting one example" is a classic sanity check in deep learning.
A model that can't even memorize ONE example has a bug somewhere (wrong loss,
wrong shapes, learning rate too high, data not lined up, ...). If it CAN memorize it,
the training code works and we can move on to training on the full dataset.

Words used below:
    LR / lr      = low-resolution input map (4 km pixels). Careful: "lr" is ALSO short
                   for "learning rate" in args.lr. They are different things!
    HR / hr      = high-resolution target map (1 km pixels), what we want to predict
    mask         = 1 where a pixel has real data, 0 where it's missing (e.g. clouds)
    AOD          = Aerosol Optical Depth, the pollution value stored in each pixel
    bicubic      = a standard "smooth stretch" way of enlarging an image; the baseline to beat
    scale        = how many times bigger the HR map is than the LR map (here 4, since 4 km -> 1 km)

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
    # A flat, boring patch would be too easy (bicubic would already do well),
    # so we look for the most "interesting" patch: the one whose values vary the most.
    best, best_std = None, -1.0
    for i in range(len(dataset)):
        # Each item in the dataset is (lr, lr_mask, hr, hr_mask). We only need the HR parts.
        _, _, hr, hr_mask = dataset[i]

        # hr_mask.mean() = fraction of pixels with real data. Skip patches with
        # more than 5% missing (cloudy) pixels.
        if hr_mask.mean() < min_valid:
            continue

        # hr[hr_mask.bool()] keeps only the valid pixels, then .std() measures
        # how spread out their values are. .item() turns a 1-number tensor into a Python float.
        std = hr[hr_mask.bool()].std().item()
        if std > best_std:
            best, best_std = i, std
    return best


def score(sharp, lr, hr, hr_mask, dataset, scale):
    """RMSE / PSNR / SSIM / detail for the model and for bicubic, in AOD units."""
    # The model trains on NORMALIZED numbers (roughly mean 0, std 1) because neural
    # nets learn better that way. to_aod() undoes that so scores are in real AOD units.
    pred = dataset.to_aod(sharp)                                    # model's output
    bicubic = dataset.to_aod(metrics.bicubic_upscale(lr, scale))   # baseline output
    target = dataset.to_aod(hr)                                     # the true answer

    # Score both the model and bicubic the same way so we can compare them.
    #   rmse   - average error size (lower is better)
    #   psnr   - signal-to-noise in decibels (higher is better)
    #   ssim   - how similar the structure/shapes look, 1.0 = identical (higher is better)
    #   detail - how much fine detail vs the real map, ~1.0 is ideal
    # Every metric takes hr_mask so cloudy (missing) pixels are ignored.
    s = {}
    for name, x in [("model", pred), ("bicubic", bicubic)]:
        s[name] = {
            "rmse": metrics.rmse(x, target, hr_mask),
            "psnr": metrics.psnr(x, target, hr_mask),
            "ssim": metrics.ssim(x, target, hr_mask),
            "detail": metrics.detail_ratio(x, target, hr_mask),
        }
    return s, pred, bicubic, target


def save_figure(path, idx, model_name, s, lr, lr_mask, bicubic, pred, target, hr_mask, dataset):
    """
    Top row: the maps (one shared AOD colour scale).
    Bottom row: bicubic error vs model error (one shared error scale).
    Missing pixels are white.
    """
    # matplotlib draws NaN ("not a number") pixels as blank/white, so we use NaN
    # to hide the missing pixels.
    nan = torch.tensor(float("nan"))

    # blank(x, m): wherever the mask is 0, replace the pixel with NaN.
    # Tensors have shape (batch, channel, height, width); [0, 0] grabs the single
    # 2-D image (first batch item, first channel) and .numpy() converts it for matplotlib.
    blank = lambda x, m: torch.where(m.bool(), x, nan)[0, 0].numpy()

    # Absolute error per pixel: how far off each method is from the truth
    bic_err = (bicubic - target).abs()
    model_err = (pred - target).abs()

    # Use the same colour range for all maps in the top row, so the same colour
    # means the same AOD value in every panel (otherwise comparing them is misleading).
    valid = target[hr_mask.bool()]
    vmin, vmax = valid.min().item(), valid.max().item()

    # 99th percentile so one bad pixel doesn't wash out the rest of the error map
    err_max = torch.quantile(bic_err[hr_mask.bool()], 0.99).item()

    # subplot_mosaic lays out panels by name. Each string is one panel; "." is an empty spot.
    fig, axes = plt.subplot_mosaic(
        [["lr", "bicubic", "model", "target"],
         [".", "bicubic_err", "model_err", "err_bar"]],
        figsize=(13, 6.6), layout="constrained",
    )
    # Tiny RMSEs would round to 0.0000, so switch to 1.2e-05 style for those
    fmt = lambda x: f"{x:.4f}" if x >= 1e-3 else f"{x:.1e}"

    # Top row: (panel name, title, image)
    maps = [
        ("lr", "low-res input (4 km)", blank(dataset.to_aod(lr), lr_mask)),
        ("bicubic", f"bicubic\nRMSE {fmt(s['bicubic']['rmse'])}", blank(bicubic, hr_mask)),
        ("model", f"{model_name}\nRMSE {fmt(s['model']['rmse'])}", blank(pred, hr_mask)),
        ("target", "1 km target", blank(target, hr_mask)),
    ]
    for key, title, img in maps:
        im = axes[key].imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[key].set_title(title)
    # One colour bar shared by all four top-row maps
    fig.colorbar(im, ax=[axes[k] for k, _, _ in maps], label="AOD", shrink=0.9)

    # Bottom row: error maps. Darker = smaller error. A good model's error map
    # should look much darker than bicubic's.
    errors = [
        ("bicubic_err", "bicubic error", blank(bic_err, hr_mask)),
        ("model_err", f"{model_name} error", blank(model_err, hr_mask)),
    ]
    for key, title, img in errors:
        im = axes[key].imshow(img, cmap="magma", vmin=0, vmax=err_max)
        axes[key].set_title(title)
    # Error colour bar goes in the empty bottom-right cell, so it doesn't push the top row around
    axes["err_bar"].axis("off")
    fig.colorbar(im, cax=axes["err_bar"].inset_axes([0.02, 0, 0.06, 1]), label="|error| (AOD)")

    # Pixel-number tick marks aren't useful on maps, so remove them
    for ax in axes.values():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Sprint 2: {model_name} trained on one patch (patch {idx}) vs bicubic", fontsize=13)
    os.makedirs(os.path.dirname(path), exist_ok=True)  # create the folder if it doesn't exist
    fig.savefig(path, dpi=120)
    plt.close(fig)  # free the memory used by the figure


def main():

    # Command-line options, e.g. `--steps 500`. Each has a default if you don't pass it.
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sprint2_patches.npz")
    parser.add_argument("--patch", type=int, default=None, help="patch index (default: most detailed)")
    parser.add_argument("--model", default="edsr", choices=list(MODELS))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (1e-3 blows up near the end)")
    parser.add_argument("--figure", default="docs/diagrams/sprint2_overfit.png")
    args = parser.parse_args()

    torch.manual_seed(0)  # same results every run

    # Use the GPU ("cuda") if one is available, since it's much faster; otherwise the CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. One real patch
    dataset, scale = load_real_patches(args.data)
    idx = args.patch if args.patch is not None else pick_patch(dataset)

    # Models expect a BATCH of examples with shape (batch, channel, height, width).
    # t[None] adds a batch dimension of size 1 at the front, e.g. (1, H, W) -> (1, 1, H, W).
    # .to(device) moves the data onto the GPU/CPU, which must match where the model lives.
    lr, lr_mask, hr, hr_mask = (t[None].to(device) for t in dataset[idx])  # batch of 1
    print(f"{len(dataset)} patches in file, using patch {idx} "
          f"({hr_mask.mean().item():.0%} valid pixels)")

    # 2. Model (no average-back rule; the uncertainty output is ignored for now)
    model = build_model(args.model, scale, use_consistency=False).to(device)

    # The optimizer is what actually changes the model's weights after each step.
    # Adam is a popular default that adapts the step size for each weight automatically.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Count the learnable numbers (weights) in the model, just for info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model}, {num_params:,} parameters")

    # 3. Train on the same patch over and over
    for step in range(1, args.steps + 1):
        model.train()  # put the model in training mode

        # Forward pass: low-res map in, predicted high-res ("sharp") map out.
        # The model also returns an uncertainty map, which we throw away (_) for this test.
        sharp, _ = model(lr, lr_mask)

        # Masked MSE (mean squared error): square each pixel's error, zero out the
        # missing pixels by multiplying by the mask, then average over the valid pixels only.
        loss = ((sharp - hr) ** 2 * hr_mask).sum() / hr_mask.sum()  # masked MSE

        # The three standard PyTorch training lines:
        optimizer.zero_grad()  # clear the gradients left over from the last step
        loss.backward()        # backpropagation: work out how each weight affects the loss
        optimizer.step()       # nudge every weight a little in the direction that lowers the loss

        # Every 100 steps (and on step 1), print progress so we can watch it learn
        if step % 100 == 0 or step == 1:
            model.eval()  # evaluation mode (turns off training-only behaviour)
            with torch.no_grad():  # no gradients needed just to look; saves memory and time
                sharp, _ = model(lr, lr_mask)
            # .cpu() moves results back to the CPU for scoring and printing.
            # `s, *_ =` keeps the first return value (the scores) and ignores the rest.
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

    # PASS if the model's error is less than 10% of bicubic's error.
    # Since it has seen this exact patch 2000 times, it should easily get there.
    ratio = s["model"]["rmse"] / s["bicubic"]["rmse"]
    verdict = "PASS" if ratio < 0.1 else "FAIL"
    print(f"\n{verdict}: model RMSE is {ratio:.1%} of bicubic RMSE "
          f"({s['model']['psnr'] - s['bicubic']['psnr']:+.1f} dB PSNR)")

    names = {"edsr": "EDSR", "unet": "U-Net", "swinir": "SwinIR-light"}
    save_figure(args.figure, idx, names[args.model], s, lr.cpu(), lr_mask.cpu(), bicubic, pred, target, hr_mask.cpu(), dataset)
    print(f"saved figure to {args.figure}")


# Only run main() when this file is run directly (not when it's imported by another file)
if __name__ == "__main__":
    main()
