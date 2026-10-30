"""

Training the SR model and comparing it to a bicubic upscaling

SR = super-resolution: turning a blurry low-res pollution map (4 km pixels) into a
sharper high-res one (1 km pixels). Bicubic upscaling is the simple "smooth stretch"
method; our model has to beat it to be worth anything.

The big picture:
    1. Load the data and split it into a TRAINING set (the model learns from it) and a
       VALIDATION set (held back, so we can check the model on maps it has never seen).
    2. Build the model and the optimizer (the thing that updates the model's weights).
    3. Loop over the training data many times ("epochs"), nudging the weights each batch.
    4. Every few epochs, score the model on the validation set and print the results.
    5. Save the trained weights to a file.

Words used below:
    lr / hr      = low-res input map / high-res target map. Careful: args.lr is the
                   LEARNING RATE, which has nothing to do with the low-res map!
    mask         = 1 where a pixel has real data, 0 where it's missing (e.g. clouds)
    AOD          = Aerosol Optical Depth, the pollution value stored in each pixel
    batch        = a small group of patches the model processes together
    epoch        = one full pass through all of the training data

Commands to use:
    python -m src.train                       # EDSR, dummy data, 10 epochs
    python -m src.train --model unet          # or: edsr, unet, swinir
    python -m src.train --epochs 5            # fewer epochs
    python -m src.train --consistency         # turn on the average-back rule
    python -m src.train --data real --epochs 300 --eval_every 25   # real DFW patches
    python -m src.train --data years --epochs 100 --eval_every 5   # year split: 2018-21 / 2022 / 2023

With --data years, the epoch with the best 2022 (val) RMSE is kept, and the 2023 test
year is scored only once, at the end.

Uses the GPU automatically if PyTorch can see one (needs the CUDA version of torch).
"""
import argparse
import copy
import os

import torch
from torch.utils.data import DataLoader

from src import metrics
from src.dataset import make_dummy_datasets, make_real_datasets, make_year_datasets
from src.models.sr_net import MODELS, build_model, uncertainty_loss



def evaluate(model, loader, dataset, scale, device):
    """Run the model on the validation set and return the average scores."""
    model.eval()  # evaluation mode (turns off training-only behaviour)
    # Collect every batch first, then score the whole validation set at once
    # (averaging per-batch RMSEs would not give exactly the overall RMSE)
    preds, bicubics, targets, masks, lrs, lr_masks, stds = [], [], [], [], [], [], []

    # no_grad(): we're only measuring, not learning, so PyTorch doesn't need to track
    # gradients. This saves a lot of memory and time.
    with torch.no_grad():
        for lr, lr_mask, hr, hr_mask in loader:
            lr, lr_mask, hr, hr_mask = lr.to(device), lr_mask.to(device), hr.to(device), hr_mask.to(device)

            # The model returns two maps:
            #   sharp   - its best guess at the high-res map
            #   log_var - how UNSURE it is at each pixel (log of the variance; see below)
            sharp, log_var = model(lr, lr_mask)

            # Convert back to real AOD units before scoring (and move to CPU to save GPU memory)
            preds.append(dataset.to_aod(sharp).cpu())
            bicubics.append(dataset.to_aod(metrics.bicubic_upscale(lr, scale)).cpu())
            targets.append(dataset.to_aod(hr).cpu())
            masks.append(hr_mask.cpu())
            lrs.append(dataset.to_aod(lr).cpu())
            lr_masks.append(lr_mask.cpu())

            # log_var -> standard deviation: std = sqrt(variance) = exp(0.5 * log_var).
            # Multiplying by dataset.std converts it from normalized units to AOD units.
            stds.append((torch.exp(0.5 * log_var) * dataset.std).cpu())  # predicted uncertainty in AOD units

    # torch.cat glues the per-batch lists into one big tensor each
    pred, bicubic, target, mask = torch.cat(preds), torch.cat(bicubics), torch.cat(targets), torch.cat(masks)
    lr, lr_mask, std = torch.cat(lrs), torch.cat(lr_masks), torch.cat(stds)

    # Scores are grouped by the project's three research questions (RQ1-RQ3)
    return {
        # RQ1: is the model more accurate than bicubic?
        #   rmse - average error size (lower is better)
        #   psnr - signal-to-noise in decibels (higher is better)
        #   ssim - how similar the structure/shapes look, 1.0 = identical (higher is better)
        "rmse": metrics.rmse(pred, target, mask),
        "rmse_bicubic": metrics.rmse(bicubic, target, mask),
        "psnr": metrics.psnr(pred, target, mask),
        "psnr_bicubic": metrics.psnr(bicubic, target, mask),
        "ssim": metrics.ssim(pred, target, mask),
        "ssim_bicubic": metrics.ssim(bicubic, target, mask),
        # RQ2: is the output physically believable?
        #   consistency - if you average each scale x scale block of the prediction, how far is it
        #                 from the original low-res pixel? (should be ~0)
        #   detail      - fine detail compared to the real map (~1.0 ideal,
        #                 <1 too smooth, >1 made-up noise)
        "consistency": metrics.consistency_error(pred, lr, lr_mask, scale),
        "detail": metrics.detail_ratio(pred, target, mask),
        "detail_bicubic": metrics.detail_ratio(bicubic, target, mask),
        # RQ3: is the model honest about how unsure it is?
        #   coverage90 - fraction of true values inside the model's 90% range (ideal 0.90)
        #   cal_err    - average gap between promised and actual coverage over many levels (0 = perfect)
        #   unc        - average predicted uncertainty (smaller is better, as long as coverage is right)
        "coverage90": metrics.coverage90(pred, std, target, mask),
        "cal_err": metrics.calibration_error(pred, std, target, mask),
        "unc": metrics.mean_uncertainty(std, mask),
    }


def print_scores(label, s):
    """Print the scores from evaluate(): model first, bicubic in brackets."""
    print(f"{label} | RMSE {s['rmse']:.4f} (bicubic {s['rmse_bicubic']:.4f}) | "
          f"PSNR {s['psnr']:.2f} (bicubic {s['psnr_bicubic']:.2f}) | "
          f"SSIM {s['ssim']:.3f} (bicubic {s['ssim_bicubic']:.3f})")
    print(f"{' ' * len(label)} | consistency {s['consistency']:.1e} | "
          f"detail {s['detail']:.2f} (bicubic {s['detail_bicubic']:.2f}) | "
          f"coverage90 {s['coverage90']:.2f} | cal_err {s['cal_err']:.3f} | unc {s['unc']:.4f}")


def main():
    # Command-line options, e.g. `--epochs 5`. Each has a default if you don't pass it.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="edsr", choices=list(MODELS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--consistency", action="store_true", help="use the average-back rule")
    parser.add_argument("--data", default="dummy", choices=["dummy", "real", "years"],
                        help="fake maps, the real patches in --data_path, or the year split in --split_dir")
    parser.add_argument("--data_path", default="data/sprint2_patches.npz")
    parser.add_argument("--split_dir", default="data/split_data")
    parser.add_argument("--eval_every", type=int, default=1, help="score the val set every N epochs")
    args = parser.parse_args()

    torch.manual_seed(0)  # same results every run

    # Use the GPU if there is one, otherwise the CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. Data: fake maps, the June 2023 patches (temporary split), or Track A's year split
    # The real data files say what their scale is, so they override --scale
    test_set = None  # only the year split has a test set
    if args.data == "years":
        train_set, val_set, test_set, args.scale = make_year_datasets(args.split_dir)
    elif args.data == "real":
        train_set, val_set, args.scale = make_real_datasets(args.data_path)
    else:
        train_set, val_set = make_dummy_datasets(args.scale)

    # A DataLoader hands out the data in batches.
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    print(f"train: {len(train_set)} patches, val: {len(val_set)} patches"
          + (f", test: {len(test_set)} patches" if test_set is not None else ""))

    # 2. Model and optimizer
    model = build_model(args.model, args.scale, args.consistency).to(device)

    # The optimizer changes the model's weights after each batch.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Count the learnable numbers (weights) in the model, just for info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model}, {num_params:,} parameters, consistency rule = {args.consistency}")

    # 3. Training loop
    # For the year split, remember the weights from the epoch with the lowest val RMSE
    best_rmse, best_epoch, best_weights = float("inf"), None, None
    for epoch in range(1, args.epochs + 1):
        model.train()  # put the model in training mode
        total_loss = 0
        for lr, lr_mask, hr, hr_mask in train_loader:
            # Move the batch to the same device as the model
            lr, lr_mask, hr, hr_mask = lr.to(device), lr_mask.to(device), hr.to(device), hr_mask.to(device)

            # Forward pass: predict the high-res map and the uncertainty map
            sharp, log_var = model(lr, lr_mask)

            # The loss rewards being accurate AND being honest about uncertainty:
            # the model can "admit" it's unsure at a pixel (high log_var) to reduce the
            # penalty for being wrong there, but it pays a small cost for saying "unsure".
            # Cloudy pixels (mask = 0) are ignored. See uncertainty_loss in sr_net.py.
            loss = uncertainty_loss(sharp, log_var, hr, hr_mask)

            # The three standard PyTorch training lines:
            optimizer.zero_grad()  # clear the gradients left over from the last batch
            loss.backward()        # backpropagation: work out how each weight affects the loss
            optimizer.step()       # nudge every weight a little in the direction that lowers the loss
            total_loss += loss.item()  # .item() turns the 1-number tensor into a Python float

        # 4. Check how we're doing on the validation set
        if epoch % args.eval_every != 0 and epoch != args.epochs:
            continue
        s = evaluate(model, val_loader, val_set, args.scale, device)

        # Note: this loss is the uncertainty loss, which can go negative. That's normal.
        print_scores(f"epoch {epoch:3d} | loss {total_loss / len(train_loader):.3f}", s)

        if test_set is not None and s["rmse"] < best_rmse:
            best_rmse, best_epoch = s["rmse"], epoch
            best_weights = copy.deepcopy(model.state_dict())

    # Year split: go back to the best epoch (picked on 2022 only), then score 2023 once
    if test_set is not None:
        model.load_state_dict(best_weights)
        print(f"best epoch on 2022 val: {best_epoch} (RMSE {best_rmse:.4f})")
        test_scores = evaluate(model, DataLoader(test_set, batch_size=64), test_set, args.scale, device)
        print_scores("TEST 2023", test_scores)

    # 5. Save the trained model
    # state_dict() is a dictionary of all the learned weights. The file name says which
    # model and settings were used, e.g. checkpoints/edsr_consistency_real.pt
    os.makedirs("checkpoints", exist_ok=True)
    suffix = {"real": "_real", "years": "_years"}.get(args.data, "")
    name = f"checkpoints/{args.model}{'_consistency' if args.consistency else ''}{suffix}.pt"
    torch.save(model.state_dict(), name)
    print(f"saved model to {name}")


if __name__ == "__main__":
    main()
