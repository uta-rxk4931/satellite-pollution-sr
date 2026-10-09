"""
Train the SR model and compare it to bicubic upscaling.

How to run:
    python -m src.train                       # EDSR, dummy data, 10 epochs
    python -m src.train --model unet          # or: edsr, unet, swinir
    python -m src.train --epochs 5            # fewer epochs
    python -m src.train --consistency         # turn on the average-back rule
    python -m src.train --data real --epochs 300 --eval_every 25   # real DFW patches

Uses the GPU automatically if PyTorch can see one (needs the CUDA version of torch).
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from src import metrics
from src.dataset import make_dummy_datasets, make_real_datasets
from src.models.sr_net import MODELS, build_model, uncertainty_loss


def evaluate(model, loader, dataset, scale, device):
    """Run the model on the validation set and return the average scores."""
    model.eval()
    # Collect every batch first, then score the whole validation set at once
    # (averaging per-batch RMSEs would not give exactly the overall RMSE)
    preds, bicubics, targets, masks, lrs, lr_masks, stds = [], [], [], [], [], [], []

    with torch.no_grad():
        for lr, lr_mask, hr, hr_mask in loader:
            lr, lr_mask, hr, hr_mask = lr.to(device), lr_mask.to(device), hr.to(device), hr_mask.to(device)
            sharp, log_var = model(lr, lr_mask)

            # Convert back to real AOD units before scoring (and move to CPU to save GPU memory)
            preds.append(dataset.to_aod(sharp).cpu())
            bicubics.append(dataset.to_aod(metrics.bicubic_upscale(lr, scale)).cpu())
            targets.append(dataset.to_aod(hr).cpu())
            masks.append(hr_mask.cpu())
            lrs.append(dataset.to_aod(lr).cpu())
            lr_masks.append(lr_mask.cpu())
            stds.append((torch.exp(0.5 * log_var) * dataset.std).cpu())  # predicted uncertainty in AOD units

    pred, bicubic, target, mask = torch.cat(preds), torch.cat(bicubics), torch.cat(targets), torch.cat(masks)
    lr, lr_mask, std = torch.cat(lrs), torch.cat(lr_masks), torch.cat(stds)

    return {
        # RQ1
        "rmse": metrics.rmse(pred, target, mask),
        "rmse_bicubic": metrics.rmse(bicubic, target, mask),
        "psnr": metrics.psnr(pred, target, mask),
        "psnr_bicubic": metrics.psnr(bicubic, target, mask),
        "ssim": metrics.ssim(pred, target, mask),
        "ssim_bicubic": metrics.ssim(bicubic, target, mask),
        # RQ2
        "consistency": metrics.consistency_error(pred, lr, lr_mask, scale),
        "detail": metrics.detail_ratio(pred, target, mask),
        "detail_bicubic": metrics.detail_ratio(bicubic, target, mask),
        # RQ3
        "coverage90": metrics.coverage90(pred, std, target, mask),
        "cal_err": metrics.calibration_error(pred, std, target, mask),
        "unc": metrics.mean_uncertainty(std, mask),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="edsr", choices=list(MODELS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--consistency", action="store_true", help="use the average-back rule")
    parser.add_argument("--data", default="dummy", choices=["dummy", "real"],
                        help="fake maps, or the real patches in --data_path")
    parser.add_argument("--data_path", default="data/sprint2_patches.npz")
    parser.add_argument("--eval_every", type=int, default=1, help="score the val set every N epochs")
    args = parser.parse_args()

    torch.manual_seed(0)  # same results every run

    # Use the GPU if there is one, otherwise the CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. Data: fake maps, or real patches (temporary date-order split until Track A's split by year)
    if args.data == "real":
        train_set, val_set, args.scale = make_real_datasets(args.data_path)
    else:
        train_set, val_set = make_dummy_datasets(args.scale)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    print(f"train: {len(train_set)} patches, val: {len(val_set)} patches")

    # 2. Model and optimizer
    model = build_model(args.model, args.scale, args.consistency).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model}, {num_params:,} parameters, consistency rule = {args.consistency}")

    # 3. Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for lr, lr_mask, hr, hr_mask in train_loader:
            # Move the batch to the same device as the model
            lr, lr_mask, hr, hr_mask = lr.to(device), lr_mask.to(device), hr.to(device), hr_mask.to(device)
            sharp, log_var = model(lr, lr_mask)
            loss = uncertainty_loss(sharp, log_var, hr, hr_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 4. Check how we're doing on the validation set
        if epoch % args.eval_every != 0 and epoch != args.epochs:
            continue
        s = evaluate(model, val_loader, val_set, args.scale, device)
        print(f"epoch {epoch:2d} | loss {total_loss / len(train_loader):.3f} | "
              f"RMSE {s['rmse']:.4f} (bicubic {s['rmse_bicubic']:.4f}) | "
              f"PSNR {s['psnr']:.2f} (bicubic {s['psnr_bicubic']:.2f}) | "
              f"SSIM {s['ssim']:.3f} (bicubic {s['ssim_bicubic']:.3f})")
        print(f"         | consistency {s['consistency']:.1e} | "
              f"detail {s['detail']:.2f} (bicubic {s['detail_bicubic']:.2f}) | "
              f"coverage90 {s['coverage90']:.2f} | cal_err {s['cal_err']:.3f} | unc {s['unc']:.4f}")

    # 5. Save the trained model
    os.makedirs("checkpoints", exist_ok=True)
    name = (f"checkpoints/{args.model}{'_consistency' if args.consistency else ''}"
            f"{'_real' if args.data == 'real' else ''}.pt")
    torch.save(model.state_dict(), name)
    print(f"saved model to {name}")


if __name__ == "__main__":
    main()
