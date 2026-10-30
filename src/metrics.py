"""
How we score the model. Everything here is in real AOD units,
and cloudy pixels (mask = 0) are left out of every score.

RQ1 - does SR beat bicubic?
  RMSE          average error (lower = better)
  PSNR          image quality in dB (higher = better)
  SSIM          how similar the structure looks, 0 to 1 (higher = better)

RQ2 - does the average-back rule help without just blurring?
  consistency   does the sharp map average back to the input? (~0 with the rule on)
  detail_ratio  how much fine detail vs. the real map (~1 = right, <1 = too smooth, >1 = made-up detail)

RQ3 - is the confidence map honest and useful?
  coverage      how often the true value is inside the model's X% range (should be ~X)
  calibration   average gap between promised and actual coverage (0 = perfect)
  mean_uncertainty  average predicted uncertainty (smaller = more useful, if still calibrated)
  v_rmse        how far sigma^2 is from the actual squared error, pixel by pixel (lower = better)
  sharpness     average width of the 90% range (narrower = more useful, if still calibrated)
"""
from statistics import NormalDist

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

# PSNR and SSIM need to know the range of the data. AOD is almost always 0 to 1.
AOD_RANGE = 1.0


def bicubic_upscale(lr, scale):
    """The simple baseline we have to beat."""
    return F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)


# ---------------- RQ1: accuracy ----------------

def rmse(pred, target, mask):
    squared_error = (pred - target) ** 2 * mask
    return torch.sqrt(squared_error.sum() / mask.sum()).item()


def psnr(pred, target, mask):
    mse = ((pred - target) ** 2 * mask).sum() / mask.sum()
    
    return (10 * torch.log10(AOD_RANGE ** 2 / mse)).item()


def ssim(pred, target, mask):
    # SSIM can't skip pixels, so copy the true values into the cloudy spots.
    # Those pixels then count as "perfect", so the score is slightly optimistic
    # on cloudy patches -- mention this in the write-up.
    pred = torch.where(mask.bool(), pred, target)
    scores = []
    for i in range(len(pred)):
        p = pred[i, 0].cpu().numpy()
        t = target[i, 0].cpu().numpy()
        scores.append(structural_similarity(p, t, data_range=AOD_RANGE))
    return float(np.mean(scores))


# ---------------- RQ2: consistency and detail ----------------

def consistency_error(pred, lr, lr_mask, scale):
    diff = (F.avg_pool2d(pred, scale) - lr) * lr_mask
    return torch.sqrt((diff ** 2).sum() / lr_mask.sum()).item()


def detail_amount(x, mask):
    """
    How much the map changes from one pixel to the next (left-right and up-down).
    Only counts pairs where both pixels are real (not cloud).
    """
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    mask_x = mask[..., :, 1:] * mask[..., :, :-1]
    mask_y = mask[..., 1:, :] * mask[..., :-1, :]
    return (dx ** 2 * mask_x).sum() + (dy ** 2 * mask_y).sum()


def detail_ratio(pred, target, mask):
    """
    Fine detail in the prediction divided by fine detail in the real map.
      ~1.0  realistic amount of detail
      < 1   too smooth (e.g. bicubic)
      > 1   noisier than reality (made-up detail)
    """
    return torch.sqrt(detail_amount(pred, mask) / detail_amount(target, mask)).item()


# ---------------- RQ3: is the uncertainty honest? ----------------

def coverage(pred, std, target, mask, level=0.9):
    """
    How often the true value falls inside the model's `level` range.
    e.g. level=0.9: for a normal distribution 90% of values are within
    1.645 standard deviations, so we check |error| <= 1.645 * std.
    """
    z = NormalDist().inv_cdf((1 + level) / 2)   # 0.9 -> 1.645, 0.8 -> 1.282, ...
    inside = ((target - pred).abs() <= z * std).float()
    return ((inside * mask).sum() / mask.sum()).item()


def coverage90(pred, std, target, mask):
    return coverage(pred, std, target, mask, level=0.9)


CALIBRATION_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def calibration_error(pred, std, target, mask):
    """
    Check coverage at 10 confidence levels and average how far off each one is.
    e.g. if the 80% range only catches 70% of true values, that level is off by 0.10.
    0 = perfectly calibrated.
    """
    gaps = [abs(coverage(pred, std, target, mask, level) - level) for level in CALIBRATION_LEVELS]
    return float(np.mean(gaps))


def mean_uncertainty(std, mask):
    """
    Average predicted standard deviation (in AOD units).
    A model can get perfect coverage by saying "very unsure" everywhere,
    so we also want this to be small.
    """
    return ((std * mask).sum() / mask.sum()).item()


def v_rmse(pred, std, target, mask):
    """
    Variance RMSE (Zheng et al. 2026): how far the predicted variance sigma^2 is from the
    actual squared error, pixel by pixel. The loss trains sigma^2 to estimate the expected
    squared error, so smaller = sigma tracks the real error better.
    """
    gap = (std ** 2 - (pred - target) ** 2) ** 2
    return torch.sqrt((gap * mask).sum() / mask.sum()).item()


def sharpness(std, mask, level=0.9):
    """Average width of the `level` range (2 * z * sigma), in AOD units. Narrower = more useful."""
    z = NormalDist().inv_cdf((1 + level) / 2)
    return ((2 * z * std * mask).sum() / mask.sum()).item()
