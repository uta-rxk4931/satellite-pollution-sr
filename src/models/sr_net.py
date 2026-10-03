"""
The shared "shell" around every SR model (EDSR, U-Net, SwinIR-light).

Every body takes (lr, lr_mask) and returns (detail, log_var) at high res.
The shell then does the same thing for all of them, so the comparison is fair:

    sharp = bicubic(lr) + detail          # the body only learns the missing detail
    sharp = make_consistent(sharp, lr)    # optional average-back rule (RQ2)

Usage:
    model = build_model("edsr")   # or "unet", "swinir"
    sharp, log_var = model(lr, lr_mask)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.edsr import EDSR
from src.models.swinir import SwinIRLight
from src.models.unet import UNet

MODELS = {"edsr": EDSR, "unet": UNet, "swinir": SwinIRLight}


class SRModel(nn.Module):
    def __init__(self, body, scale=4, use_consistency=False):
        super().__init__()
        self.body = body
        self.scale = scale
        self.use_consistency = use_consistency

    def forward(self, lr, lr_mask):
        detail, log_var = self.body(lr, lr_mask)

        # Start from a plain bicubic upscale, so the network only has to learn
        # the missing detail on top of it (easier than learning everything)
        bicubic = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
        sharp = bicubic + detail

        if self.use_consistency:
            sharp = make_consistent(sharp, lr, lr_mask, self.scale)

        # log(variance) instead of variance, so it can never be negative.
        # Clamp it so exp() can't blow up.
        return sharp, log_var.clamp(-10, 10)


def build_model(name="edsr", scale=4, use_consistency=False):
    body = MODELS[name](scale=scale)
    return SRModel(body, scale, use_consistency)


def make_consistent(sharp, lr, lr_mask, scale):
    """
    Average-back rule: each scale x scale block of the sharp map should
    average to the low-res pixel it came from.

    We work out how far off each block's average is and shift the whole block
    by that amount. After this, averaging the sharp map gives back lr exactly.
    """
    block_avg = F.avg_pool2d(sharp, scale)            # average of each block
    error = (lr - block_avg) * lr_mask                # skip cloudy low-res pixels
    error_big = F.interpolate(error, scale_factor=scale, mode="nearest")  # copy to every pixel in the block
    return sharp + error_big


def uncertainty_loss(sharp, log_var, target, mask):
    """
    Gaussian negative log-likelihood (Kendall & Gal, 2017).

    - If the model is wrong at a pixel, it can "admit it" by predicting
      a high variance there, which shrinks the first term.
    - The second term (log_var) stops it from saying "unsure" everywhere.

    mask = 1 where the real high-res pixel exists, 0 where it's cloud,
    so cloudy pixels don't count.
    """
    loss = 0.5 * (torch.exp(-log_var) * (target - sharp) ** 2 + log_var)
    return (loss * mask).sum() / mask.sum()

