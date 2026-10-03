"""
SwinIR-light body: a small wrapper around the OFFICIAL SwinIR code.

The model itself is in swinir_official.py, copied unchanged from
https://github.com/JingyunLiang/SwinIR (models/network_swinir.py, Apache-2.0 license).

Papers to cite:
  - J. Liang, J. Cao, G. Sun, K. Zhang, L. Van Gool, R. Timofte,
    "SwinIR: Image Restoration Using Swin Transformer," ICCV Workshops, 2021.
  - Z. Liu et al., "Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows," ICCV, 2021. (SwinIR is built on its code, MIT license)

This wrapper only does two things so SwinIR fits our shared shell (sr_net.py):
  1. glue the AOD map and cloud mask into a 2-channel input
  2. split SwinIR's 2-channel output into (detail, uncertainty)
"""
import torch
import torch.nn as nn

from src.models.swinir_official import SwinIR


class SwinIRLight(nn.Module):
    def __init__(self, scale=4, img_size=16):
        super().__init__()
        # The official "lightweight SR" settings from the SwinIR paper
        self.net = SwinIR(
            upscale=scale,
            in_chans=2,                     # AOD + cloud mask in, 2 channels out
            img_size=img_size,              # low-res patch size (64 / 4 = 16)
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffledirect",
        )

    def forward(self, lr, lr_mask):
        out = self.net(torch.cat([lr, lr_mask], dim=1))   # (B, 2, 64, 64)
        return out[:, 0:1], out[:, 1:2]                     # detail, uncertainty
