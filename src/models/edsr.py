"""
EDSR body: a CNN that works at low resolution and upsamples at the end.

Matches the official EDSR-baseline (16 residual blocks, 64 channels, res_scale 1):
  - B. Lim, S. Son, H. Kim, S. Nah, K. M. Lee, "Enhanced Deep Residual Networks
    for Single Image Super-Resolution," CVPR Workshops, 2017.
  - Code: https://github.com/sanghyun-son/EDSR-PyTorch (MIT license),
    src/model/edsr.py and src/model/common.py

Changes for this project (on purpose):
  - 2 input channels (AOD + cloud mask) instead of 3 (RGB)
  - no MeanShift (that subtracts the RGB mean of photos; our data is normalized already)
  - the final conv is split into 2 heads: detail + uncertainty
  - the shell in sr_net.py adds bicubic, so this body only predicts the missing detail

    (2,16,16) -> conv -> 16 residual blocks -> conv (+ skip) -> 2 x [conv, PixelShuffle x2] -> (64,64,64)
              -> detail head (1,64,64) and uncertainty head (1,64,64)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Two conv layers plus a skip connection: output = input + res_scale * change.
    No batch norm, same as EDSR: batch norm changes the scale of the values,
    and for AOD the actual values matter.
    """

    def __init__(self, channels, res_scale=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # EDSR-baseline uses 1.0. The big EDSR (256 channels) uses 0.1 to keep training stable.
        self.res_scale = res_scale

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return x + self.res_scale * out


def make_upsampler(channels, scale):
    """
    Same as the official EDSR Upsampler:
      scale 2, 4, 8: repeat [conv -> PixelShuffle(2)] (each step doubles the size)
      scale 3:       one [conv -> PixelShuffle(3)]
    PixelShuffle turns extra channels into extra pixels.
    """
    layers = []
    if scale & (scale - 1) == 0:                 # scale is a power of 2
        for _ in range(int(math.log2(scale))):
            layers.append(nn.Conv2d(channels, 4 * channels, kernel_size=3, padding=1))
            layers.append(nn.PixelShuffle(2))
    elif scale == 3:
        layers.append(nn.Conv2d(channels, 9 * channels, kernel_size=3, padding=1))
        layers.append(nn.PixelShuffle(3))
    else:
        raise ValueError(f"EDSR only supports scale 2, 3, 4, 8 (got {scale})")
    return nn.Sequential(*layers)


class EDSR(nn.Module):
    def __init__(self, scale=4, channels=64, num_blocks=16, res_scale=1.0):
        super().__init__()
        # The input has 2 channels: the AOD values and a mask (1 = real value, 0 = cloud)
        self.first_conv = nn.Conv2d(2, channels, kernel_size=3, padding=1)

        # The main part of the network, which learns the features
        self.blocks = nn.Sequential(*[ResidualBlock(channels, res_scale) for _ in range(num_blocks)])
        self.after_blocks = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.upsample = make_upsampler(channels, scale)

        # Two separate heads: one for the detail, one for the uncertainty
        self.detail_head = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.uncertainty_head = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def forward(self, lr, lr_mask):
        x = torch.cat([lr, lr_mask], dim=1)
        features = self.first_conv(x)
        features = features + self.after_blocks(self.blocks(features))  # long skip connection
        features = self.upsample(features)
        return self.detail_head(features), self.uncertainty_head(features)
