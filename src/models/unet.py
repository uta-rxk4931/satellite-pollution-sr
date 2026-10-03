"""
U-Net body: upsample first, then correct.

Based on:
  - O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional Networks for
    Biomedical Image Segmentation," MICCAI, 2015. https://arxiv.org/abs/1505.04597
    (no official PyTorch code; this follows Figure 1 of the paper)

Changes for this project (on purpose):
  - 3 levels instead of 4, and 32 -> 256 channels instead of 64 -> 1024
    (sized for 64x64 patches and ~2M parameters, close to EDSR's 1.5M)
  - padded convs so the output is the same size as the input (the paper uses
    unpadded convs and crops the skip connections)
  - no dropout
  - 2 input channels (AOD + cloud mask) and 2 output heads (detail + uncertainty)
  - the input is upsampled with bicubic first, since U-Net keeps the size the same

The low-res map is blown up to 64x64 with bicubic first. Then the U-Net
shrinks it (to see the big picture) and grows it back (to put detail back),
with skip connections carrying fine detail straight across.

    (2,64,64) -> 32ch 64x64 ---------------------------> concat -> 32ch 64x64 -> heads
                   -> 64ch 32x32 ------------> concat -> 64ch 32x32
                        -> 128ch 16x16 -> concat -> 128ch 16x16
                             -> 256ch 8x8 (bottleneck: sees the whole patch)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """conv -> ReLU -> conv -> ReLU, keeps the image size the same."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, scale=4, base=32):
        super().__init__()
        self.scale = scale
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8   # 32, 64, 128, 256

        # Encoder (going down): each level halves the size and doubles the channels
        self.down1 = DoubleConv(2, c1)
        self.down2 = DoubleConv(c1, c2)
        self.down3 = DoubleConv(c2, c3)
        self.bottom = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)

        # Decoder (going up): ConvTranspose doubles the size, then we glue on
        # the matching encoder output (skip connection) and clean it up
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 * 2, c1)

        self.detail_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.uncertainty_head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, lr, lr_mask):
        # Step 1: upsample first (bicubic for values, nearest for the cloud mask)
        big = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
        big_mask = F.interpolate(lr_mask, scale_factor=self.scale, mode="nearest")
        x = torch.cat([big, big_mask], dim=1)                 # (2, 64, 64)

        # Step 2: go down
        e1 = self.down1(x)                                    # (32, 64, 64)
        e2 = self.down2(self.pool(e1))                        # (64, 32, 32)
        e3 = self.down3(self.pool(e2))                        # (128, 16, 16)
        b = self.bottom(self.pool(e3))                        # (256, 8, 8)

        # Step 3: come back up, gluing on the skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))   # (128, 16, 16)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (64, 32, 32)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (32, 64, 64)

        return self.detail_head(d1), self.uncertainty_head(d1)
