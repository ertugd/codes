"""
consam.py
Conv-SAM (Convolutional Spatial Attention Module) for Ultralytics YOLO
(YOLO26 / YOLOv8-style configs).

Reference: the spatial-attention branch of CBAM
("Convolutional Block Attention Module", Woo et al., ECCV 2018).

Motivation for crack/dent detection - this is the direct complement to ECA:
ECA answers "WHICH channel matters" (channel attention, no spatial
awareness). ConvSAM answers "WHERE in the image matters" (spatial
attention, no channel awareness) - it pools across the channel dimension
(avg + max), runs a conv over the resulting 2-channel map to learn a
per-pixel importance mask, and rescales the original feature map with it.

This is exactly the piece your ablation results suggested was missing:
ECA improved precision/recall but consistently hurt mAP@0.5:0.95
(localization quality) because it has zero spatial reasoning. ConvSAM
should behave the opposite way - since it directly reweights WHERE the
network looks, it's a natural fit for compact, oddly-shaped, spatially
localized defects (dents, crack segments) and is expected to help
box-localization quality more than pure classification confidence.

--------------------------------------------------------------------------
Like ECA, this is the SIMPLEST kind of module to integrate: channel-
preserving (c_out == c_in) AND doesn't need to know the channel count at
construction time (the attention conv operates on the 2-channel
avg/max-pooled map, regardless of how many channels the real input has).
So:

  - No parse_model registration / channel-injection issues at all.
  - No need to hardcode c1/c2 per scale like DySnakeConv/DCNv3.
  - Works in the YAML with an empty or single-value args list, e.g.:

        - [-1, 1, ConvSAM, []]        # default kernel_size=7
        - [-1, 1, ConvSAM, [3]]       # custom kernel_size=3

Notebook registration (same pattern as your other custom modules):

    import ultralytics.nn.tasks as tasks
    from consam import ConvSAM
    tasks.ConvSAM = ConvSAM

    import ultralytics.nn.modules as modules
    modules.ConvSAM = ConvSAM
    if hasattr(modules, "__all__"):
        modules.__all__ = tuple(modules.__all__) + ("ConvSAM",)
--------------------------------------------------------------------------
"""

import torch
import torch.nn as nn


class ConvSAM(nn.Module):
    """Convolutional Spatial Attention Module. Channel-preserving (c_out == c_in).

    Pipeline: x -> [avg-pool-over-channels, max-pool-over-channels] (2, H, W)
                 -> concat -> Conv(2 -> 1, k x k) -> Sigmoid -> reweight x

    Args (YAML): [k_size]   optional, defaults to 7 (odd number recommended,
                             larger kernel = wider spatial context per pixel).
    """

    def __init__(self, k_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        y = torch.cat([avg_out, max_out], dim=1)         # (B, 2, H, W)
        y = self.sigmoid(self.conv(y))                    # (B, 1, H, W)
        return x * y