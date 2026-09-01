"""
eca.py
Efficient Channel Attention (ECA) for Ultralytics YOLO (YOLO26 / YOLOv8-style configs).

Reference: "ECA-Net: Efficient Channel Attention for Deep Convolutional
Neural Networks" (Wang et al., CVPR 2020).

Motivation for crack/dent detection: cheap channel attention. ECA does a
global average pool over each channel, then a lightweight 1D conv across
the channel dimension (no dimensionality reduction, unlike SE-block) to
learn which channels matter most, and rescales the feature map accordingly.
Adds almost no parameters/FLOPs, so it's a very low-risk ablation add-on -
useful as a cheap baseline attention module to compare against heavier
options (CA, SimAM, C2PSA, etc.).

--------------------------------------------------------------------------
This is the SIMPLEST custom module in this set to integrate, because ECA
is inherently channel-preserving AND doesn't even need to know the channel
count at construction time (the 1D conv operates on the pooled channel
vector regardless of how many channels there are). So:

  - No parse_model registration / channel-injection issues at all.
  - No need to hardcode c1/c2 per scale like DySnakeConv/DCNv3.
  - Works in the YAML with an empty or single-value args list, e.g.:

        - [-1, 1, ECA, []]        # default kernel_size=3
        - [-1, 1, ECA, [5]]       # custom kernel_size=5

Notebook registration (same pattern as your other custom modules):

    import ultralytics.nn.tasks as tasks
    from eca import ECA
    tasks.ECA = ECA

    import ultralytics.nn.modules as modules
    modules.ECA = ECA
    if hasattr(modules, "__all__"):
        modules.__all__ = tuple(modules.__all__) + ("ECA",)
--------------------------------------------------------------------------
"""

import torch
import torch.nn as nn


class ECA(nn.Module):
    """Efficient Channel Attention block. Channel-preserving (c_out == c_in).

    Args (YAML): [k_size]   optional, defaults to 3 (odd number recommended).
    """

    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x)                          # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)            # (B, 1, C)
        y = self.conv(y)                               # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)           # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)