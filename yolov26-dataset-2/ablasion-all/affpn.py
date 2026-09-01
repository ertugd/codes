"""
affpn.py
AF-FPN (Attention Fusion FPN) module for Ultralytics YOLO26 experiments.

Usage idea with the provided yolo26n_affpn.yaml:
    from affpn import patch_concat_to_affpn
    patch_concat_to_affpn()
    model = YOLO("yolo26n_affpn.yaml").load("yolo26n.pt")

The YAML keeps `Concat` at multi-input fusion points because Ultralytics parses
Concat specially. This file patches Concat to an attention-weighted fusion layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AFFPNConcat2(nn.Module):
    """
    Two-input Attention Fusion concat for FPN/PAN necks.

    Input:
        x = [x1, x2]
        x1, x2 must have the same H and W.

    Output:
        torch.cat([a1 * x1, a2 * x2], dim=dimension)

    Attention design:
      1) Learnable BiFPN-like scalar weights decide global branch importance.
      2) Lightweight spatial attention gates each input independently.
      3) Output keeps concat behavior, so downstream C3k2 channel expectations stay valid.

    This module does not require channel count in YAML, so it is safe as a Concat replacement.
    """

    def __init__(self, dimension=1, epsilon=1e-4, kernel_size=3):
        super().__init__()
        self.d = dimension
        self.epsilon = epsilon

        padding = kernel_size // 2

        # Learnable scalar branch weights. No in-place operation is used on this parameter.
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        # Spatial attention shared by both branches.
        # Input descriptor: [avg_map, max_map] -> [B, 2, H, W]
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        # Residual attention strength. Starts at 0, so the model begins close to normal Concat.
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _spatial_gate(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        desc = torch.cat([avg_map, max_map], dim=1)
        attn = self.spatial_attn(desc)
        return x * (1.0 + self.gamma * attn)

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"AFFPNConcat2 expects list/tuple input, got {type(x)}")

        if len(x) != 2:
            raise ValueError(f"AFFPNConcat2 expects exactly 2 inputs, got {len(x)}")

        x1, x2 = x

        if x1.shape[-2:] != x2.shape[-2:]:
            raise ValueError(
                f"AFFPNConcat2 input spatial sizes must match, got {x1.shape[-2:]} and {x2.shape[-2:]}"
            )

        # Non-inplace positive normalization, safe for autograd.
        w = F.relu(self.w, inplace=False)
        weight = w / (w.sum() + self.epsilon)

        x1 = self._spatial_gate(x1) * weight[0]
        x2 = self._spatial_gate(x2) * weight[1]

        return torch.cat([x1, x2], dim=self.d)


class AFFPNConcat3(nn.Module):
    """
    Three-input version for future AF-FPN designs.
    The current yolo26n_affpn.yaml uses only 2-input fusion points.
    """

    def __init__(self, dimension=1, epsilon=1e-4, kernel_size=3):
        super().__init__()
        self.d = dimension
        self.epsilon = epsilon
        padding = kernel_size // 2
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _spatial_gate(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        desc = torch.cat([avg_map, max_map], dim=1)
        attn = self.spatial_attn(desc)
        return x * (1.0 + self.gamma * attn)

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"AFFPNConcat3 expects list/tuple input, got {type(x)}")

        if len(x) != 3:
            raise ValueError(f"AFFPNConcat3 expects exactly 3 inputs, got {len(x)}")

        if not (x[0].shape[-2:] == x[1].shape[-2:] == x[2].shape[-2:]):
            raise ValueError("AFFPNConcat3 input spatial sizes must match")

        w = F.relu(self.w, inplace=False)
        weight = w / (w.sum() + self.epsilon)

        y = [self._spatial_gate(x[i]) * weight[i] for i in range(3)]
        return torch.cat(y, dim=self.d)


def patch_concat_to_affpn():
    """
    Patch Ultralytics Concat to AF-FPN attention fusion.

    Call this BEFORE YOLO("yolo26n_affpn.yaml").

    Example:
        from ultralytics import YOLO
        from affpn import patch_concat_to_affpn

        patch_concat_to_affpn()
        model = YOLO("yolo26n_affpn.yaml")
    """
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.Concat = AFFPNConcat2
    modules.Concat = AFFPNConcat2

    try:
        import ultralytics.nn.modules.conv as conv_modules
        conv_modules.Concat = AFFPNConcat2
    except Exception:
        pass

    return AFFPNConcat2
