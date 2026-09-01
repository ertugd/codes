# consam.py

import torch
import torch.nn as nn


class ConSAM(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()

        assert kernel_size in [3, 5, 7], "kernel_size 3, 5 veya 7 olmalı"

        padding = kernel_size // 2

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=1,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        spatial = torch.cat([avg_out, max_out], dim=1)
        attention = self.spatial_conv(spatial)

        # Inplace değil, güvenli residual attention
        return x * (1.0 + self.gamma * attention)


def register_consam_for_ultralytics():
    import ultralytics.nn.tasks as tasks
    tasks.ConSAM = ConSAM
    return ConSAM