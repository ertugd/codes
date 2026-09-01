# eca.py

import torch
import torch.nn as nn


class ECA(nn.Module):
    """
    Efficient Channel Attention block.

    YOLO26n için ilk denemede k_size=3 öneriyorum.
    Bu modül feature map boyutunu değiştirmez.
    Input : [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(self, k_size: int = 3):
        super().__init__()

        assert k_size % 2 == 1, "k_size tek sayı olmalı. Örn: 3 veya 5"

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]

        y = self.avg_pool(x)                 # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)                     # [B, 1, C]
        y = self.sigmoid(y)

        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]

        return x * y.expand_as(x)


def register_eca_for_ultralytics():
    """
    Ultralytics YAML parser'ın ECA sınıfını görebilmesi için pratik kayıt yöntemi.

    Kullanım:
        from eca import register_eca_for_ultralytics
        register_eca_for_ultralytics()

        model = YOLO("yolo26n_eca_custom.yaml")
    """

    import ultralytics.nn.tasks as tasks

    tasks.ECA = ECA

    return ECA