import torch
import torch.nn as nn


class ResidualECA(nn.Module):
    """
    Residual Efficient Channel Attention.

    Klasik ECA:
        out = x * attn

    Daha güvenli sürüm:
        out = x * (1 + gamma * attn)

    gamma başlangıçta 0 olduğu için model eğitime baseline davranışına yakın başlar.
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

        # Başlangıçta ECA etkisi sıfır.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)                 # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)

        return x * (1.0 + self.gamma * y)


def register_residual_eca_for_ultralytics():
    import ultralytics.nn.tasks as tasks

    tasks.ResidualECA = ResidualECA

    return ResidualECA