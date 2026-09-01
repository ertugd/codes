# coordatt.py

import torch
import torch.nn as nn


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu6(x + 3) / 6


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.hsigmoid = HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.hsigmoid(x)


class CoordAtt(nn.Module):
    """
    Coordinate Attention block.

    YOLO26n için kanal değerlerini gerçek nano kanala göre vermeliyiz:
    P3: 64
    P4: 128
    P5: 256

    Örnek YAML:
    - [-1, 1, CoordAtt, [64, 32]]
    """

    def __init__(self, channels, reduction=32):
        super().__init__()

        mip = max(8, channels // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = HSwish()

        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()

        x_h = self.pool_h(x)                  # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, W, 1]

        y = torch.cat([x_h, x_w], dim=2)      # [B, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)

        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w

        return out


def register_coordatt_for_ultralytics():
    import ultralytics.nn.tasks as tasks

    tasks.CoordAtt = CoordAtt

    return CoordAtt