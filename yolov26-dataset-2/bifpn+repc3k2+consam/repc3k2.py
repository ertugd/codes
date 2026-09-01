import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, g=1, act=True):
        super().__init__()

        self.conv3 = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, autopad(k), groups=g, bias=False),
            nn.BatchNorm2d(c2)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, s, 0, groups=g, bias=False),
            nn.BatchNorm2d(c2)
        )

        self.identity = nn.BatchNorm2d(c1) if c1 == c2 and s == 1 else None
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        out = self.conv3(x) + self.conv1(x)

        if self.identity is not None:
            out = out + self.identity(x)

        return self.act(out)


class RepBottleneck(nn.Module):
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1 = RepConv(c, c, k=3, s=1)
        self.cv2 = ConvBNAct(c, c, k=1, s=1)
        self.add = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class RepC3k2(nn.Module):
    """
    Shape-preserving RepC3k2.
    Input : [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(self, c, n=1, shortcut=True, e=0.5):
        super().__init__()

        c_ = int(c * e)

        self.cv1 = ConvBNAct(c, c_, k=1, s=1)
        self.cv2 = ConvBNAct(c, c_, k=1, s=1)
        self.m = nn.Sequential(*(RepBottleneck(c_, shortcut=shortcut) for _ in range(n)))
        self.cv3 = ConvBNAct(2 * c_, c, k=1, s=1)

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


def register_repc3k2_for_ultralytics():
    import ultralytics.nn.tasks as tasks
    tasks.RepC3k2 = RepC3k2
    return RepC3k2