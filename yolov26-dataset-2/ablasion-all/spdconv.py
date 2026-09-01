"""
spdconv.py
YOLO26n icin SPDConv drop-in modulu.

Bu surum Ultralytics parser'i bozmamak icin Conv sinifini patch eder.
YAML'de Conv olarak kalan stride=2 katmanlari runtime'da SPDConv davranisi alir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution.

    Conv ile ayni imzayi kullanir:
        SPDConv(c1, c2, k=1, s=1, p=None, g=1, d=1, act=True)

    s == 2 ise:
        Space-to-depth ile H,W yariya iner, kanal 4 katina cikar.
        Sonra stride=1 Conv uygulanir.

    s != 2 ise:
        Normal Conv-BN-Act gibi davranir.

    Boylece Ultralytics parse_model kanal hesaplari bozulmaz.
    """

    default_act = nn.SiLU

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.use_spd = s == 2
        self.s = s

        conv_c1 = c1 * 4 if self.use_spd else c1
        conv_s = 1 if self.use_spd else s

        self.conv = nn.Conv2d(
            conv_c1,
            c2,
            kernel_size=k,
            stride=conv_s,
            padding=autopad(k, p, d),
            dilation=d,
            groups=g,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)

        if act is True:
            self.act = nn.SiLU(inplace=False)
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    @staticmethod
    def space_to_depth(x):
        # Eger feature map boyutu tek sayi gelirse guvenli padding uygula.
        _, _, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        return torch.cat(
            [
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2],
            ],
            dim=1,
        )

    def forward(self, x):
        if self.use_spd:
            x = self.space_to_depth(x)
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        if self.use_spd:
            x = self.space_to_depth(x)
        return self.act(self.conv(x))


def patch_conv_to_spdconv():
    """
    Ultralytics icindeki Conv referanslarini SPDConv ile degistirir.
    Bu fonksiyon YOLO('yaml') satirindan once cagrilmalidir.
    """
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.Conv = SPDConv
    modules.Conv = SPDConv

    try:
        import ultralytics.nn.modules.conv as conv_modules
        conv_modules.Conv = SPDConv
    except Exception:
        pass

    # Bazi bloklar Conv referansini kendi modul namespace'inde tutabilir.
    try:
        import ultralytics.nn.modules.block as block_modules
        block_modules.Conv = SPDConv
    except Exception:
        pass

    return SPDConv
