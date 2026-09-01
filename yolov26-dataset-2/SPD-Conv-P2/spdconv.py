"""
spdconv.py
==========
SPD-Conv: "No More Strided Convolutions or Pooling"
(Sunkara & Purushotham, ECML-PKDD 2022).

Stride-2 konvolusyon yerine space-to-depth: 2x2 alt-ornekleme ile 4 alt-harita
cikarilir, kanal ekseninde birlestirilir (H,W yariya iner, kanal 4x olur),
sonra stride-1 Conv ile hedef kanala indirilir. Downsample sirasinda hicbir
piksel atilmaz - ince catlak gibi kucuk yapilar icin kritik.

KULLANIM (spd-covn-p2.yaml ile)
-------------------------------
    from spdconv import register_spdconv
    register_spdconv()                      # YOLO(...) cagrisindan ONCE
    model = YOLO("spd-covn-p2.yaml")

YAML imzasi `Conv` ile birebir aynidir ([out_channels, kernel, stride]), bu
yuzden parse_model kanal hesabi bozulmaz: SPDConv `base_modules` icinde
oldugu icin `c1, c2 = ch[f], args[0]` dogru calisir, c1*4 genislemesi modulun
KENDI icinde yapilir.

!! SITE-PACKAGES SURUMU ILE KARISTIRMA !!
-----------------------------------------
Bu projede ultralytics'e elle eklenmis `ultralytics/nn/modules/inn_modules_v2.py`
icinde de bir `SPDConv` var. Ikisi ayni cikti seklini uretir AMA state_dict
anahtarlari FARKLIDIR:

    site-packages : conv.conv.weight / conv.bn.weight   (icine Conv sarar)
    bu dosya      : conv.weight      / bn.weight        (duz)

`SPD-Conv/runs/detect/ablation_spd_conv_v2` checkpoint'i BU DOSYANIN formatinda
egitildi. Notebook `register_spdconv()` cagirmazsa `tasks.SPDConv` site-packages
surumune cozulur ve SPD-Conv agirliklari SESSIZCE transfer olmaz (intersect_dicts
anahtar ismi tutmadigi icin atlar - hata vermez, sadece dusuk transfer sayisi).
Bu yuzden register_spdconv() her notebook'ta cagrilmalidir.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv'u MODUL YUKLENIRKEN sabitliyoruz. patch_conv_to_spdconv() gibi bir yama
# sonradan ultralytics.Conv'u degistirse bile bu referans etkilenmez
# (aksi halde SPDConv kendi kendisinin alt sinifi olmaya calisirdi).
from ultralytics.nn.modules.conv import Conv as _UltralyticsConv


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SPDConv(_UltralyticsConv):
    """Space-to-Depth Convolution. Conv ile ayni imza:

        SPDConv(c1, c2, k=1, s=1, p=None, g=1, d=1, act=True)

    s == 2 -> space-to-depth (H,W yariya, kanal 4x) + stride-1 Conv
    s != 2 -> duz Conv-BN-Act

    NEDEN Conv'DAN TUREMIS: Ultralytics `BaseModel.fuse()` yalnizca
    `isinstance(m, (Conv, Conv2, DWConv))` olan katmanlari fuse eder. Duz
    nn.Module olarak birakilirsa SPDConv katmanlari fuse() sirasinda SESSIZCE
    ATLANIR; BatchNorm katlanmaz ve `forward_fuse` hic cagrilmaz (olu kod).
    Conv'dan turetince conv+bn katlanmasi calisir, `.conv`/`.bn`/`.act`
    yapisi ayni oldugu icin `fuse_conv_and_bn` dogrudan islevini gorur.

    Conv.__init__ CAGRILMAZ (o, c1 kanalli bir conv kurardi); yerine
    nn.Module.__init__ ile katmanlar burada kuruluyor.
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        nn.Module.__init__(self)
        self.use_spd = s == 2
        self.s = s

        conv_c1 = c1 * 4 if self.use_spd else c1
        conv_s = 1 if self.use_spd else s

        self.conv = nn.Conv2d(
            conv_c1, c2,
            kernel_size=k, stride=conv_s, padding=autopad(k, p, d),
            dilation=d, groups=g, bias=False,
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
        # Tek sayili feature-map boyutlarina karsi guvenli padding.
        # (640 girdide hicbir seviye tek sayi olmaz; bu bir guvenlik agi.)
        _, _, h, w = x.shape
        pad_h, pad_w = h % 2, w % 2
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
        """fuse() sonrasi cagrilir (bn katlanmis, artik yok)."""
        if self.use_spd:
            x = self.space_to_depth(x)
        return self.act(self.conv(x))


def register_spdconv():
    """`SPDConv` adini BU dosyadaki sinifa baglar.

    YAML'da `SPDConv` yazan her dugum bu sinifa cozulur. `YOLO(...)`
    cagrisindan ONCE calistirilmali.

    Bu, site-packages'taki farkli-anahtarli SPDConv'un devreye girmesini
    engeller (bkz. modul docstring'i) - checkpoint uyumlulugu icin sarttir.
    """
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    tasks.SPDConv = SPDConv
    modules.SPDConv = SPDConv
    return SPDConv


# --------------------------------------------------------------------- #
# LEGACY - yeni notebook'larda KULLANMA
# --------------------------------------------------------------------- #
def patch_conv_to_spdconv():
    """TUM `Conv` referanslarini SPDConv ile degistirir. (Eski yontem.)

    ONERILMEZ: `block_modules.Conv` da degistigi icin C3k2 / SPPF / C2PSA
    gibi bloklarin ICINDEKI stride-1 conv'lar da SPDConv olur. Davranis
    ayni kalir (s != 2 -> duz Conv) ama model repr'i ve state_dict
    anahtarlari gereksiz yere degisir, hangi katmanin gercekten SPD
    oldugu okunamaz hale gelir.

    Yeni yaml'lar SPDConv'u ACIKCA yaziyor; `register_spdconv()` yeterli
    ve daha guvenli. Bu fonksiyon yalnizca eski notebook'lar kirilmasin
    diye duruyor.
    """
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    tasks.Conv = SPDConv
    modules.Conv = SPDConv

    try:
        import ultralytics.nn.modules.conv as conv_modules

        conv_modules.Conv = SPDConv
    except Exception:
        pass

    try:
        import ultralytics.nn.modules.block as block_modules

        block_modules.Conv = SPDConv
    except Exception:
        pass

    return SPDConv
