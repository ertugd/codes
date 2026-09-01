"""
INN-YOLO ozel modulleri
------------------------
Bu dosya, ultralytics/nn/modules/ yapisina eklenmek uzere hazirlanmistir.
Icerik:
    1) SAM               -> Spatial Attention Module (CBAM'in spatial kolu)
    2) Conv_SAM           -> Conv (k=3,s=2) + BN + SiLU + SAM
    3) RepVGGBlock         -> Egitimde cok dalli (3x3 + 1x1 + identity),
                              inference'ta tek 3x3'e reparametrize edilebilen blok
    4) Bottleneck_Rep      -> Bottleneck, RepVGGBlock ile
    5) C3k_Rep / C3k2_Rep  -> Diyagramdaki "C3k2-RepVGG" bloğu
    6) BiFPN_Add2 / BiFPN_Add3 -> Ogrenilebilir agirlikli ozellik birlestirme
    7) BiFPN_Block         -> Conv-SAM ile birlikte tam bir BiFPN dugumu

Kullanim (ultralytics reposu icinde):
    - Bu dosyayi ultralytics/nn/modules/innyolo.py olarak kaydedin.
    - ultralytics/nn/modules/__init__.py icine:
        from .innyolo import Conv_SAM, C3k2_Rep, BiFPN_Add2, BiFPN_Add3, BiFPN_Block
      satirini ekleyin ve __all__ listesine isimleri dahil edin.
    - ultralytics/nn/tasks.py icindeki parse_model fonksiyonunda bu siniflari
      taniyacak sekilde (Conv, C3k2 gibi diger moduller nasil isleniyorsa)
      ilgili if/elif bloklarina ekleyin.
    - yolo11-inn.yaml gibi bir config dosyasinda backbone/neck katmanlarinda
      bu modul isimlerini kullanin (Conv -> Conv_SAM, C3k2 -> C3k2_Rep, neck'te BiFPN_Block).
"""

import torch
import torch.nn as nn

# ultralytics.nn.modules.conv icindeki autopad ve Conv sinifini kullaniyoruz
from ultralytics.nn.modules.conv import Conv, autopad


# ---------------------------------------------------------------------------
# 1) Spatial Attention Module (SAM)
# ---------------------------------------------------------------------------
class SAM(nn.Module):
    """Spatial Attention Module.

    Kanal boyutunda avg-pool ve max-pool alip concat eder, k=7 conv + sigmoid
    ile tek kanallik bir mekansal agirlik haritasi uretir ve girdiyi bu harita
    ile carpar. (CBAM makalesindeki spatial attention kolu)
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size 3 ya da 7 olmali"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class Conv_SAM(nn.Module):
    """Diyagramdaki 'Conv-SAM' bloğu: standart Conv (k=3, s=2, Conv+BN+SiLU) + SAM."""

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True, sam_kernel=7):
        super().__init__()
        self.conv = Conv(c1, c2, k, s, p, g, d, act)
        self.sam = SAM(sam_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sam(self.conv(x))


# ---------------------------------------------------------------------------
# 2) RepVGG tarzi reparametrize edilebilir blok
# ---------------------------------------------------------------------------
class RepVGGBlock(nn.Module):
    """Egitimde: 3x3 conv-bn + 1x1 conv-bn + (varsa) identity-bn dallarinin toplami, SiLU.
    Inference'ta: fuse_convs() cagrildiktan sonra tek bir 3x3 Conv2d'ye donusur
    (RepVGG makalesindeki 'structural re-parameterization').
    """

    def __init__(self, c1, c2, k=3, s=1, g=1, act=True):
        super().__init__()
        assert k == 3 and s == 1, "RepVGGBlock bu implementasyonda k=3, s=1 icin tanimlidir"
        self.c1, self.c2, self.g = c1, c2, g
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        self.dense_3x3 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 1, 1, groups=g, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.dense_1x1 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0, groups=g, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.identity = nn.BatchNorm2d(c1) if c1 == c2 else None
        self.reparam_conv = None  # fuse_convs() sonrasi dolar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparam_conv is not None:
            return self.act(self.reparam_conv(x))
        id_out = 0 if self.identity is None else self.identity(x)
        return self.act(self.dense_3x3(x) + self.dense_1x1(x) + id_out)

    # --- Reparametrizasyon (deploy/export oncesi cagirin) ---
    def _pad_1x1_to_3x3(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _fuse_bn_identity(self, bn):
        input_dim = self.c1 // self.g
        kernel = torch.zeros((self.c1, input_dim, 3, 3), device=bn.weight.device)
        for i in range(self.c1):
            kernel[i, i % input_dim, 1, 1] = 1
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    def fuse_convs(self):
        if self.reparam_conv is not None:
            return
        k3, b3 = self._fuse_bn(self.dense_3x3[0], self.dense_3x3[1])
        k1, b1 = self._fuse_bn(self.dense_1x1[0], self.dense_1x1[1])
        k1 = self._pad_1x1_to_3x3(k1)
        if self.identity is not None:
            kid, bid = self._fuse_bn_identity(self.identity)
        else:
            kid, bid = 0, 0
        final_k = k3 + k1 + kid
        final_b = b3 + b1 + bid

        self.reparam_conv = nn.Conv2d(
            self.c1, self.c2, 3, 1, 1, groups=self.g, bias=True
        ).to(self.dense_3x3[0].weight.device)
        self.reparam_conv.weight.data = final_k
        self.reparam_conv.bias.data = final_b

        for attr in ("dense_3x3", "dense_1x1", "identity"):
            if hasattr(self, attr):
                delattr(self, attr)


class Bottleneck_Rep(nn.Module):
    """Diyagramdaki sari 'Bottleneck' kutusu, ikinci dal RepVGGBlock ile degistirilmis hali."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = RepVGGBlock(c_, c2, k=3, s=1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


# ---------------------------------------------------------------------------
# 3) C3k_Rep / C3k2_Rep  (diyagramdaki "C3k2-RepVGG")
# ---------------------------------------------------------------------------
class C3k_Rep(nn.Module):
    """C3k bloğu, Bottleneck yerine Bottleneck_Rep kullanan versiyonu."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(Bottleneck_Rep(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2_Rep(nn.Module):
    """Diyagramdaki 'C3k2' bloğunun RepVGG'li versiyonu ('C3k2-RepVGG').

    Yapi orijinal C3k2 ile ayni: Conv+split, ardindan n adet C3k_Rep (ya da
    dogrudan Bottleneck_Rep, c3k=False iken) zinciri, sonunda concat + Conv.
    """

    def __init__(self, c1, c2, n=1, c3k=True, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            C3k_Rep(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck_Rep(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ---------------------------------------------------------------------------
# 4) BiFPN - ogrenilebilir agirlikli ozellik birlestirme
# ---------------------------------------------------------------------------
class BiFPN_Add2(nn.Module):
    """Iki girdiyi ogrenilebilir, normalize edilmis agirliklarla toplar (EfficientDet BiFPN)."""

    def __init__(self, c1, c2):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4
        self.conv = Conv(c1, c2, 1, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        w = self.w.relu()  # negatif olmamasi icin
        w = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.conv(self.act(w[0] * x[0] + w[1] * x[1]))


class BiFPN_Add3(nn.Module):
    """Uc girdi icin agirlikli toplama (BiFPN'in orta seviye dugumleri: top-down + bottom-up + skip)."""

    def __init__(self, c1, c2):
        super().__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4
        self.conv = Conv(c1, c2, 1, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        w = self.w.relu()
        w = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.conv(self.act(w[0] * x[0] + w[1] * x[1] + w[2] * x[2]))


class BiFPN_Block(nn.Module):
    """Diyagramdaki pembe 'BiFPN' kutusunu tek modulde toplar: agirlikli toplama + Conv-SAM.

    n_inputs=2 -> en ust/en alt seviyeler icin (2 giris: upsample/downsample + skip)
    n_inputs=3 -> orta seviyeler icin (3 giris: top-down + bottom-up + orijinal skip)
    """

    def __init__(self, c1, c2, n_inputs=2):
        super().__init__()
        assert n_inputs in (2, 3)
        self.add = BiFPN_Add2(c1, c1) if n_inputs == 2 else BiFPN_Add3(c1, c1)
        self.out = Conv_SAM(c1, c2, k=3, s=1)

    def forward(self, x):
        return self.out(self.add(x))