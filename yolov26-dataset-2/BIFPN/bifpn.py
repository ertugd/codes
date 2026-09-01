# bifpn.py
"""
BiFPN agirlikli fuzyon modulu (EfficientDet, Tan et al., CVPR 2020, Sec. 3.2).

ESKI SURUME GORE NE DEGISTI
---------------------------
(a) N GIRIS. Eski surum `len(x) != 2` icin ValueError atiyordu. Gercek
    BiFPN'de bottom-up yolundaki ARA seviyeler 3 girislidir:
    (bottom-up + top-down + backbone kisayolu). O kisayol BiFPN'i
    PANet'ten ayiran ana yapisal fark; 2 giris zorunlulugu onu
    imkansiz kiliyordu. bifpn.yaml'daki P4 cikis dugumu artik 3 girisli.

(b) OLCEK-KORUYAN NORMALIZASYON. Klasik BiFPN agirlikli TOPLAM icin
    sum(w)=1 kullanir. Burada yapilan islem toplam degil CONCAT: sum(w)=1
    her dali ~1/n ile carpar, yani fuzyon ciktisinin genligi n kat KUCULUR
    (n=3'te ~3x). Ardindan gelen C3k2 on-egitimli agirliklarla geliyor ve
    girisinin birim olcekte olmasini bekliyor. Bu yuzden w, sum(w)=n
    olacak sekilde olceklenir -> ortalama agirlik 1, egitim basindaki
    sinyal genligi duz Concat ile ayni. Ogrenilebilir ORANLAR korunur.

(c) POZITIFLIK ICIN SOFTPLUS. relu(w) tum dallarda 0'a duserse gradyan
    tamamen olur (olu bolge, geri donusu yok). softplus'ta boyle bir
    bolge yoktur.

(d) SABIT BOYUTLU AGIRLIK VEKTORU (n_max). Her dugum ayni shape'te `w`
    tasir, boylece checkpoint yuklemesi dugumun kac girisli oldugundan
    bagimsiz calisir; 2 girisli bir dugumu 3 girisliye cevirmek
    state_dict'i bozmaz.

SINIF ADI NEDEN AYNI KALDI
--------------------------
`BIFPN/runs/detect/ablation_bifpn_v1/weights/best.pt` ve onu zincire alan
`ablasion-all/pretrained_v2.py` pickle icinde `bifpn.WeightedConcat` adini
ariyor. Ad degisirse o checkpoint'ler acilamaz. Eski checkpoint'lerden
unpickle edilen ornekler `__init__`'ten GECMEZ (pickle sadece __dict__'i
geri yukler), bu yuzden `n_max`/`eps` sinif duzeyinde de tanimli - aksi
halde legacy bir ornekte forward AttributeError verirdi.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# softplus(0.5413) ~= 1.0 -> baslangicta tum dallar esit ve birim agirlikli
_W_INIT = 0.5413


class WeightedConcat(nn.Module):
    """BiFPN fast-normalized-fusion tarzi agirlikli concat (N giris).

    YAML:
        - [[-1, 6], 1, Concat, [1]]          # 2 girisli
        - [[-1, 13, 6], 1, Concat, [1]]      # 3 girisli (BiFPN kisayolu)

    Kanal hesabi Ultralytics `parse_model` icindeki `elif m is Concat:
    c2 = sum(ch[x] for x in f)` dalindan gelir; giris sayisi degisse bile
    otomatik dogrudur. Modul giris sayisini calisma aninda ogrenir, YAML'a
    ekstra arguman yazmak gerekmez.
    """

    # Legacy (pickle ile geri yuklenen) ornekler icin sinif duzeyi
    # varsayilanlar - bkz. modul docstring'i.
    n_max = 4
    eps = 1e-4

    def __init__(self, dimension=1, n_max=4, eps=1e-4):
        super().__init__()
        self.d = dimension
        self.n_max = n_max
        self.eps = eps
        self.w = nn.Parameter(torch.full((n_max,), _W_INIT, dtype=torch.float32))

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"WeightedConcat list/tuple bekler, gelen tip: {type(x)}")

        n = len(x)
        if n > self.w.numel():
            raise ValueError(
                f"WeightedConcat n_max={self.w.numel()} ile kuruldu ama {n} giris geldi. "
                f"bifpn.yaml'da daha genis bir fuzyon varsa n_max'i buyut."
            )

        # Agirliklar her zaman fp32'de hesaplanir (AMP altinda softplus +
        # normalizasyon fp16'da hassasiyet kaybeder), sonra girisin dtype'ina
        # dondurulur.
        w = F.softplus(self.w[:n].float())
        w = w * (n / (w.sum() + self.eps))  # sum(w) = n  ->  ortalama agirlik 1
        w = w.to(x[0].dtype)

        ref = x[0].shape[2:]
        outs = []
        for i, t in enumerate(x):
            # Guvenlik agi: tek sayili feature-map boyutlarinda (or. 13 -> 6 -> 12)
            # upsample/downsample bir piksel kayabiliyor.
            if t.shape[2:] != ref:
                t = F.interpolate(t, size=ref, mode="nearest")
            outs.append(t * w[i])
        return torch.cat(outs, self.d)

    def extra_repr(self):
        return f"n_max={self.w.numel()}"


# --------------------------------------------------------------------- #
# Parser kaydi
# --------------------------------------------------------------------- #
# `parse_model` modul adini tasks.py'nin globals()'undan cozer ve kanal
# hesabini `m is Concat` kimlik testiyle secer. Ikisi de ayni ismi
# gosterdigi icin tasks.Concat'i degistirmek yeterli; site-packages
# icindeki dosyalara dokunulmaz.
_ORIGINAL_CONCAT = None


def patch_concat_to_bifpn():
    """Tum `Concat` dugumlerini `WeightedConcat` ile degistirir.

    `YOLO(...)` cagrisindan ONCE calistirilmali.

    DIKKAT: yama sureclidir (global). Ayni kernel'da sonradan BiFPN'siz
    bir model kurulacaksa once `restore_concat()` cagir - aksi halde o
    model de sessizce agirlikli concat alir ve ablasyon kirlenir.
    """
    global _ORIGINAL_CONCAT

    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    if _ORIGINAL_CONCAT is None:
        _ORIGINAL_CONCAT = tasks.Concat

    tasks.Concat = WeightedConcat
    modules.Concat = WeightedConcat
    return WeightedConcat


def restore_concat():
    """`patch_concat_to_bifpn()` yamasini geri alir (idempotent)."""
    global _ORIGINAL_CONCAT

    if _ORIGINAL_CONCAT is None:
        return None

    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    tasks.Concat = _ORIGINAL_CONCAT
    modules.Concat = _ORIGINAL_CONCAT
    restored, _ORIGINAL_CONCAT = _ORIGINAL_CONCAT, None
    return restored
