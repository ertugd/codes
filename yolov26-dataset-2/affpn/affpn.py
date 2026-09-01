"""
affpn.py
========
AF-FPN (Attention Fusion FPN) - Ultralytics YOLO26 icin dikkat-agirlikli
fuzyon modulu.

Kullanim (affpn.yaml ile):
    from affpn import patch_concat_to_affpn
    patch_concat_to_affpn()                  # YOLO(...) cagrisindan ONCE
    model = YOLO("affpn.yaml").load("../baseline/runs/detect/train/weights/best.pt")

YAML fuzyon noktalarinda `Concat` ismi BILEREK korunur: Ultralytics
`parse_model` kanal hesabini `elif m is Concat: c2 = sum(ch[x] for x in f)`
dalindan yapar, yani yeni bir isim yazilirsa `else: c2 = ch[f]` dalina duser
ve kanal sayisi yanlis hesaplanir. Bu dosya `Concat` ismini calisma aninda
dikkat-fuzyon katmanina baglar.

ESKI SURUME GORE NE DEGISTI
---------------------------
(a) N GIRIS. `len(x) != 2` -> ValueError kaldirildi. Eskiden 3 girisli bir
    fuzyon dugumu eklemek imkansizdi; ustelik `AFFPNConcat3` yazilmis ama
    `patch_concat_to_affpn()` her zaman `AFFPNConcat2`ye bagladigi icin
    O SINIF HIC KULLANILMIYORDU (olu kod).

(b) OLCEK-KORUYAN NORMALIZASYON. Klasik agirlikli TOPLAM icin sum(w)=1
    dogrudur; ama burada yapilan islem CONCAT. sum(w)=1, 2 girisli bir
    dugumde her dali ~0.5 ile carpar ve fuzyon ciktisinin genligi YARIYA
    duser. Ardindan gelen C3k2 on-egitimli agirliklarla geliyor ve girisinin
    birim olcekte olmasini bekliyor. Artik sum(w)=n -> ortalama agirlik 1,
    egitim basindaki sinyal genligi duz Concat ile ayni. Ogrenilebilir
    ORANLAR aynen korunur.

(c) POZITIFLIK ICIN SOFTPLUS. relu(w) tum dallarda 0'a duserse gradyan
    tamamen olur (geri donusu olmayan olu bolge). softplus'ta yok.

(d) SPATIAL ATTENTION'DAN BatchNorm2d(1) CIKARILDI. Tek kanal uzerinde BN,
    sigmoid'e giren haritayi her batch'te sifir-ortalama/birim-varyansa
    zorluyordu; yani dikkat haritasi ICERIKTEN BAGIMSIZ olarak her zaman
    ~%50 pozisyonu 0.5'in ustunde birakmak zorunda kaliyordu. "Her yere
    bak" ya da "hicbir yere bakma" ifade edilemiyordu. Ustelik egitimde
    batch istatistigi, cikarimda running istatistik kullanildigi icin
    train/val tutarsizligi da uretiyordu. Standart CBAM'deki gibi
    Conv(bias=True) + Sigmoid birakildi.

(e) Boyut uyusmazliginda ValueError yerine interpolate (guvenlik agi;
    tek sayili feature-map boyutlarinda up/downsample bir piksel kayabilir).

(f) `restore_concat()` eklendi - global yama geri alinabiliyor.

SINIF ADLARI NEDEN AYNI KALDI
-----------------------------
`affpn/runs/detect/affpn/weights/best.pt` pickle icinde `affpn.AFFPNConcat2`
adini ariyor. Ad degisirse o checkpoint acilamaz. Pickle `__init__`'i
CAGIRMAZ (sadece __dict__'i geri yukler), bu yuzden `n_max`/`eps` sinif
duzeyinde de tanimli.

!! DIKKAT: (d) maddesi state_dict anahtarlarini degistirir
   (spatial_attn.1.* BN anahtarlari kayboldu). Eski affpn checkpoint'i
   ACILIR ama dikkat katmanlarinin agirliklari yeni modele TRANSFER OLMAZ;
   yeniden egitim gerekir.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# softplus(0.5413) ~= 1.0 -> baslangicta tum dallar esit ve birim agirlikli
_W_INIT = 0.5413


class AFFPNConcat2(nn.Module):
    """Dikkat-agirlikli fuzyon concat (N giris).

    Cikis:
        cat([g(x_1) * w_1, ..., g(x_n) * w_n], dim)

    Tasarim:
      1) Ogrenilebilir skaler dal agirliklari (w) global dal onemini secer.
         softplus ile pozitif, sum(w)=n olacak sekilde normalize.
      2) Hafif spatial attention her dali kendi icerigine gore kapilar.
         Kapi PAYLASIMLI (CBAM'deki gibi tek saliency dedektoru) - dal
         basina ayri kapi, parse_model dugum arity'sini module
         gecirmedigi icin kullanilmayan parametre birakirdi.
      3) `gamma` 0'dan baslar: model tam olarak duz Concat davranisiyla
         baslar, dikkati kademeli ogrenir (rezidual kapilama).

    Modul YAML'dan kanal sayisi ISTEMEZ, bu yuzden Concat yerine guvenle
    gecirilebilir. Giris sayisini calisma aninda ogrenir.

    Ad neden "2" ile bitiyor: eski checkpoint uyumlulugu (bkz. modul
    docstring'i). Artik N girisi destekler.
    """

    # Legacy (pickle ile geri yuklenen) ornekler icin sinif duzeyi varsayilanlar
    n_max = 4
    eps = 1e-4

    def __init__(self, dimension=1, n_max=4, eps=1e-4, kernel_size=3):
        super().__init__()
        self.d = dimension
        self.n_max = n_max
        self.eps = eps

        self.w = nn.Parameter(torch.full((n_max,), _W_INIT, dtype=torch.float32))

        # Spatial attention: [avg_map, max_map] -> [B, 2, H, W] -> [B, 1, H, W]
        # BN YOK (bkz. modul docstring (d)); bias=True ile CBAM standardi.
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, bias=True),
            nn.Sigmoid(),
        )

        # Dikkat siddeti. 0'dan baslar -> baslangicta saf Concat.
        self.gamma = nn.Parameter(torch.zeros(1))

    def _spatial_gate(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.spatial_attn(torch.cat([avg_map, max_map], dim=1))
        return x * (1.0 + self.gamma * attn)

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"AFFPNConcat2 list/tuple bekler, gelen tip: {type(x)}")

        n = len(x)
        if n > self.w.numel():
            raise ValueError(
                f"AFFPNConcat2 n_max={self.w.numel()} ile kuruldu ama {n} giris geldi. "
                f"affpn.yaml'da daha genis bir fuzyon varsa n_max'i buyut."
            )

        # Agirliklar her zaman fp32'de hesaplanir (AMP altinda softplus +
        # normalizasyon fp16'da hassasiyet kaybeder), sonra girisin dtype'ina donulur.
        w = F.softplus(self.w[:n].float())
        w = w * (n / (w.sum() + self.eps))  # sum(w) = n  ->  ortalama agirlik 1
        w = w.to(x[0].dtype)

        ref = x[0].shape[2:]
        outs = []
        for i, t in enumerate(x):
            if t.shape[2:] != ref:  # guvenlik agi: tek sayili boyutlarda 1px kayma
                t = F.interpolate(t, size=ref, mode="nearest")
            outs.append(self._spatial_gate(t) * w[i])
        return torch.cat(outs, self.d)

    def extra_repr(self):
        return f"n_max={self.w.numel()}"


# Geriye donuk uyumluluk: AFFPNConcat2 artik N giris destekledigi icin ayri
# bir 3-girisli sinifa gerek kalmadi. Eski import'lar kirilmasin diye alias.
AFFPNConcat3 = AFFPNConcat2


# --------------------------------------------------------------------- #
# Parser kaydi
# --------------------------------------------------------------------- #
_ORIGINAL_CONCAT = None


def patch_concat_to_affpn():
    """Tum `Concat` dugumlerini `AFFPNConcat2` ile degistirir.

    `YOLO(...)` cagrisindan ONCE calistirilmali.

    DIKKAT: yama global. Ayni kernel'da sonradan AF-FPN'siz bir model
    kurulacaksa once `restore_concat()` cagir - aksi halde o model de
    sessizce dikkat-fuzyonu alir ve ablasyon kirlenir.
    """
    global _ORIGINAL_CONCAT

    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    if _ORIGINAL_CONCAT is None:
        _ORIGINAL_CONCAT = tasks.Concat

    tasks.Concat = AFFPNConcat2
    modules.Concat = AFFPNConcat2

    try:
        import ultralytics.nn.modules.conv as conv_modules

        conv_modules.Concat = AFFPNConcat2
    except Exception:
        pass

    return AFFPNConcat2


def restore_concat():
    """`patch_concat_to_affpn()` yamasini geri alir (idempotent)."""
    global _ORIGINAL_CONCAT

    if _ORIGINAL_CONCAT is None:
        return None

    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    tasks.Concat = _ORIGINAL_CONCAT
    modules.Concat = _ORIGINAL_CONCAT
    try:
        import ultralytics.nn.modules.conv as conv_modules

        conv_modules.Concat = _ORIGINAL_CONCAT
    except Exception:
        pass

    restored, _ORIGINAL_CONCAT = _ORIGINAL_CONCAT, None
    return restored
