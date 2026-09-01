"""
pretrained_v2.py
================
v2 mimarisini, bu veri setinde ZATEN EGITILMIS kendi checkpoint'lerinden
baslatmak icin zincirleme agirlik yukleyici.

NEDEN COCO YERINE BUNLAR
------------------------
v2 modeline transfer olan parametre orani (`yolo26n-...-v2.yaml` hedefi,
`intersect_dicts` ile isim+shape eslesmesi):

    COCO yolo26n.pt                   353/1118 tensor   %20.2 parametre
    runs/detect/train-2 (best)        353/1118 tensor   %20.2 parametre
    BIFPN/ablation_bifpn_v1 (best)    353/1118 tensor   %20.2 parametre
    SPD-Conv/ablation_spd_conv_v      358/1118 tensor   %58.7 parametre

Aradaki ucurumun sebebi backbone'daki SPD-Conv katmanlari (model.0/1/3/5/7):
SPDConv'un conv agirligi `c2 x (4*c1) x k x k`, stok `Conv` ise
`c2 x c1 x k x k`. Yani COCO ya da SPD-Conv'suz bir checkpoint bu
katmanlarin SADECE BatchNorm'unu doldurabiliyor, konvolusyonu bos
biraksiyor - ve bunlar modelin en agir katmanlari. SPD-Conv'lu bir
checkpoint ise hepsini dolduruyor.

ZINCIRLEME MANTIGI
------------------
`YOLO.load()` her cagrisinda isim+shape eslesen TUM anahtarlari ezer;
yani SON cagri onceki degerleri gecersiz kilar. Bu yuzden zincir
"en zayif kaynak once, en iyi kaynak sonda" siralanir:

    train-2  ->  BIFPN  ->  SPD-Conv

Bu, `ablasion-main.ipynb`'deki `.load().load().load()` zincirinin ayni
mantigi - ama burada her adimda KAC tensor'un yeni geldigi ve kacinin
uzerine yazildigi raporlaniyor, "yukledim ama aslinda hicbir sey
degismedi" durumu gorunur olsun diye.

Rapordan gorulecegi uzere `train-2` ve `bifpn` sifir YENI tensor
katiyor - ikisinin doldurdugu her anahtari `spdconv` da dolduruyor ve
sonra uzerine yaziyor. Yani sonuc `--pretrained spdconv` ile birebir
ayni. Zincir, senin mevcut `ablasion-main.ipynb` akisinla tutarli
olsun diye ve bu durumu GORUNUR kilmak icin duruyor.

ESKI CHECKPOINT'LER ICIN SHIM
-----------------------------
SPD-Conv'lu checkpoint'ler `ultralytics.nn.modules.spdconv` modulunu
ariyor (o zamanki dosya yolu). O dosya artik yok; `torch.load` bu yuzden
`No module named 'ultralytics.nn.modules.spdconv'` veriyor.
`install_legacy_shims()` bu isme projedeki `spdconv.SPDConv`'u bagliyor.
Guvenli: unpickle `__init__` cagirmaz, sadece `__dict__`'i geri yukler;
biz de bu nesnelerden yalnizca `state_dict()` okuyoruz, forward'lari hic
calismiyor.
"""

from __future__ import annotations

import os
import sys
import types

import torch

# Proje kokunden (ablasion-all) gorece checkpoint yollari.
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

CKPT = {
    "train-2": "baseline/runs/detect/train/weights/best.pt",
    "bifpn": "BIFPN/runs/detect/ablation_bifpn_v1/weights/best.pt",
    "spdconv": "SPD-Conv/runs/detect/ablation_spd_conv_v/weights/best.pt",
}

# Varsayilan zincir: zayiftan iyiye. Son eleman kazanir.
# `ablasion-main.ipynb`'deki zincirin aynisi.
DEFAULT_CHAIN = ["train-2", "bifpn", "spdconv"]

# BILEREK LISTEDE YOK: ablasion-all/runs/detect/spdconv_bifpn{,-2}
# Bunlar SPD-Conv + BiFPN'in BIRLESIK modeli, yani v2'nin yenmeye
# calistigi baseline'in ta kendisi (mAP50-95 0.6660 / 0.6666).
# Onlardan baslatmak "baseline'i yendik" iddiasini gecersiz kilar:
# model, olcmeye calistigimiz birlesimin 500 epoch'luk cozumunden
# baslamis olur. Zincir sadece TEK TEK bilesen ablasyonlarindan
# (train-2 = duz YOLO26n, bifpn = yalniz BiFPN, spdconv = yalniz
# SPD-Conv) besleniyor.


def resolve(key_or_path: str) -> str:
    """Kisa ad ya da dosya yolu -> mutlak yol."""
    if key_or_path in CKPT:
        return os.path.join(ROOT, CKPT[key_or_path])
    return os.path.abspath(key_or_path)


def install_legacy_shims():
    """Eski checkpoint'lerin unpickle sirasinda aradigi modul yollarini
    bugunku siniflara bagla. Idempotent."""
    if "ultralytics.nn.modules.spdconv" in sys.modules:
        return

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from spdconv import SPDConv

    shim = types.ModuleType("ultralytics.nn.modules.spdconv")
    shim.SPDConv = SPDConv
    sys.modules["ultralytics.nn.modules.spdconv"] = shim

    import ultralytics.nn.modules as um

    um.spdconv = shim


def _state_dict_of(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck.get("ema") or ck.get("model")
    return m.float().state_dict()


def load_chain(model, chain=None, verbose=True):
    """Zinciri sirayla `model`e yukler ve adim adim rapor basar.

    Args:
        model: `YOLO(...)` nesnesi (ultralytics.YOLO).
        chain: kisa ad / dosya yolu listesi. None -> DEFAULT_CHAIN.
        verbose: adim adim rapor.

    Returns:
        model (zincirlemeye uygun sekilde ayni nesne).
    """
    from ultralytics.utils.torch_utils import intersect_dicts

    install_legacy_shims()
    chain = list(chain if chain is not None else DEFAULT_CHAIN)

    dst = model.model.state_dict()
    total_params = sum(v.numel() for v in dst.values())
    filled: set[str] = set()  # simdiye kadar herhangi bir kaynaktan dolan anahtarlar

    if verbose:
        print(f"[pretrained] hedef: {len(dst)} tensor / {total_params/1e6:.2f}M parametre")
        print(f"{'kaynak':<26} {'eslesen':>9} {'YENI':>6} {'uzerine yazilan':>16} {'kumulatif param':>16}")
        print("-" * 78)

    for name in chain:
        path = resolve(name)
        if not os.path.exists(path):
            print(f"{name:<26} {'ATLANDI - dosya yok':>9}  {path}")
            continue
        try:
            src = _state_dict_of(path)
        except Exception as e:
            print(f"{name:<26} ATLANDI - yuklenemedi: {e}")
            continue

        inter = intersect_dicts(src, dst)
        new = set(inter) - filled
        overwritten = set(inter) & filled
        filled |= set(inter)

        model = model.load(path)

        if verbose:
            cum = sum(dst[k].numel() for k in filled)
            print(f"{name:<26} {len(inter):>9} {len(new):>6} {len(overwritten):>16} "
                  f"{100*cum/total_params:>15.1f}%")

    if verbose:
        miss = len(dst) - len(filled)
        print("-" * 78)
        print(f"toplam dolan: {len(filled)}/{len(dst)} tensor "
              f"({100*sum(dst[k].numel() for k in filled)/total_params:.1f}% parametre), "
              f"rastgele kalan: {miss} tensor")
        print("  (rastgele kalanlar: CAA katmanlari, P2 dali, 3-girisli fuzyon "
              "dugumleri ve Detect basi - bunlar v2'ye YENI, beklenen durum)")

    return model


def report(model, sources=None):
    """Hicbir sey yuklemeden, her kaynagin tek basina ne kadar
    ortusectigini yazdirir (zincir sirasi secmek icin)."""
    from ultralytics.utils.torch_utils import intersect_dicts

    install_legacy_shims()
    dst = model.model.state_dict()
    total = sum(v.numel() for v in dst.values())
    for name in (sources or list(CKPT)):
        path = resolve(name)
        if not os.path.exists(path):
            print(f"{name:<26} dosya yok: {path}")
            continue
        try:
            inter = intersect_dicts(_state_dict_of(path), dst)
            print(f"{name:<26} {len(inter):>4}/{len(dst)} tensor  "
                  f"{100*sum(v.numel() for v in inter.values())/total:>5.1f}% parametre")
        except Exception as e:
            print(f"{name:<26} yuklenemedi: {e}")
