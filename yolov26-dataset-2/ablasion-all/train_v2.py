"""
train_v2.py
===========
spdconv_bifpn_p2 **v2** egitim scripti - dataset-2 (ke-project/yolo-tb1l6).

Calistirma (ablasion-all klasorunden):
    python train_v2.py                 # KANITLANMIS recete + v2 mimari, s scale
    python train_v2.py --scale n       # hafif ablasyon
    python train_v2.py --batch 6       # VRAM sikisirsa

==========================================================================
!! ONCE OKU - ILK v2 KOSUSU (0.5718) NEDEN BASELINE'IN (0.666) ALTINDA KALDI
==========================================================================
`spdconv-bifpn-p2/runs/detect/v2_s_bifpn3in_caa` -> mAP50-95 **0.5718**
`ablasion-all/runs/detect/spdconv_bifpn`         -> mAP50-95 **0.6660**

Ikisi de AYNI veri seti. (Eski kosularin `../../../YOLO.v1i.yolo26`
yolu simdi `dataset2/yolo26/YOLO.v1i.yolo26` - etiket dosyalari
byte-byte AYNI, md5 ile dogrulandi. Onceki README'de "farkli dataset"
denmisti, bu YANLISTI.)

Fark mimariden degil, RECETEDEN geliyordu. Ilk v2 kosusunda ayni anda
6 degisken degistirilmisti; en agiri:

  1) OPTIMIZER/LR: SGD lr0=0.01 -> AdamW lr0=0.001.
     Bu veri setinde dusuk lr'nin cokturdugu SENIN KENDI kosularinda
     zaten kanitlanmis (ayni YAML, ayni optimizer, sadece lr0 degisiyor):

       spdconv_bifpn.yaml + SGD
         lr0=0.01    -> 0.6215 / 0.6235 / 0.6251   (3 kosu, cok tutarli)
         lr0=0.0003  -> 0.5346 / 0.5403
         lr0=0.0002  -> 0.4983

       yolo26n_repc3k2_bifpn_consam.yaml + SGD
         lr0=0.01    -> 0.6123
         lr0=0.001   -> 0.4557 / 0.3793

     Yani lr0'i 0.01'den dusurmek bu kurulumda 0.16-0.23 mAP50-95
     goturuyor. AdamW 0.001 onerisi bu kaniti gormeden verildi - hata.
     Not: v2 mimarisi lr0=0.001'de **0.5718** aldi; senin ayni lr'deki
     en iyi kosun 0.4557'ydi. Yani mimari muhtemelen ISE YARIYOR,
     receteyle sabote edilmis.

  2) ULTRALYTICS SURUMU: butun eski kosular **8.4.118** ile yapilmis
     (checkpoint 'version' alani ve args.yaml'daki `dis/dlam/dlog/dgrad/
     cls_remap/cls_pw/channels_last` anahtarlari bunu gosteriyor).
     Kurulu surum **8.4.14** ve bu anahtarlar onda YOK. v2 kosusu bu
     yuzden hicbir eski kosuyla karsilastirilabilir degil.
     (Not: `distill_model: null` idi, yani distilasyon acik degildi -
     fark distilasyondan gelmiyor; ama 104 patch surum arasi YOLO26
     egitim tarafinda baska degisiklikler icerebilir.)
     -> Once `pip install ultralytics==8.4.118` yap, sonra
        `nn/modules/inn_modules_v2.py` yamasini geri uygula. Detay:
        README_v2.md, "Surum eslestirme".

  3) flipud 0.0 -> 0.5, scale 0.5 -> 0.6, translate 0.1 -> 0.15,
     close_mosaic 10 -> 30, warmup 3 -> 5, cos_lr False -> True.
     Hicbiri bu veri setinde test edilmemisti; hepsi ayni anda acildi.

  4) Kosu 500 yerine 415. epoch'ta kesildi ve mAP hala tirmaniyordu
     (400 -> 0.5705, 415 -> 0.5718).

  5) `s` degil `n` yaml'i kosuldu (4.92M parametre).

  6) BASLANGIC AGIRLIGI COCO'ydu. Senin kendi SPD-Conv/BiFPN
     checkpoint'lerin bu veri setinde egitilmis VE v2'nin cok daha
     buyuk bir kismini dolduruyor (olculdu, `pretrained_v2.report()`):

       COCO yolo26n.pt              353/1118 tensor  %20.2 parametre
       runs/detect/train-2          353/1118 tensor  %20.2 parametre
       BIFPN/ablation_bifpn_v1      353/1118 tensor  %20.2 parametre
       SPD-Conv/ablation_spd_conv_v 358/1118 tensor  %58.7 parametre

     Fark backbone'daki SPDConv katmanlarindan (model.0/1/3/5/7):
     SPDConv conv agirligi c2 x (4*c1) x k x k, stok Conv ise
     c2 x c1 x k x k -> shape uyusmuyor, COCO o katmanlarin sadece
     BatchNorm'unu doldurabiliyor. Bunlar da modelin en agir katmanlari.
     -> `--pretrained chain` (train-2 -> bifpn -> spdconv) VARSAYILAN.

     BIRLESIK `spdconv_bifpn{,-2}` checkpoint'leri BILEREK zincire
     alinmadi: onlar v2'nin yenmeye calistigi baseline'in kendisi
     (0.6660 / 0.6666). Onlardan baslatmak, olcmeye calistigimiz
     birlesimin 500 epoch'luk cozumunden baslamak demek olurdu.

DOGRU YONTEM: baseline'a gore SADECE MIMARIYI degistir. Bu scriptin
varsayilani artik `--recipe proven`, yani `spdconv_bifpn` kosusunun
hiperparametrelerinin birebir aynisi.

--------------------------------------------------------------------------
Wise-IoU hala VARSAYILAN KAPALI. Gerekce (bu tespit gecerli):
`wise-iou` kosularinda `train/box_loss` 1. epoch'ta 295-344. BboxLoss'ta
`box_loss = ((1-iou)*w).sum()/scores_sum * 7.5` ve
`sum(w) <= scores_sum` oldugu icin bu sayi 15'i asamaz -> o kosular
clamp'siz wiseiou.py surumuyle yapilmis ve kayip olcegi patlamis.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATA = "../../../dataset2/yolo26/YOLO.v1i.yolo26/data.yaml"

# `ablasion-all/runs/detect/spdconv_bifpn/args.yaml` ile BIREBIR ayni.
# Bu kosu mAP50 0.9074 / mAP50-95 0.6660 aldi; karsilastirma tabani bu.
RECIPE_PROVEN = dict(
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    cos_lr=False,
    warmup_epochs=3.0,
    weight_decay=0.0005,
    momentum=0.937,
    nbs=64,
    box=7.5, cls=0.5, dfl=1.5,
    scale=0.5,
    translate=0.1,
    mosaic=1.0,
    close_mosaic=10,
    fliplr=0.5,
    flipud=0.0,
    degrees=0.0,
    shear=0.0,
    perspective=0.0,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    mixup=0.0,
    cutmix=0.0,
)

# Ilk v2 kosusunda kullanilan (ve geri tepen) ayarlar. Sadece TEK TEK,
# proven baseline'i aldiktan SONRA denenmeli.
RECIPE_EXPERIMENTAL = dict(
    RECIPE_PROVEN,
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    warmup_epochs=5.0,
    scale=0.6,
    translate=0.15,
    close_mosaic=30,
    flipud=0.5,
)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=["n", "s"], default="s")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--recipe", choices=["proven", "experimental"], default="proven",
                   help="proven = spdconv_bifpn (0.6660) kosusunun hiperparametreleri")
    p.add_argument("--set", nargs="*", default=[], metavar="K=V",
                   help="tek tek hiperparametre ez, orn: --set flipud=0.5 scale=0.6")
    p.add_argument("--pretrained", default="chain",
                   help="'chain' -> kendi SPD-Conv/BiFPN best.pt zinciri (varsayilan, "
                        "%%58.7 parametre kapsar), 'coco' -> yolo26{scale}.pt (%%20.2), "
                        "'none' -> sifirdan, ya da kisa ad / .pt yolu "
                        "(train-2 | bifpn | spdconv)")
    p.add_argument("--chain", nargs="*", default=None,
                   help="zinciri elle ver, orn: --chain bifpn spdconv")
    p.add_argument("--no-bifpn", action="store_true", help="BiFPN ablasyonu: duz Concat")
    p.add_argument("--wiou", action="store_true", help="Wise-IoU ablasyonu (varsayilan KAPALI)")
    p.add_argument("--cfg", default=None, help="YAML'i elle ver (mimari ablasyonu icin)")
    p.add_argument("--name", default=None)
    return p


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    return v


def main():
    args = build_argparser().parse_args()

    # ---------- 0) Surum uyarisi ----------
    import ultralytics

    if ultralytics.__version__ != "8.4.118":
        print("=" * 74)
        print(f"!! UYARI: ultralytics {ultralytics.__version__} kurulu, eski kosularin")
        print("!! hepsi 8.4.118 ile yapildi. Bu kosu onlarla KARSILASTIRILABILIR DEGIL.")
        print("!! Karsilastirma yapacaksan once:  pip install ultralytics==8.4.118")
        print("!! ve inn_modules_v2.py yamasini geri uygula (README_v2.md).")
        print("=" * 74)

    # ---------- 1) Custom modulleri kaydet (YOLO(...) SATIRINDAN ONCE) ----------
    import ultralytics.nn.tasks as tasks
    from spdconv import SPDConv

    tasks.SPDConv = SPDConv

    from modules_v2 import register_v2

    register_v2(bifpn=not args.no_bifpn)

    # ---------- 2) Kayip fonksiyonu ----------
    if args.wiou:
        from wiseiou import patch_bbox_iou_with_wiou

        patch_bbox_iou_with_wiou(monotonous=False)
        print("!! WIoU acik. 1. epoch sonunda results.csv'de train/box_loss'u KONTROL ET:")
        print("!! 2-5 araliginda olmali. 20'nin ustundeyse kosuyu durdur.")

    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    cfg = args.cfg or f"yolo26{args.scale}-spdconv-bifpn-p2-v2.yaml"

    model = YOLO(cfg)

    # ---------- 3) On-egitimli agirlik ----------
    # Bu veri setinde zaten egitilmis kendi checkpoint'lerin, COCO'dan cok
    # daha fazlasini dolduruyor: SPD-Conv'lu bir kaynak %58.7 parametre,
    # COCO / SPD-Conv'suz kaynaklar %20.2. Fark backbone'daki SPDConv
    # katmanlarindan (model.0/1/3/5/7) geliyor - SPDConv'un conv agirligi
    # c2 x (4*c1) x k x k, stok Conv ise c2 x c1 x k x k, yani shape
    # uyusmuyor ve COCO o katmanlarin sadece BatchNorm'unu doldurabiliyor.
    if args.pretrained == "none":
        print("[init] sifirdan egitim")
    elif args.pretrained == "chain":
        from pretrained_v2 import load_chain

        model = load_chain(model, args.chain)
    elif args.pretrained == "coco":
        w = f"yolo26{args.scale}.pt"
        model = model.load(w)
        print(f"[init] COCO agirligi yuklendi: {w}")
    else:
        from pretrained_v2 import load_chain

        model = load_chain(model, [args.pretrained])

    # ---------- 4) Hiperparametreler ----------
    hyp = dict(RECIPE_PROVEN if args.recipe == "proven" else RECIPE_EXPERIMENTAL)
    for kv in args.set:
        k, _, v = kv.partition("=")
        if k not in hyp:
            print(f"[uyari] '{k}' recetede yok, yine de gonderiliyor")
        hyp[k] = _coerce(v)

    name = args.name or (
        f"v2_{args.scale}_{args.recipe}"
        f"{'_nobifpn' if args.no_bifpn else ''}"
        f"{'_wiou' if args.wiou else ''}"
    )

    print(f"[recete] {args.recipe}: optimizer={hyp['optimizer']} lr0={hyp['lr0']} "
          f"cos_lr={hyp['cos_lr']} scale={hyp['scale']} flipud={hyp['flipud']} "
          f"close_mosaic={hyp['close_mosaic']}")

    # ---------- 5) Egitim ----------
    model.train(
        data=DATA,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=4,
        name=name,
        patience=100,
        plots=True,
        val=True,
        **hyp,
    )

    # ---------- 6) Test seti ----------
    best = model.trainer.best
    print(f"\n[test] {best}")
    res = YOLO(str(best)).val(data=DATA, split="test", imgsz=args.imgsz, device=device)
    d = res.results_dict
    print("\n=== TEST METRIKLERI ===")
    for k, label in [
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall   "),
        ("metrics/mAP50(B)", "mAP@0.5  "),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95"),
    ]:
        v = d.get(k)
        print(f"{label}: {v * 100:.2f}%" if v is not None else f"{label}: yok")


if __name__ == "__main__":
    main()
