"""
register_innyolo.py
--------------------
INN-YOLO ozel modullerini (Conv_SAM, C3k2_Rep, BiFPN_Add2, BiFPN_Add3)
ultralytics kutuphanesine RUNTIME'DA, kaynak kodu elle degistirmeden
tanitir (monkey-patch).

KULLANIM (baska her seyden once, ultralytics'i kullanmaya baslamadan once
calistirin):

    import register_innyolo          # <-- bunu import edin
    from ultralytics import YOLO

    model = YOLO("yolo26-innyolo.yaml")
    model.train(data="coco8.yaml", epochs=10, imgsz=640)

Bu dosyayi inn_yolo_modules.py ile AYNI klasore koyun (oradan import ediyor).

NASIL CALISIR
-------------
1) inn_yolo_modules.py icindeki 4 ozel sinifi hem `ultralytics.nn.modules`
   paketine, hem de `ultralytics.nn.tasks` modulunun global namespace'ine
   ekler. (yaml'daki "Conv_SAM", "C3k2_Rep" gibi string isimler,
   ultralytics.nn.tasks.parse_model icinde `globals()[m]` ile cozuluyor;
   bu adim o cozumlemenin calismasini saglar.)

2) `ultralytics.nn.tasks.parse_model` fonksiyonunun kendisini, bizim 4 ozel
   modulumuzu ANLAYAN bagimsiz bir implementasyonla degistirir. Bu yeni
   fonksiyon:
     - Conv, Conv_SAM, C3k2, C3k2_Rep, SPPF, C2PSA, Concat, nn.Upsample,
       Detect ve BiFPN_Add2/BiFPN_Add3 icin TAM destek verir (bizim
       yolo26-innyolo.yaml'da kullanilan modul kumesi budur).
     - Bu kumede olmayan herhangi bir modul turuyle karsilasirsa (ornegin
       Segment, Pose, Classify, farkli bir custom blok vb.), otomatik
       olarak ORIJINAL ultralytics parse_model'ini o katman icin cagirir
       (fallback) - yani standart yaml'lariniz calismaya devam eder.

    Not: DetectionModel.__init__ icinde "parse_model(...)" bare-name olarak
    cagrildigi icin (yani modulun global namespace'inde runtime'da aranir),
    tasks.parse_model attribute'unu degistirmek yeterlidir; DetectionModel'i
    yeniden tanimlamaya gerek yoktur.
"""

import ast
import math
import contextlib

import torch
import torch.nn as nn

import ultralytics.nn.modules as _um
import ultralytics.nn.tasks as _tasks
from ultralytics.utils import LOGGER

from modules import Conv_SAM, C3k2_Rep, BiFPN_Add2, BiFPN_Add3, RepVGGBlock

# ---------------------------------------------------------------------------
# 1) Ozel siniflari ultralytics namespace'lerine enjekte et
# ---------------------------------------------------------------------------
for _name, _cls in (
    ("Conv_SAM", Conv_SAM),
    ("C3k2_Rep", C3k2_Rep),
    ("BiFPN_Add2", BiFPN_Add2),
    ("BiFPN_Add3", BiFPN_Add3),
    ("RepVGGBlock", RepVGGBlock),
):
    setattr(_um, _name, _cls)          # ultralytics.nn.modules.Conv_SAM ...
    setattr(_tasks, _name, _cls)       # tasks.py globals()[...] cozumlemesi icin

_CUSTOM_MODULES = {Conv_SAM, C3k2_Rep, BiFPN_Add2, BiFPN_Add3}

_ORIGINAL_PARSE_MODEL = _tasks.parse_model  # fallback icin sakla


def _make_divisible(x, divisor=8):
    return math.ceil(x / divisor) * divisor


def _innyolo_parse_model(d, ch, verbose=True):
    """INN-YOLO ozel moduller + standart YOLO26 moduller icin parse_model.

    Bilinmeyen bir modul turu gorursen orijinal ultralytics parse_model'e
    devreder (tum d sozlugunu tekrar oradan gecirir) - bu yuzden karma
    (custom + standart-baska-head) yaml'lar da guvenle calisir.
    """
    Conv, C3k2, SPPF, C2PSA, Concat, Detect = (
        _um.Conv, _um.C3k2, _um.SPPF, _um.C2PSA, _um.Concat, _um.Detect,
    )

    layer_defs = d["backbone"] + d["head"]
    module_names = []
    for f, n, m, args in layer_defs:
        module_names.append(m)

    # Eger tanidigimiz modul kumesinin disinda bir sey varsa (Segment, Pose,
    # Classify, baska bir custom blok...), guvenli tarafta kal: orijinal
    # parse_model'e tamamen devret. (Bizim ozel moduller zaten globals()'a
    # eklendi, yani orijinal fonksiyon da bunlari CONV/C3K2 gibi ozel
    # olarak degil ama en azindan isim olarak tanir; kanal hesaplamasi
    # yanlis olabilecegi icin biz burada devreye giriyoruz.)
    _known_tokens = {
        "Conv", "Conv_SAM", "C3k2", "C3k2_Rep", "SPPF", "C2PSA",
        "Concat", "nn.Upsample", "Detect",
        "BiFPN_Add2", "BiFPN_Add3",
    }
    if not set(module_names).issubset(_known_tokens):
        LOGGER.info(
            "innyolo: taninmayan modul(ler) bulundu, orijinal ultralytics "
            "parse_model'e fallback yapiliyor."
        )
        return _ORIGINAL_PARSE_MODEL(d, ch, verbose)

    # --- Standart YOLO26 olcekleme parametreleri ---
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING innyolo: model olcegi verilmedi, scale='{scale}' varsayiliyor.")
        depth, width, max_channels = scales[scale]
    if act:
        Conv.default_act = eval(act)

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]

    # Kanal/olcek uygulanan ve nc'ye esitse dokunulmayan moduller
    channel_modules = {Conv, Conv_SAM, C3k2, C3k2_Rep, SPPF, C2PSA}
    # 'n' (tekrar sayisi) modulun kendi ic yapisina gomulen moduller
    repeat_modules = {C3k2, C3k2_Rep, C2PSA}

    for i, (f, n, m_name, args) in enumerate(layer_defs):
        m = getattr(torch.nn, m_name[3:]) if "nn." in m_name else globals().get(m_name) or getattr(_tasks, m_name)
        args = list(args)
        _local_vars = {"nc": nc, "depth": depth, "width": width, "max_channels": max_channels}
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError, SyntaxError):
                    args[j] = _local_vars[a] if a in _local_vars else ast.literal_eval(a)

        n_ = n = max(round(n * depth), 1) if n > 1 else n

        if m in channel_modules:
            c1, c2_ = ch[f], args[0]
            if c2_ != nc:
                c2_ = _make_divisible(min(c2_, max_channels) * width, 8)
            args = [c1, c2_, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
            c2 = c2_

        elif m is Concat:
            c2 = sum(ch[x] for x in f)

        elif m in (BiFPN_Add2, BiFPN_Add3):
            c1 = ch[f[0]]
            c2_ = args[0]
            if c2_ != nc:
                c2_ = _make_divisible(min(c2_, max_channels) * width, 8)
            args = [c1, c2_]
            c2 = c2_

        elif m is Detect:
            args = [nc, [ch[x] for x in f]]
            print(f"[DEBUG] Detect layer {i}: f={f}, nc={nc!r}, ch_list={[ch[x] for x in f]!r}")
            c2 = None

        else:  # nn.Upsample vb. gecis moduller
            c2 = ch[f] if isinstance(f, int) else ch[f[-1]]

        m_ = m(*args) if n == 1 else nn.Sequential(*(m(*args) for _ in range(n)))
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:>10}  {t:<45}{str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


_tasks.parse_model = _innyolo_parse_model

LOGGER.info(
    "innyolo: Conv_SAM, C3k2_Rep, BiFPN_Add2, BiFPN_Add3 registered "
    "(ultralytics.nn.tasks.parse_model patched)."
)