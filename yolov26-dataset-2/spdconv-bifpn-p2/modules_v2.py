"""
modules_v2.py
=============
spdconv_bifpn_p2 mimarisinin v2 (gelistirilmis) surumu icin gereken TUM
custom moduller + parser kaydi tek dosyada.

NEDEN BU DOSYA VAR (dataset-2 = ke-project/yolo-tb1l6, crack+dent):
Etiket istatistigi cikarildi (train 8688 kutu / valid 1907 / test 1023):

  * Olcek dagilimi BIMODAL:
        train : %30.5 kucuk (<32px) | %15.6 orta | %53.9 buyuk (>96px)
        valid : %38.9 kucuk         | %20.0 orta | %41.1 buyuk
        test  : %38.3 kucuk         | %20.8 orta | %40.9 buyuk
    -> valid/test train'e gore DAHA COK kucuk nesne iceriyor. Yani P2
       dali (stride 4) sadece "guzel olur" degil, uzerinden olcum
       yapilan dagilimin ~%40'ini ilgilendiriyor. P2 KALIYOR ve
       guclendiriliyor.

  * En-boy orani asiri uc: w/h yuzdelikleri  1% -> 0.06,  95% -> 4.25,
    99% -> 39.16. Yani "crack" sinifi neredeyse cizgisel. Kare (kxk)
    receptive field bu yapiya kotu oturuyor.
    -> CAA (serit/strip konvolusyonlu dikkat, 1x11 + 11x1) eklendi.

  * Kutu/goruntu 3.16, bos goruntu yok, sinif dengesi 2.4:1
    (crack:dent) -> agir sinif-dengesi mudahalesi gerekmiyor.

ICERIK
------
1) WeightedConcatN : BiFPN "fast normalized fusion" - N GIRISLI surum.
   Eski bifpn.py sadece 2 giris destekliyordu; gercek BiFPN'de ara
   seviyeler 3 girislidir (bottom-up + top-down + backbone kisayolu).
   Ayrica olcek-koruyan normalizasyon kullanir (asagida aciklandi).

2) CAA : Context Anchor Attention (PKINet, CVPR 2024) - serit
   konvolusyonlu, cok genis ama ucuz (depthwise) anizotropik dikkat.

3) C3k2CAA : C3k2 + arkasinda CAA. Neck'te secili bloklarda kullanilir.

4) register_v2() : Bunlari Ultralytics parser'ina KAYIT EDER.
   parse_model icindeki `base_modules` / `repeat_modules` frozenset'leri
   fonksiyon govdesinde literal isimlerle kuruldugu icin YENI bir isim
   eklemek mumkun degil (yeni isim `else: c2 = ch[f]` dalina duser;
   kanal hesabi bozulur ve c1 modulun __init__'ine hic verilmez).
   Cozum: parse_model'i saran ince bir wrapper, YAML sozlugundeki
   "CAA"/"C3k2CAA" isimlerini parser'in ZATEN tanidigi bos "host"
   isimlere (FEM, C3Ghost) cevirir; o host isimlere de bizim
   siniflarimiz baglanir. Boylece
     - YAML okunabilir kalir (gercek isimler yazilir),
     - kanal hesabi (c1, c2, width scaling, repeat) DOGRU calisir,
     - site-packages icindeki ultralytics DOSYALARI DEGISTIRILMEZ,
     - lazy-build (ilk forward'da katman yaratma) HIC KULLANILMAZ.

   ! LAZY-BUILD NEDEN YASAK: Ultralytics `model.train()` cagrildiginda
   modeli YAML'dan YENIDEN kurar ve optimizer'i o anki parametre
   listesinden olusturur. Ilk forward'da yaratilan parametreler o
   listede OLMAZ -> egitim boyunca hic guncellenmez, init degerinde
   donar. hybrid_fusion.py tam olarak bu tuzaga dusuyor; v2'de
   kullanilmiyor.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- #
# Yerel Conv (ultralytics'in Conv'unu import ETMIYORUZ: spdconv.py'deki
# patch_conv_to_spdconv() o ismi degistirebiliyor, buradaki 1x1'ler
# yanlislikla SPD davranisi almasin)
# --------------------------------------------------------------------- #
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ConvBNAct(nn.Module):
    """Conv-BN-SiLU (yerel kopya)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# --------------------------------------------------------------------- #
# 1) BiFPN weighted concat - N girisli
# --------------------------------------------------------------------- #
class WeightedConcatN(nn.Module):
    """BiFPN fast-normalized-fusion tarzi agirlikli concat (N giris).

    Eski bifpn.py'ye gore 3 fark:

    (a) N GIRIS: `len(x) != 2` artik hata degil. Gercek BiFPN'de P3/P4
        bottom-up dugumleri 3 girislidir (bottom-up + top-down +
        backbone kisayolu). Agirlik vektoru sabit boyutta (n_max)
        tutulur ki state_dict her dugumde ayni shape'e sahip olsun ve
        checkpoint yuklemesi dugum arity'sinden bagimsiz calissin.

    (b) OLCEK-KORUYAN NORMALIZASYON: klasik BiFPN, agirlikli TOPLAM icin
        sum(w)=1 kullanir. Burada yapilan islem toplam degil CONCAT
        oldugu icin sum(w)=1 her dali ortalama 1/n ile carpar ve fuzyon
        ciktisinin genligi ~n kat KUCULUR (n=3'te ~3x). Bunu telafi
        etmek icin w, sum(w)=n olacak sekilde olceklenir; ortalama
        agirlik 1 olur ve egitim basinda sinyal genligi plain Concat ile
        ayni kalir. Ogrenilebilir ORANLAR aynen korunur.

    (c) Pozitiflik icin relu yerine softplus: relu(w) tum dallarda 0
        olursa gradyan tamamen oluyor (olu bolge). softplus'ta boyle bir
        bolge yok.
    """

    def __init__(self, dimension=1, n_max=4, eps=1e-4):
        super().__init__()
        self.d = dimension
        self.n_max = n_max
        self.eps = eps
        # softplus(0.5413) ~= 1.0 -> baslangicta tum dallar esit agirlikli
        self.w = nn.Parameter(torch.full((n_max,), 0.5413, dtype=torch.float32))

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"WeightedConcatN list/tuple bekler, gelen: {type(x)}")
        n = len(x)
        if n > self.n_max:
            raise ValueError(f"WeightedConcatN n_max={self.n_max} ile kuruldu, {n} giris geldi.")

        w = F.softplus(self.w[:n].float())
        w = w * (n / (w.sum() + self.eps))  # sum(w) = n -> ortalama agirlik 1
        w = w.to(x[0].dtype)

        ref = x[0].shape[2:]
        outs = []
        for i, t in enumerate(x):
            if t.shape[2:] != ref:  # guvenlik: tek sayili feature-map boyutlarina karsi
                t = F.interpolate(t, size=ref, mode="nearest")
            outs.append(t * w[i])
        return torch.cat(outs, self.d)

    def extra_repr(self):
        return f"n_max={self.n_max}"


# --------------------------------------------------------------------- #
# 2) CAA - Context Anchor Attention (serit konvolusyonlu dikkat)
# --------------------------------------------------------------------- #
class CAA(nn.Module):
    """Context Anchor Attention (Cai et al., PKINet, CVPR 2024).

    Akis:
        AvgPool7x7 -> 1x1 (kanal daralt) -> DWConv(1,k) -> DWConv(k,1)
        -> 1x1 (kanal geri ac) -> Sigmoid -> giris ile carp

    Neden bu veri seti icin: ayrik (1,k) ve (k,1) depthwise
    konvolusyonlar ~kxk'lik bir baglami sadece ~2k maliyetle tarar VE
    yatay/dikey yonleri AYRI ogrenir. Etiket istatistiginde w/h orani
    95. yuzdelikte 4.25, 99. yuzdelikte 39 -> crack kutulari neredeyse
    cizgisel. Kare receptive field bu kutularda bol miktarda alakasiz
    arka plan topluyor; serit konvolusyon catlagin GITTIGI yon boyunca
    baglam toplar.

    Maliyet: c=128, k=11, ratio=4 icin ~ (128*32) + (32*11)*2 + (32*128)
    ~= 9k parametre. Ihmal edilebilir.
    """

    def __init__(self, c1, c2=None, k=11, ratio=4, pool=7):
        super().__init__()
        c2 = c1 if c2 is None else c2
        ch = max(8, c1 // ratio)
        self.pool = nn.AvgPool2d(pool, stride=1, padding=pool // 2)
        self.reduce = ConvBNAct(c1, ch, 1)
        self.h_conv = nn.Conv2d(ch, ch, (1, k), padding=(0, k // 2), groups=ch, bias=False)
        self.v_conv = nn.Conv2d(ch, ch, (k, 1), padding=(k // 2, 0), groups=ch, bias=False)
        self.expand = ConvBNAct(ch, c1, 1, act=False)
        self.gate = nn.Sigmoid()
        # Kanal sayisi degisiyorsa (YAML'da farkli c2 yazildiysa) hizala
        self.proj = nn.Identity() if c2 == c1 else ConvBNAct(c1, c2, 1)

    def forward(self, x):
        a = self.gate(self.expand(self.v_conv(self.h_conv(self.reduce(self.pool(x))))))
        return self.proj(x * a)


# --------------------------------------------------------------------- #
# 3) C3k2 + CAA
# --------------------------------------------------------------------- #
def _c3k2_base():
    from ultralytics.nn.modules.block import C3k2

    return C3k2


class C3k2CAA(_c3k2_base()):
    """Standart C3k2 blogu + cikisinda CAA serit-dikkat.

    Imza C3k2 ile BIREBIR ayni olmali:
        (c1, c2, n, c3k, e, attn, g, shortcut)
    -- ozellikle 6. konumdaki `attn` atlanirsa YAML'daki 4. arguman
    yanlis parametreye duser (g=True gibi) ve conv2d patlar.
    Bu sayede parser'in `repeat_modules` yolundan (args'a n enjekte
    edilerek) sorunsuz kurulur ve COCO on-egitimli agirliklarin C3k2
    alt-anahtarlari isim/shape olarak eslesmeye devam eder (CAA
    anahtarlari yeni oldugu icin rastgele baslar).
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True, caa_k=11):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.caa = CAA(c2, c2, k=caa_k)

    def forward(self, x):
        return self.caa(super().forward(x))


# --------------------------------------------------------------------- #
# 4) Parser kaydi
# --------------------------------------------------------------------- #
# YAML'daki okunabilir isim -> parser'in tanidigi "host" isim.
#  * FEM      : base_modules icinde, repeat_modules DISINDA -> (c1, c2, *args)
#  * C3Ghost  : base_modules VE repeat_modules icinde       -> (c1, c2, n, *args)
# Bu iki host modul bu projenin hicbir YAML'inda kullanilmiyor.
_ALIAS = {
    "CAA": "FEM",
    "C3k2CAA": "C3Ghost",
}

_PARSE_PATCHED = False


def register_v2(bifpn: bool = True, verbose: bool = True):
    """Custom modulleri Ultralytics parser'ina kaydeder.

    YOLO("....yaml") satirindan ONCE, bir kez cagrilmali.

    Args:
        bifpn: True ise YAML'daki tum `Concat`'lar WeightedConcatN ile
            degistirilir (BiFPN agirlikli fuzyon). False ise duz Concat
            kalir - ablasyon icin.
    """
    global _PARSE_PATCHED
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    # --- host isimlere siniflarimizi bagla ---
    tasks.FEM = CAA
    tasks.C3Ghost = C3k2CAA

    # --- BiFPN agirlikli concat ---
    if bifpn:
        tasks.Concat = WeightedConcatN
        modules.Concat = WeightedConcatN
        if hasattr(modules, "conv") and hasattr(modules.conv, "Concat"):
            modules.conv.Concat = WeightedConcatN

    # --- YAML isimlerini host isimlere ceviren parse_model wrapper'i ---
    if not _PARSE_PATCHED:
        _orig_parse_model = tasks.parse_model

        def parse_model(d, ch, verbose=True):
            d = copy.deepcopy(d)
            for section in ("backbone", "head"):
                for layer in d.get(section, []):
                    if isinstance(layer[2], str) and layer[2] in _ALIAS:
                        layer[2] = _ALIAS[layer[2]]
            return _orig_parse_model(d, ch, verbose)

        tasks.parse_model = parse_model
        _PARSE_PATCHED = True

    if verbose:
        print("[modules_v2] kayit tamam:")
        print("   CAA      -> Context Anchor Attention (1x11 + 11x1 strip DW conv)")
        print("   C3k2CAA  -> C3k2 + CAA")
        if bifpn:
            print("   Concat   -> WeightedConcatN (BiFPN, N girisli, olcek-koruyan)")
        else:
            print("   Concat   -> (degistirilmedi, duz Concat)")
