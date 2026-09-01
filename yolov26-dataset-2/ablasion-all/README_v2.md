# spdconv_bifpn_p2 **v2** — dataset-2 (roboflow `ke-project/yolo-tb1l6`)

---

# 0. POST-MORTEM: ilk v2 kosusu neden 0.5718 geldi

| kosu | mAP50 | mAP50-95 | ultralytics | optimizer / lr0 | epoch |
|---|---|---|---|---|---|
| `spdconv_bifpn-2` (baseline) | 0.9105 | **0.6666** | 8.4.118 | SGD 0.01 | 499 |
| `spdconv_bifpn` (baseline) | 0.9074 | **0.6660** | 8.4.118 | SGD 0.01 | 500 |
| `bifpn+spd-covn+p2+wise-iou-2` | 0.5862 | 0.4259 | 8.4.118 | SGD 0.01 | 500 |
| **`v2_s_bifpn3in_caa` (ilk v2)** | 0.8458 | **0.5718** | **8.4.14** | **AdamW 0.001** | **415** |

## 0.1 Once bir duzeltme: veri seti AYNI

Onceki notumda eski kosularin (`../../../YOLO.v1i.yolo26/data.yaml`)
farkli bir veri setinde oldugunu yazmistim. **Yanlisti.** O yol
tasinmis; etiketler byte-byte ayni:

```
train  root-v11: n=2747 box=8688 md5=2a74bc7f7880
train  dataset2: n=2747 box=8688 md5=2a74bc7f7880
valid  root-v11: n=568  box=1907 md5=cb5247c03a5a
valid  dataset2: n=568  box=1907 md5=cb5247c03a5a
test   root-v11: n=279  box=1023 md5=38efd00e3cb1
test   dataset2: n=279  box=1023 md5=38efd00e3cb1
```

Yani **0.6660 gecerli bir karsilastirma tabani** ve v2 kosusu onun
0.094 altinda kaldi. Asagidaki 5 sebep bu farki aciklıyor.

## 0.2 Ana sebep: optimizer/LR degisikligi

Bu veri setinde dusuk lr'nin cokturdugu **senin kendi kosularinda**
zaten kanitlanmis. Ayni YAML, ayni optimizer, sadece `lr0` degisiyor:

| model | lr0 | mAP50 | mAP50-95 |
|---|---|---|---|
| `spdconv_bifpn.yaml` + SGD | **0.01** | 0.8683 / 0.8741 / 0.8681 | **0.6215 / 0.6235 / 0.6251** |
| | 0.0003 | 0.7956 / 0.8054 | 0.5346 / 0.5403 |
| | 0.0002 | 0.7578 | 0.4983 |
| `yolo26n_repc3k2_bifpn_consam.yaml` + SGD | **0.01** | 0.8731 | **0.6123** |
| | 0.001 | 0.7548 / 0.6489 | 0.4557 / 0.3793 |

`lr0`'i 0.01'den dusurmek bu kurulumda **0.16–0.23 mAP50-95** goturuyor.
AdamW `lr0=0.001` onerisini bu kaniti gormeden verdim — bu benim hatam;
`repc3k2/` ve `spdconv_bifpn_eca/` klasorlerini taramadan oneri yaptim.

**Onemli ayrinti:** v2 mimarisi `lr0=0.001`'de **0.5718** aldi. Senin ayni
lr'deki en iyi kosun **0.4557**'ydi. Yani mimari muhtemelen calisiyor,
recete sabote etti.

## 0.3 Ikinci sebep: ultralytics surumu farkli

Butun eski kosular **8.4.118**, v2 kosusu **8.4.14** ile yapildi.
Kanit — `train-2/best.pt` icindeki `version` alani `8.4.118`, ve eski
`args.yaml`'larda su anahtarlar var, yenide **yok**:

```
dis: 6.0   dlam: 1.0   dlog: 1.0   dgrad: 0.5
cls_remap: True   cls_pw: 0.0   channels_last: False
```

Bunlar 8.4.118'in `cfg/default.yaml`'inda var, 8.4.14'te hic yok
(dogrulandi: wheel indirilip iceri bakildi).

Dikkat: `dis` distilasyon **agirligi**, ama eski kosularda
`distill_model: null` — yani distilasyon acik degildi. Fark
distilasyondan gelmiyor. Yine de 104 patch surum arasi YOLO26 egitim
tarafinda baska degisiklikler icerebilir ve **v2 kosusu hicbir eski
kosuyla karsilastirilabilir degil.**

### Surum eslestirme (karsilastirma yapacaksan sart)

```bash
pip install ultralytics==8.4.118
```

> **Bu, elle eklenmis `ultralytics/nn/modules/inn_modules_v2.py` yamasini
> siler.** Yama `SPDConv, CoTAttention, AAM, FEM, RepCSP, AFF_Add2`
> tanimliyor ve `nn/modules/__init__.py` + `nn/tasks.py` bunlari import
> ediyor. Kurulumdan sonra bu uc dosyayi geri koyman gerekiyor
> (`nn/modules/__init__.py.bak_inn_yolo26_v2` yedegi duruyor).
>
> `modules_v2.py` artik bu yamaya **bagimli degil**: host modul olarak
> upstream'de garanti olan `Focus` ve `C3Ghost` isimlerini kullaniyor
> (ilk surumde `FEM` kullaniyordu — o upstream'de yok, kurulum silince
> kaybolurdu). Ama `spdconv_bifpn_p2.yaml` disindaki eski YAML'larin
> (`AAM`, `FEM`, `RepCSP`, `AFF_Add2` kullananlar) yamaya ihtiyaci var.

## 0.4 Ucuncu sebep: ayni anda 6 degisken degistirildi

| ayar | baseline | ilk v2 kosusu |
|---|---|---|
| optimizer / lr0 | SGD 0.01 | AdamW 0.001 |
| ultralytics | 8.4.118 | 8.4.14 |
| `cos_lr` | False | True |
| `warmup_epochs` | 3 | 5 |
| `flipud` | 0.0 | 0.5 |
| `scale` | 0.5 | 0.6 |
| `translate` | 0.1 | 0.15 |
| `close_mosaic` | 10 | 30 |
| mimari | `bifpn-spdconv.yaml` | v2 (P2 + CAA + 3-giris BiFPN) |

Bu haliyle hangi degisiklik ne yapti anlasilamaz. Ablasyon kurali:
**baseline'a gore sadece mimariyi degistir.** `train_v2.py`'nin
varsayilani artik bu (`--recipe proven`).

## 0.5 Dorduncu ve besinci sebep

* Kosu **415/500** epoch'ta kesildi ve mAP hala tirmaniyordu
  (400 → 0.5705, 415 → 0.5718). Kucuk etki (~+0.005) ama var.
* `s` degil **`n`** yaml'i kosuldu (`yolo26n-spdconv-bifpn-p2-v2.yaml`,
  4.92M parametre). `s` (19.6M) hic denenmedi.

## 0.6 Simdi ne yapmali

```bash
pip install ultralytics==8.4.118          # + inn_modules_v2 yamasini geri koy
cd ablasion-all
python train_v2.py --scale n              # baseline'a gore SADECE mimari degisir
```

Beklenti: `n` scale + proven recete, `spdconv_bifpn`'in 0.6660'i ile
dogrudan karsilastirilabilir olur. Ondan sonra `--scale s` ve tek tek
`--set flipud=0.5` gibi denemeler anlamli olur.

---

# 1. Dosyalar

| dosya | ne |
|---|---|
| `modules_v2.py` | `WeightedConcatN` (N girisli BiFPN), `CAA`, `C3k2CAA`, `register_v2()` |
| `yolo26s-spdconv-bifpn-p2-v2.yaml` | v2 mimari, `s` scale (19.6M param) |
| `yolo26n-spdconv-bifpn-p2-v2.yaml` | ayni mimari, `n` scale (5.0M param) |
| `train_v2.py` | egitim scripti; `--recipe proven` varsayilan |
| `../spdconv-bifpn-p2/training_v2.ipynb` | notebook surumu |

---

# 2. Wise-IoU tespiti (bu gecerli)

`runs/detect/` icindeki `results.csv` dosyalarindan `train/box_loss`:

| epoch | `wise-iou-2` | `wise-iou-4` | saglikli kosu (`bifpn+spd-covn+p2`) |
|---|---|---|---|
| 1 | **295.1** | **344.5** | – |
| 4 | – | – | **2.33** |
| 500 | 35.3 | 33.9 | – |

`BboxLoss.forward` icinde
`box_loss = ((1-iou)*w).sum() / target_scores_sum * box_gain(7.5)`
ve `sum(w) <= target_scores_sum` oldugu icin makul bir `(1-iou)` ile bu
sayi **15'i asamaz**. 295 / 344 sinirsiz buyume demek — o kosular
`wiseiou.py`'nin clamp'siz surumuyle yapilmis (dosyanin kendi
"BUG FIX HISTORY" notu da bunu soyluyor).

Ayrica `ablasion-main.ipynb` 2. hucrede `patch_concat_to_bifpn()` hemen
ardindan `patch_concat_to_hybrid()` cagriliyor — **ikincisi birincisini
eziyor**, yani "BiFPN" etiketli kosular hybrid ile calismis. Ustelik
`HybridBiAFFPNConcat` alt katmanlarini ilk forward'da yaratiyor (lazy
build); `model.train()` modeli YAML'dan yeniden kurup optimizer'i o anki
parametre listesinden olusturdugu icin o parametreler **hic
egitilmemis** olabilir.

**v2'de:** WIoU varsayilan kapali, tek fusion patch'i var, lazy build yok.

---

# 3. Veriden turetilen tasarim kararlari

Etiket istatistigi (8688 / 1907 / 1023 kutu):

| | train | valid | test |
|---|---|---|---|
| kucuk (<32px) | %30.5 | **%38.9** | **%38.3** |
| orta (32–96px) | %15.6 | %20.0 | %20.8 |
| buyuk (>96px) | %53.9 | %41.1 | %40.9 |
| sinif | crack 6156 / dent 2532 | 1297 / 610 | 688 / 335 |

en-boy orani (w/h), train: `1% → 0.06`, `75% → 1.58`, `95% → 4.25`,
`99% → 39.16`.

1. **valid/test, train'den daha kucuk nesneli** → P2 dali olcum
   dagiliminin ~%40'ini ilgilendiriyor; P2 kaliyor, kanali 4 katina cikti.
2. **Kutular cizgisel** (crack) → `CAA` (1×11 + 11×1 depthwise serit dikkat).
3. **Rotasyon zararli:** 39:1 oranli bir catlagi 10° dondurunce
   eksen-hizali kutunun genisligi `L·sin10° + w·cos10° ≈ 0.199L`
   olur (onceki `0.026L`) → **~7.6 kat**. `degrees=0.0`.
4. Goruntuler zaten 640×640 → `imgsz` buyutmek yeni bilgi getirmiyor.

> `scale=0.6` ve `flipud=0.5` de bu analizden turemisti, ama **ikisi de
> bu veri setinde hic test edilmedi** ve proven receteden sapiyorlar.
> Bu yuzden varsayilana alinmadilar; `--set scale=0.6` ile tek tek dene.

---

# 4. v2 mimari degisiklikleri

| | v1 | v2 |
|---|---|---|
| scale | n (0.25) | s (0.50) — `n` varyanti da var |
| parametre | 4.70M | 19.59M (`s`) / 4.92M (`n`) |
| BiFPN dugum arity | hepsi 2 girisli | kat. 21 ve 24 **3 girisli** (backbone kisayolu) |
| fuzyon agirlik normu | `sum(w)=1` | **`sum(w)=n`** (olcek koruyan) |
| pozitiflik | `relu` | **`softplus`** |
| serit dikkat | yok | **CAA ×3** (P3-td, P2-out, P3-out) |
| neck derinligi | `2` → depth 0.5 → **1 blok** | `4` → **2 blok** (P3/P4) |
| P2 detect kanali | 128 nominal | 256 nominal |

**3 girisli fuzyon:** v1'de tum dugumler 2 girisliydi — bu PANet'in
agirlikli hali, gercek BiFPN degil. BiFPN'de bottom-up yolundaki ara
seviyeler `bottom-up + top-down + backbone kisayolu` alir:

```
21: [20 (P2'den asagi), 16 (P3 top-down), 4 (P3 backbone)]
24: [23 (P3'ten asagi), 13 (P4 top-down), 6 (P4 backbone)]
```

**Olcek-koruyan normalizasyon:** klasik BiFPN agirlikli *toplam* icin
`sum(w)=1` kullanir. Buradaki islem *concat* oldugu icin `sum(w)=1` her
dali `1/n` ile carpar ve ciktinin genligi ~n kat kuculur. `sum(w)=n`
ortalama agirligi 1'de tutar; ogrenilebilir oranlar korunur.

**Neck derinligi:** `depth=0.50` oldugu icin YAML'daki `2`,
`max(round(2*0.5),1) = 1`'e dusuyordu — v1'de neck'teki her C3k2 tek
bloktu. `4` yazinca gercekten 2 blok oluyor.

## 4.1 Parser kaydi — site-packages'a dokunmadan

`parse_model` icindeki `base_modules`/`repeat_modules` frozenset'leri
fonksiyon govdesinde literal isimlerle kuruluyor; yeni bir isim eklemek
mumkun degil (yeni isim `else: c2 = ch[f]` dalina duser, `c1` hic
verilmez). `register_v2()` bunun yerine `parse_model`'i saran bir
wrapper kuruyor: YAML'daki `CAA`/`C3k2CAA` isimlerini parser'in tanidigi
bos host isimlere cevirip o isimlere bizim siniflari bagliyor.

Host secimi calisma aninda yapiliyor, upstream'de garanti olan
adaylardan:

| custom modul | aday host'lar | frozenset gereksinimi |
|---|---|---|
| `CAA` | `Focus` → `SPP` → `GhostBottleneck` → `FEM` | base, repeat DEGIL → `(c1, c2, *args)` |
| `C3k2CAA` | `C3Ghost` → `C3x` → `RepC3` → `C3TR` | base VE repeat → `(c1, c2, n, *args)` |

> `C3k2CAA.__init__` imzasi `C3k2` ile birebir ayni olmali:
> `(c1, c2, n, c3k, e, attn, g, shortcut)`. 6. konumdaki `attn`
> atlanirsa YAML'daki 4. arguman `g=True` olarak duser ve `conv2d` patlar.

---

# 5. Olculen maliyet (RTX 4060 8GB)

`yolo26s-...-v2.yaml`, batch 8, imgsz 640:
**peak VRAM 6.9 GB**, **~130 s/epoch**, cikarim 9.2 ms/goruntu.
500 epoch ≈ 18 saat. `n` scale belirgin daha ucuz.

Fuzyon agirliklarinin gercekten egitildigi 2 epoch sonrasi `last.pt`'den
dogrulandi (init 0.5413'ten ayrilmislar, kullanilmayan 4. slot yerinde):

```
kat.12 giris=2 [0.999, 1.001]      kat.21 giris=3 [1.009, 0.991, 1.000]
kat.15 giris=2 [0.990, 1.010]      kat.24 giris=3 [1.014, 0.985, 1.000]
kat.18 giris=2 [0.978, 1.022]      kat.27 giris=2 [1.003, 0.997]
```

---

# 6. Kullanim

```bash
cd ablasion-all
python train_v2.py --scale n                      # baseline ile karsilastirilabilir
python train_v2.py --scale s                      # buyuk model
python train_v2.py --scale s --no-bifpn           # BiFPN ablasyonu
python train_v2.py --scale s --pretrained none    # COCO transfer ablasyonu
python train_v2.py --scale n --set flipud=0.5     # tek degisken dene
python train_v2.py --recipe experimental          # ilk v2 kosusunun ayarlari
```

**Karsilastirma tabani:** `spdconv_bifpn` → mAP50 **0.9074** /
mAP50-95 **0.6660** (valid, SGD 0.01, ultralytics 8.4.118).

**Saglik kontrolu:** 1. epoch'tan sonra `train/box_loss` 2–5 araliginda
olmali. 20'nin ustundeyse kayip olcegi patlamis, kosuyu durdur.

---

# 7. Notlar

* Veri seti karisik (detect + segment): 8688 kutunun 3905'inde poligon
  da var; ultralytics uyari verip poligonlari atiyor — detect icin sorun degil.
* `train_v2.py` Windows'ta `if __name__ == "__main__":` korumasiyla
  yazildi; onsuz `workers>0` ile DataLoader spawn'i
  `RuntimeError: An attempt has been made to start a new process...` veriyor.
