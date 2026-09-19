# TV modu ve sıkıştırma onarımı v0

- **Tarih:** 2026-09-19
- **Durum:** onaylı (ölçüldü)
- **Kapsam:** Bu akşamki maç için laptoptan HDMI TV'ye 1080p 50 FPS canlı hat (A), TOD sıkıştırmasını onaran ilk model (B), modelin TV moduna takılması ve TOD'da kıyas (C). F27'nin ilk adımı. Önceki durum: [hat14](2026-09-16-hat14-mac-gunu-surumu.md), [F26](2026-09-16-f26-tod-yumusaklik.md).

## A. TV modu (`watch --tv`)

Ne yapıyor: TOD Chrome penceresi laptop panelinde (1920x1080, 144 Hz) yakalanır, TV'de (HDMI, 1920x1080, 50 Hz, birincil olmayan ekran) gösterilir. Ara kare yok, SR yok: çıkış hızı kaynak hızı (50), her tik gerçek kareye oturur. Ekran 50 Hz olduğu için sunucu vsync kilidinde (swap interval 1). Gecikmeli ses TV'nin HDMI çıkışından (`--audio-device NVIDIA`).

Kod:
- `upscaler/process.py`: `PassProcessor` (kare olduğu gibi, gerekirse 1080p'ye sığdırma), `RepairProcessor` (C için).
- `upscaler/present.py`: `pick_monitor(prefer="external")`. İki ekranın adı da "Generic PnP Monitor" olduğu için ad değil "birincil olmayan" kuralı. Ekran değişimi karşılaştırması artık konum ve Hz de içeriyor (aynı adlı iki ekranı ayırt etmek için).
- `upscaler/watch.py`: `--tv` ön ayarı (1080p doku, 50 FPS, `pass`, `external`), `--repair <ad>`.

| Koşu | Süre | Çıkış FPS | Geç tik | Geç sunum | Hattın eklediği A/V (p5 / p50 / p95) | Mod |
|---|---|---|---|---|---|---|
| Test klibi, ffplay, `runs/tv_kisa1` | 60 sn | 50,001 | 0 | %0,10 | -5,9 / 6,7 / 15,0 ms | lock, interval 1 |
| **Test klibi, ffplay, `runs/tv_10dk`** | **10 dk** | **50,001** | **0** | %0,03 | **-12,9 / -2,1 / 11,1 ms** | lock |
| **TOD, `runs/tv_tod2`** | **2 dk** | **50,0** | **0** | 0 | ölçülemez (DRM, test deseni yok) | lock |
| Masaüstü kısayolu, TOD, `runs/kisayol_ham` | 20 sn | 50,002 | 0 | | | lock |

- 10 dk: işlem p50 0,2 ms (sadece kare seçimi), uçtan uca p95 21 ms (swap vsync'i bekliyor), VRAM 1210 MB baştan sona sabit, ses sert atlama 0, eksik 0.
- Doğrusal eğim "görüntü gecikmesi -37 ms/saat" verdi ama dakika ortalamaları 1497-1507 ms arasında, eğilim yok. Eğimi ilk dakikadaki tek aykırı ölçüm (1721 ms) çekiyor. Kayma birikmiyor.
- Gerçek kare tik oranı %93-95: kalan tiklerde kare seçimi zaman oranından yapılıyor (TV'nin 50 Hz'i ile kaynağın 50'si tam aynı değil). `pass` harman yapmaz, en yakın kareyi verir.
- TOD: giriş 6017 benzersiz kare / 120 sn = 50,1 FPS, saat periyodu 20,000 ms, ses TV çıkışında (aygıt 10, gecikme 22 ms).
- TOD'da A/V ölçülemez; hattın eklediği A/V içerikten bağımsız, test klibiyle ölçülen değer geçerli. TV'nin kendi ses/görüntü işleme gecikmesi (TV ayarı) hattın dışında.

Kısayol: masaüstünde `Maç TV ham (upscaler).bat` = `watch.bat --tv --audio-device NVIDIA`. Eski `Maç izle (upscaler).bat` değişmedi.

## B. Onarım modeli v0 (1080p -> 1080p, tek kare)

**Ağ** (`upscaler/models/repair.py`, kendi kodumuz): kare 2x2 bloklara katlanır (12 kanal, 540x960), baş evrişim, N artık blok, kuyruk, geri açma, girdiye eklenir. Kuyruk sıfırla başlar (başta ağ = girdi).

**Hız taraması** (1080p TensorRT FP16, sessiz sistem, `runs/repair_bench.json`):

| Kanal x blok | Parametre | GPU p50 / p95 (ms) |
|---|---|---|
| 16 x 2 | 12,8 bin | 1,26 / 1,33 |
| 24 x 4 | 46,9 bin | 3,91 / 4,00 |
| 32 x 4 | 80,9 bin | 4,34 / 4,44 |
| **32 x 6** | **117,9 bin** | **5,77 / 5,89** |
| 48 x 4 | 176,7 bin | 6,94 / 7,03 |

Seçim 32 x 6: 20 ms bütçenin üçte birinden az, ısınma kısıtlamasına pay. Not: ilk tarama 4 paralel ffmpeg varken yapıldı ve aynı motoru 6,4 ms (p95 21) gösterdi; CPU yükü ve düşük GPU saat hızı ölçümü bozuyor.

**Veri** (sadece `data/clips` CC klipleri): `tools/make_pairs.py --target 1080`. Girdi: `tod_sim` zinciri + 4,8 Mbps H.264 50p (F26 rastgele bozulma: ön küçültme 0,5-1,0, unsharp 0-0,8). Hedef: **aynı zincirin kodlayıcıya girmeden önceki kare** (`tod_sim.clean_vf`), yani girdiyle tek fark H.264. Eğitim: 9 klip, 74,6 bin yama 128x128 (`rep_train_{a,b,c}`, c yarıda kesildi). Doğrulama: O3gD6n0zoik + 5bgF-5I2P_M (eğitimde yok), sabit ön küçültme 0,7 + unsharp 0,4, 3596 yama (`rep_val`).

**Eğitim** (`tools/train_repair.py`): L1 + genlik spektrumu, GAN yok, EMA 0,999, batch 16, laptop, maks 74 °C.
- İlk deneme (lr 4e-4, kuyruk önünde ReLU) 4000 adım boyunca girdiyi aynen verdi: özelliklerin tamamı negatife kaydı, ReLU hepsini sıfırladı ("ölü ReLU"). Düzeltme: kuyrukta aktivasyon yok, bloklarda LeakyReLU, lr 2e-4, gradyan kırpma 0,5.

| Model | Genlik ağırlığı | Adım | PSNR-Y girdi -> çıkış | hf girdi -> çıkış (hedef = 1) |
|---|---|---|---|---|
| rep_v0 | 1 | 40 bin (10 dk) | 35,895 -> **36,206 (+0,31 dB)** | 0,828 -> **0,817 (düşüyor: detay siliyor)** |
| rep_v0b | 8 | 16 bin (en iyi) | 35,895 -> **36,093 (+0,20 dB)** | 0,828 -> **0,857** |

hf: yamanın Y kanalında max(|fx|,|fy|) >= 0,5 Nyquist bandı enerjisi / hedefinki.

## C. TV moduna takma ve TOD ölçümü

`watch --tv --repair <ad>` (`RepairProcessor`, motor girdisi/çıktısı BGR 0..255 FP16, dönüşümler motorun içinde). `--split`: sol yarı onarımlı, sağ yarı ham.

| Koşu | Süre | Çıkış FPS | Geç tik | İşlem p50/p95 | VRAM (smi) |
|---|---|---|---|---|---|
| rep_v0, simüle klip ffplay, `runs/tv_klip_onarim5` | 5 dk | 50,001 | 0 | 5,8 / 8,1 ms | 1858 MB sabit |
| rep_v0 `--split`, simüle klip, `runs/tv_klip_split` | 60 sn | 50,001 | 0 | 5,9 / 8,3 ms | |
| **rep_v0, canlı TOD maçı, `runs/tv_tod_onarim5`** | **5 dk** | **50,001** | **0** | 5,9 / 6,4 ms | **1862 MB sabit** |
| **rep_v0b, canlı TOD maçı, `runs/tv_tod_onarim5_b`** | **5 dk** | **50,005** | **0** | 5,9 / 6,3 ms | **1936 MB sabit** |

TOD ilk denemede duraklatılmıştı, sonra "oynuyor" görünüp kare vermedi (ham modda da); Yiğit maçı açınca ölçüldü.

**Keskinlik** (`tools/sharpness_probe.py --repair`, kadrajın ortası, logo/tabela hariç, sadece sayı; spk = Nyquist bandı enerji oranı):

| Kaynak | spk_50_75 | spk_75_100 |
|---|---|---|
| Simüle klip hedefi (sıkıştırmasız, aynı zincir) | 0,208 | 0,036 |
| Simüle klip, ham (4,8 Mbps) | 0,153 | 0,029 |
| Simüle klip, rep_v0 | 0,132 | 0,009 |
| Simüle klip, rep_v0b | 0,147 | 0,010 |
| **Canlı TOD, ham** | 0,161 / 0,164 | 0,042 / 0,043 |
| **Canlı TOD, rep_v0** (2 dk, 135 kare) | **0,127 (-%21)** | 0,012 (-%71) |
| **Canlı TOD, rep_v0b** (2 dk, 235 kare) | **0,148 (-%10)** | 0,017 (-%61) |

**Sonuç (süslemeden):** v0 hattı bozmadan canlı çalışıyor (50 FPS, 6 ms). Ama ikisi de görüntüyü **keskinleştirmiyor, yumuşatıyor**: PSNR artışı sıkıştırma gürültüsünü ortalamaya çekmekten geliyor, ince bant (0,75-1 Nyquist) %60-70 düşüyor. Hedef bu bantta hamdan bile yüksek; model yönü tersine gidiyor. v0b daha az yumuşatıyor, bu yüzden kısayollar v0b'de. Doğrulamadaki hf (kare bant) v0b'de artarken spk (radyal oran) düşüyor: iki ölçü farklı bantlara bakıyor, rapora ikisi birden yazıldı.

Kısayollar (masaüstü): `Maç TV ham`, `Maç TV onarımlı` (rep_v0b), `Maç TV kıyas bölünmüş` (rep_v0b, S ile aç/kapa). Üçü de kısayoldan denendi (50,0 FPS, geç tik 0).

## Açık sorular

- Hangi kısayol? Gözle kıyas: bölünmüş kıyasta sol onarımlı, sağ ham. Sayılara göre ham daha keskin, onarımlı daha "temiz" ama yumuşak.
- L1 türü kayıplar belirsizlikte ortalamaya kaçıyor. Sonraki: çok kareli onarım (F27/F15, komşu karede silinmemiş detay), algısal/GAN son kat (F26), ya da kayıpta ince bandı hedefin enerjisine zorlamak.
- Paused/donmuş yayında TV modu "hazırlanıyor" ekranında kalıyor (saat kilitlenmiyor). Maçta yayın akarken sorun değil.
- TV'nin kendi görüntü işleme gecikmesi (oyun modu) hattın dışında; ses-görüntü uyumu gözle kontrol edilmeli.
- TV 50 Hz kilidi ölçüldü (`sunum_modu lock`, interval 1). TV ayarlarında "oyun modu" açık mı bilinmiyor.

## Kaynaklar

- Kayıtlar (repo dışı): `runs/tv_kisa1`, `runs/tv_10dk`, `runs/tv_tod2`, `runs/kisayol_*`, `runs/tv_klip_onarim5`, `runs/tv_klip_split`, `runs/tv_tod_onarim5`, `runs/tv_tod_onarim5_b`, `runs/train_rep_v0`, `runs/train_rep_v0b`, `runs/keskinlik_*`, `runs/repair_bench.json`.
- Kod: `upscaler/models/repair.py`, `upscaler/process.py` (Pass/RepairProcessor), `upscaler/present.py`, `upscaler/watch.py`, `tools/make_pairs.py`, `tools/tod_sim.py`, `tools/train_repair.py`, `tools/build_trt_repair.py`, `tools/sharpness_probe.py`.
