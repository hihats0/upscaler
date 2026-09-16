# Hat 1.4: Maç günü sürümü

- **Tarih:** 2026-09-16
- **Durum:** onaylı (ölçüldü). TOD'da 60-100 dk canlı sınav ve 4K ekran kontrolü bekliyor.
- **Kapsam:** tek komutla (`python -m upscaler watch` ya da masaüstü kısayolu) TOD canlı maçını 4K 60 FPS, senkron sesle, uzun süre kesintisiz izlemek. Önceki durum: [2026-09-15 hat12](2026-09-15-hat12-sr-modeli.md).

## Özet

| Ölçüt | Hedef | Sonuç | Kaynak |
|---|---|---|---|
| Hattın eklediği A/V (60 dk) | \|A/V\| <= 40 ms | medyan 5,4 ms, p5-p95 -8..+16 ms | `runs/av_60dk` |
| A/V kayması (60 dk) | <= 20 ms/saat | -0,7 ms/saat | `runs/av_60dk` |
| Çıkış FPS (60 dk) | >= 59,9 | 59,994 | `runs/av_60dk` |
| Geç tik | < %0,5 | %0,009 (20 / 215857) | `runs/av_60dk` |
| Çökme | 0 | 0 (4 uzun ve 3 dayanıklılık koşusu) | |
| VRAM büyümesi | < 100 MB | 0 MB (60 dk), +16 MB (dayanıklılık) | `runs/av_60dk`, `runs/robust3` |
| RSS büyümesi | makul | yok (60 dk'da 1280 -> 1163 MB) | `runs/av_60dk` |
| Uzun koşu `--dump-timing --info` | 60 dk | 59,999 FPS, geç tik 5 (%0,002), hata 0 | `runs/uzun60_info` |
| Dayanıklılık | çökme 0 | 6/6 kontrol geçti | `runs/robust3` |
| 144 Hz'de kilit modu | çalışsın | 48 FPS, interval 3, geç sunum 0 | `runs/lock48` |
| TOD 60-100 dk | canlı maç | **yapılmadı** (TOD açık değildi) | |

Uzun koşular kendi test klibimizle (`data/av_test_1080p50.mp4`, 1080p50 + her 2 sn flaş ve bip) ffplay penceresinde, aynı canlı hattan (WGC -> RIFE akışı + RT4KSR TensorRT -> GL sunucu, ProcTap ses -> gecikmeli çalma) yapıldı.

## 1. A/V kayması: ölçüm hatasıydı

**Belirti (2026-09-15):** 3 dk'lık koşuda çıkış A/V 36,9 -> 58,6 ms (eğim 680 ms/saat).

**İnceleme:** Sonda ham olay zamanlarını (`av_ham.json`), yakalama saati durumunu (c0, fs) ve sunum gecikmesi EMA'sını kaydedecek şekilde genişletildi. İkinci 3 dk koşusu (`runs/av_3dk_b`):

- Girdideki A/V (ffplay'in kendi farkı) t≈110 sn'de 63 -> 42 ms **basamak** yaptı. Klip 60 sn'de bir başa sarıyor, ffplay her sarışta kendi senkronunu yeniden kuruyor.
- Çıkış A/V aynı basamağı izledi; doğrusal eğim bunu "kayma" diye raporladı (bu koşuda -600 ms/saat, ters yön).
- Hattın gecikmeleri sabit: görüntü (ham flaş -> swap) 1509,5 ms, ses (tahmini yakalama -> DAC) 1512,3 ms. Yakalama saati c0 3 dk'da en fazla 1,8 ms oynadı.
- Eski özetteki "görüntü gecikmesi -491 ms" ayrı bir eşleştirme hatasıydı: flaşlar 2 sn arayla, gecikme 1,5 sn; "en yakın" flaş -0,5 sn'deki.

**Düzeltme (ölçüm):**
- Eşleştirme beklenen gecikme etrafında yapılır.
- Asıl ölçüt, olay başına **hattın eklediği A/V**: `(ses çıkış - ses giriş) - (görüntü çıkış - görüntü giriş)`. Kaynağın kendi A/V farkından bağımsızdır.
- Görüntü ve ses gecikmesinin kayması ayrı raporlanır.

**60 dk sonucu:** hattın eklediği A/V medyan 5,4 ms, kayma -0,7 ms/saat. Görüntü gecikmesi kayması 0,5, ses -0,2 ms/saat. Ses sert atlama 0, eksik blok 0, kilit hatası p99 3,2 ms. Ham çıkış A/V (40 -> 64 ms) ffplay'in kendi farkını taşıyor (kaynak 17 ms/saat, çıkış 16 ms/saat).

Not: Gerçek maçta Chrome'un kendi A/V farkı ölçülemez (DRM'li içerikte test deseni yok). Hat bu farkı olduğu gibi taşır ve ~+5 ms ekler.

## 2. Dayanıklılık sınavı

`tools/robustness_test.py`, watch çalışırken sırayla: 8 sn donma (süreç askıda), 6 sn simge durumu, 1280x720, 2560x1440, 1920x1080 geri, oynatıcı kapanıp 6 sn sonra yeniden açılma, ekran değişimi benzetimi (sunucu yeniden açılır). Sonunda otomatik kontroller.

| Koşu | Bulgu | Düzeltme |
|---|---|---|
| robust1 | Çökme 0 ama VRAM 2,6 -> 5,1 GB. Test aracı DPI farkında değildi: %125 ölçekte "1920x1080 geri" pencereyi 2400x1350 yaptı; tampon 150 kareyi bu boyutta (2 GB) tuttu. | Test aracı `dpi_aware()`. **Tampon 1920x1080'i aşan kaynağı GPU'da küçültüp saklar** (`GpuFrameRing.MAX_HW`). Sebep: 4K ekranda tam ekran Chrome (3840x2160) tamponu ~5 GB yapardı. |
| robust2 | 2560x1440 artık `motor 1920x1080` yolunda. Ama VRAM +1 GB: tampon yeniden ayrılırken işlemdeki seçim eski slotları tutuyor, `empty_cache` bırakamıyordu. | Ana döngü, nesil değişmiş seçimi bıraktıktan sonra önbelleği bir kez boşaltır. |
| robust3 | **6/6 kontrol geçti:** çıkış kodu 0, işlem hatası 0, hata olayı 0, VRAM +16 MB, son 30 sn >= 59,5 FPS, yol sonunda `motor 1920x1080`. | |

Bilinen davranışlar (hata değil):
- Boyut değişiminde tampon sıfırlanır; 1,5 sn gecikme dolana kadar son kare ekranda kalır (o 10 sn penceresinde ~51 FPS görünür).
- Ekran değişiminde sunucu yeniden açılır: ~1,2 sn'de ~75 geç tik (tek seferlik).
- Donma ve simge durumunda son kare tutulur, ara kare üretilmez (`schedule.GAP_PERIODS`).

## 3. Kilit modu (vsync)

Laptop paneli 144 Hz: 60 FPS için tam bölen yok, hat kendi zamanlayıcısıyla (timer) akar. Kilit modu `--out-fps 48` ile (144/3) denendi (`runs/lock48`, 90 sn):

- mod lock, interval 3, çıkış 48,03 FPS, geç tik 0, geç sunum 0
- sunum aralığı p1/p50/p99: 16,5 / 20,8 / 25,2 ms (ideal 20,83)
- sunum gecikmesi (swap bitişi - hedef vsync) p50 0,01, p99 4,4 ms
- swap vsync beklediği için sunum süresi p50 7,3 ms
- hattın eklediği A/V 1,5 ms

4K 60 Hz ekranda aynı yol interval 1 ile çalışacak (ekran yok, denenmedi).

## 4. Uzun koşu (`--dump-timing --info`)

`runs/uzun60_info` (60 dk, bilgi katmanı açık, kare ve tik zamanları kaydediliyor, A/V ölçümü açık). TOD açık olmadığı için hedefteki TOD koşusunun yerine yapıldı.

- Çıkış 59,999 FPS (215874 kare), geç tik 5 (%0,002; 1 tanesi ilk 10 sn'de), işlem hatası 0, tampon sıfırlama 0
- işlem p50/p95/p99: 12,97 / 14,6 / 18,5 ms; sunum dahil uçtan uca: 14,2 / 16,2 / 20,7 ms; sunum: 1,05 / 2,1 / 3,0 ms; geç sunum %0,35
- İlk kare süreç başlangıcından 6,4 sn
- Hattın eklediği A/V medyan 4,9 ms (p5-p95 -8..+16), kayma -1,0 ms/saat; ses sert atlama 1, eksik 0
- VRAM (nvidia-smi) 2595 MB baştan sona sabit
- RSS 1292 -> tepe 1303 -> 1279 MB. `--dump-timing` her tik için satır biriktirir (~1,4 MB/dk, sadece hata ayıklama bayrağı); normal izlemede büyüme yok (`runs/av_60dk`).
- GPU ort. 85,8 °C, maks 88 °C, ~83 W; örneklerin %47'sinde ısıl yavaşlama bayrağı. Bu sürede geç tik yok: bütçede pay var (işlem p95 14,6 ms / 16,7 ms).

## 5. Bellek

- `PresentStats` her karede 4 float ekleyen sınırsız listeler tutuyordu (~0,5 MB/dk): sınırlı deque yapıldı.
- 60 dk'da RSS büyümesi yok. Önceki "~3 MB/dk" gözlemi 3 dk'lık koşulardaki ısınma artışıydı (ilk dakikalarda ayırıcılar büyüyor).
- tracemalloc (25 kare iz) RSS'i dakikada ~180 MB şişirir. Sızıntı ararken RSS'e değil Python nesne farkına bakılmalı.

## 6. Isı

60 dk'da GPU ortalaması 85,5 °C, en fazla 88 °C, güç ~86 W. 10 sn'lik örneklerin %9'unda sürücü ısıl yavaşlama bayrağı (0x20) vardı. Geç tik kümeleri (620-750 sn, 1850-1860 sn) bu örneklere denk gelmedi. Öneri: maçta laptop soğutucu altlıkta ve prizde olsun.

## 4K ekran gelince: Yiğit için 5 dk kontrol listesi

1. 4K ekranı HDMI 2.1 ile bağla. Windows Ayarlar > Ekran: 4K ekranı seç, **3840x2160** ve **60 Hz** olsun. Laptop ekranı "genişlet" modunda kalsın.
2. Chrome'u **laptop ekranında** aç, yayını tam ekran yap (F11). Chrome'u 4K ekrana taşıma: sunucu o ekranı kaplar, Chrome örtülünce çizmeyi bırakır.
3. Masaüstündeki **Maç izle (upscaler)** kısayoluna çift tıkla. 10 sn içinde 4K ekranda görüntü gelmeli. Gelmezse `runs/` altındaki en yeni klasörün `events.jsonl` dosyasını agent'a göster.
4. **I** tuşuna bas. Bilgi katmanında şunlara bak:
   - `ekran 3840x2160 60 Hz, mod lock` (timer yazıyorsa ekran 60 Hz değil)
   - `cikis 60.0 fps`, `gec tik 0` ya da çok az
   - `yol motor 1920x1080` (Chrome tam ekran değilse "sigdirma" yazar)
   - `ses: çalıyor`, hata birkaç ms
5. 1 dk izle. Top hızlı giderken titreme ya da yırtılma var mı, ses ağızla uyumlu mu? **S** tuşu ekranı ikiye böler (sol SR, sağ bicubic): fark görünüyor mu? Çift ses duyarsan `notes/soru-kuyrugu.md`'deki yönlendirmeyi yap.
6. **Esc** ile çık. Agent'a "4K denendi" de. Agent `runs/` kaydından sunum ve A/V sayılarını çıkarır.

## Açık sorular

- TOD'da 60-100 dk canlı sınav (yayın açıkken): `watch --dump-timing --info --run-name tod_uzun`.
- 4K 60 Hz ekranda kilit modu ve 4K tarama maliyeti (sunum p95 şu an 1080p panelde 1,9 ms).
- Chrome'un kendi A/V farkı: kullanıcı gözüyle kontrol; gerekirse `--av-offset-ms`.
- Isıl yavaşlama uzun maçta artar mı (90 dk + devre arası)?

## Kaynaklar

- Kayıtlar: `runs/av_3dk`, `runs/av_3dk_b`, `runs/av_60dk`, `runs/robust1..3`, `runs/lock48`, `runs/uzun60_info` (repo dışı).
- Kod: `upscaler/watch.py` (AvProbe, döngü), `upscaler/ring.py` (MAX_HW), `upscaler/present.py`, `tools/av_sync_test.py`, `tools/robustness_test.py`.
- Microsoft, Windows.Graphics.Capture ve WASAPI process loopback belgeleri (önceki raporlarda).
