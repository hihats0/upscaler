# Hat 1.3 + 2.0 + 2.1: Canlı ölçüm, mini veri, TOD simülatörü

- **Tarih:** 2026-09-16
- **Durum:** onaylı (ölçüldü)
- **Kapsam:** CC BY 4K60 futbol klipleri (Hat 2.0), TOD benzeri bozulma (Hat 2.1), bozulmuş klibin oynatıcıda oynayıp **aynı canlı hattan** geçtiği ve çıkışın GT 4K karesiyle karşılaştırıldığı canlı puan (Hat 1.3), dört işlemcinin kıyası.

## Özet

- Canlı puan çalışıyor ve 60 FPS'i bozmuyor (puanlama açıkken geç tik en fazla %0,014).
- **1 GT kare kaydırınca puan 33,8 dB'den 23,3 dB'ye düşüyor:** ölçüm doğru kareyle hizalı.
- **TOD benzeri girdide hazır RT4KSR, bicubic'e göre neredeyse fark yaratmıyor:** iki segment ortalaması +0,10 dB PSNR, SSIM'de -0,004. Temiz girdide fark +2,9 dB idi (hat12 raporu). Model sıkıştırma hasarını tanımıyor. Hat 2.2'deki ince ayarın gerekçesi bu.
- **RIFE ara karesi doğrusal harmandan +4,6 dB iyi** (iki segment ortalaması).

## 1. Veri (Hat 2.0)

Arama: YouTube, "Creative Commons" + "4K" filtresi (`sp=EgQwAXAB`), yt-dlp ile sadece üst veri okundu.

**Eleme kuralı:** "CC" etiketli Messi/Ronaldo derlemeleri ve maç özetleri, yayın görüntülerinin yeniden yüklenmesidir. Yükleyen hak sahibi değilse CC etiketi geçersizdir; bunlar alınmadı. Oyun görüntüleri (FC 25, eFootball) de alınmadı (grafiklerin hakkı oyun şirketinde). Çocuk maçları tercih edilmedi.

**Seçilen kaynak:** Fabio di Mauro Official. İtalyan amatör lig (Eccellenza, Juniores) maçlarını kendisi 4K çeken bir video çekimcisi. Yayın açısına benzer geniş plan, skor grafiği, 3840x2160 **59,94 FPS**, CC BY. Çeşitlilik için Mylo'nun drone klibi (Bury vs Glossop, 59,94 FPS, CC BY) eklendi.

| Kimlik | Başlık | Yükleyen | Süre | Boyut | Kullanım |
|---|---|---|---|---|---|
| 63pDvVidZJA | FC Alfonsine vs USD Classe | Fabio di Mauro Official | 13:11 | 1,42 GB (AV1) | eğitim |
| u5w_du22wSQ | Masi Torello vs Argentana | Fabio di Mauro Official | 11:20 | 1,21 GB (AV1) | eğitim |
| 5bgF-5I2P_M | Argentana vs Virtus Castelfranco | Fabio di Mauro Official | 12:56 | 1,37 GB (AV1) | **doğrulama ve canlı puan (eğitime girmez)** |
| O3gD6n0zoik | Bury vs Glossop North End (drone) | Mylo | 1:02 | 0,19 GB (VP9) | doğrulama |

Toplam 4,19 GB (sınır 5 GB). Lisans hem yt-dlp `license` alanından hem tarayıcıda sayfa metninden ("Creative Commons Atıf lisansı (yeniden kullanılabilir)") doğrulandı. Kayıt: `data/clips/manifest.json` (URL, yükleyen, lisans, çözünürlük, FPS, atıf metni; `tools/clip_manifest.py`). `data/` git dışında. Sadece görüntü akışı indirildi.

## 2. TOD simülatörü (Hat 2.1)

`tools/tod_sim.py`: GT segmenti -> 1920x1080 50p H.264.

- 59,94 FPS GT kareleri sırayla tam 60 FPS sayılır (`setpts`, içerik %0,1 yavaşlar); yoksa 10 sn'de 0,6 karelik kayma birikir.
- `fps=50:round=near`: kaynak kare i = GT kare round(1,2·i) (seçim, ara kare yok). i % 5 == 0 olanlar GT ile aynı anda.
- lanczos küçültme, BT.709 sınırlı aralık (açık matris), x264 High, 2 geçiş, 4800 kbps (maxrate 6000, bufsize 9600), GOP 100, 3 B kare.
- Kare numarası şeridi (ust 16 satır, 24 hücre: 20 bit numara + 4 bit denetim). Puanlamada 4K'da üst 48 satır ve 8 px kenar dışarıda.

**Doğrulama (ölçüldü):**
- ffprobe: h264, High, 1920x1080, 50/1, 4800-4833 kbps (TOD ölçümü ~4,8 Mbps).
- `tools/check_sim_align.py`: 40 kare, GT round(1,2·i) eşleşmesi **40/40**; doğru kare ~35,5 dB, komşu kare ~22,6 dB (belirgin ayrım).
- Göz: geniş planda yayına benziyor; yakın planda çim dokusu H.264 ile yumuşamış, kenarlarda blok izi.

Sınır: TOD'un gerçek kaynağı büyük ihtimalle doğal 1080p kamera, burada 4K'dan küçültme. Bitrate ve kodek aynı, kodlayıcı ayarları (TOD'un x264/donanım kodlayıcısı, GOP) bilinmiyor.

## 3. Canlı puan (Hat 1.3)

`tools/live_score.py <meta.json> --proc X --seconds N [--shift K]`:
1. Simüle klip ffplay'de döngüde oynar (renk dönüşümü açıkça BT.709 sınırlı -> tam aralık).
2. `watch --score` aynı canlı hattı çalıştırır (WGC -> tampon -> işlemci -> GL sunucu), ses kapalı.
3. Yakalanan her karenin klip numarası şeritten okunur, tampona `meta` olarak girer.
4. Çıkış tikinde gösterilen içeriğin klip konumu `i + alpha`; GT karesi `g = 1,2·(i + alpha)`.
5. GT bankası: segment ffmpeg ile çözülür, Y (BT.709) RAM'de uint8 tutulur; `g % 6 ∈ {0, 3}` (gerçek kare tikleri ve alpha 0,5 ara kareleri), 10 sn için 200 kare, ~1,7 GB RAM.
6. Puan: Y kanalında PSNR ve SSIM (11x11 Gauss, sigma 1,5). Her GT karesi bir kez puanlanır, saniyede en fazla 2 kare, gerçek ve ara sırayla.

**Tasarımda bulunan sorunlar (ölçüldü):**

| Sorun | Belirti | Çözüm |
|---|---|---|
| Kırpılmış (bitişik olmayan) görüntüde evrişim | 4K SSIM 54-65 ms | `contiguous()`: 13 ms |
| GPU'da SSIM (ayrı akışta bile) | işlem p99 71 ms, geç tik %9 | Y GPU'da (0,4 ms), pinned belleğe asenkron kopya, SSIM **CPU'da** arka plan iş parçacığında (cv2, 2 iş parçacığı, ~350 ms/kare) |
| Çıkış fazı rastgele | tiklerin %80'i GT ızgarası dışında | Puan modunda hizalama ızgarasının fazı `-(klip no - saat sırası) mod 5` seçilir (zaman en fazla ±6,7 ms kayar, işlem yükü dağılımı aynı) |
| Klip başa sarınca | ~1,5 sn ızgara dışı | Faz gösterilen zamandaki kareden alınır. Kalan %10-20 ızgara dışı tik, ffplay'in sarış duraklamasından sonra saatin yeniden çapalanması (izlemeye etkisi yok, o anlar atlanır). |
| İlk çağrıda cuDNN seçimi | ~70 ms takılma | Artık CPU'da |

**Kaydırma testi** (`runs/score_kaydirma1`, fused-sr, 60 sn): kaydırmasız 33,84 dB / 0,887 -> 1 GT kare kaydırmalı **23,29 dB / 0,668** (gerçek kareler), ara kareler 30,61 -> 24,23 dB.

## 4. Kıyas (B4)

Her koşu 120 sn, doğrulama klibi 5bgF-5I2P_M, puanlar tüm koşularda **ortak** GT karelerinin ortalaması (`tools/score_table.py`). Hepsi 1920x1080 kaynak, 4K 60 FPS çıkış, laptop paneli 144 Hz (timer modu).

**Segment 1 (200. sn, 10 sn):**

| İşlemci | Büyütme | Ara kare | Çıkış FPS | Geç tik | İşlem p95 ms | Gerçek PSNR / SSIM (n=79) | Ara PSNR / SSIM (n=73) |
|---|---|---|---|---|---|---|---|
| baseline | bicubic | doğrusal harman | 60,00 | 0 | 7,0 | 33,70 / 0,8901 | 25,47 / 0,7184 |
| sr | RT4KSR | doğrusal harman | 60,00 | 0 | 9,2 | 33,84 / 0,8868 | 25,45 / 0,7137 |
| rife-flow-trt | bicubic | RIFE | 60,00 | 0 | 17,0 | 33,70 / 0,8901 | 30,36 / 0,8413 |
| **fused-sr** (watch varsayılanı) | RT4KSR | RIFE | 60,00 | 0 | 16,0 | 33,84 / 0,8868 | 30,36 / 0,8375 |

**Segment 2 (500. sn, 10 sn):**

| İşlemci | Büyütme | Ara kare | Çıkış FPS | Geç tik | İşlem p95 ms | Gerçek PSNR / SSIM (n=73) | Ara PSNR / SSIM (n=70) |
|---|---|---|---|---|---|---|---|
| baseline | bicubic | doğrusal harman | 60,00 | 0 | 6,8 | 33,67 / 0,9067 | 27,30 / 0,8014 |
| sr | RT4KSR | doğrusal harman | 60,00 | 0 | 8,7 | 33,73 / 0,9014 | 27,36 / 0,7981 |
| rife-flow-trt | bicubic | RIFE | 59,99 | %0,014 | 16,4 | 33,67 / 0,9067 | 31,54 / 0,8797 |
| **fused-sr** | RT4KSR | RIFE | 60,00 | %0,014 | 15,6 | 33,74 / 0,9014 | 31,56 / 0,8741 |

İki segmentin ortalaması (fused-sr'e göre): RT4KSR - bicubic = **+0,10 dB PSNR, -0,0043 SSIM** (gerçek kare); RIFE - harman = **+4,58 dB** (ara kare).

Gözlemler:
- Aynı büyütücüyü kullanan koşuların gerçek kare puanı birebir aynı: ölçüm tekrarlanabilir.
- RT4KSR'nin katkısı PSNR'de +0,1 dB mertebesinde, SSIM'de hafif olumsuz. Sıkıştırma hasarlı girdide keskinleştirme, blok ve halka izlerini de büyütüyor (tahmin).
- Ara karede RIFE belirleyici; SR ara karede de fark yaratmıyor.
- Ara kare puanı (alpha 0,5) gerçek kareden 2,2-3,5 dB düşük: 50->60 FPS'in bedeli.
- p95 işlem süresi puanlama açıkken 16-17 ms'ye çıkıyor (puanlamasız 14,6 ms); geç tik yine 0.

## Açık sorular

- Gerçek TOD'da sıkıştırma farklı olabilir (kodlayıcı, GOP, sahne). Simülatör bitrate'i TOD'un Wi-Fi toplamından ölçüldü.
- LPIPS ve zamansal titreme ölçülmedi (sadece PSNR/SSIM).
- Puanlama 144 Hz timer modunda yapıldı; 4K 60 Hz kilit modunda da çalışmalı (denenmedi).
- Klip sarışında saat yeniden çapalanıyor; uzun kliplerde (60 sn+) ızgara dışı oran düşer.

## Kaynaklar

- YouTube sayfaları: https://www.youtube.com/watch?v=63pDvVidZJA, u5w_du22wSQ, 5bgF-5I2P_M (Fabio di Mauro Official), O3gD6n0zoik (Mylo); lisans CC BY (sayfada doğrulandı, 2026-09-16).
- Wang vd. 2004, "Image Quality Assessment: From Error Visibility to Structural Similarity" (SSIM).
- Kod: `tools/tod_sim.py`, `tools/check_sim_align.py`, `tools/live_score.py`, `tools/score_table.py`, `tools/clip_manifest.py`, `upscaler/score.py`, `upscaler/schedule.py` (faz), `upscaler/ring.py` (meta_offset).
- Kayıtlar: `runs/k200_*`, `runs/k500_*`, `runs/score_kaydirma1`, `runs/score_deneme1..4` (repo dışı).
