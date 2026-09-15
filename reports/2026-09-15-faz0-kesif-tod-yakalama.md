# Faz 0 keşif: TOD yakalanabiliyor mu, kaynak gerçekte ne?

- Tarih: 2026-09-15 (12:30-12:42 arası ölçüm)
- Durum: **Ölçüm (onaylı veri).** Tek kanal (beIN Sports 1), tek oturum, tek ağ.
- Kapsam: TOD penceresinin Windows'ta yakalanıp yakalanamadığı, kaynağın gerçek çözünürlüğü, kare hızı, codec, DRM seviyesi, bitrate, ekran ve GPU durumu.
- Kurallar: DRM atlatılmadı. Hiçbir kare diske yazılmadı, kareler bellekte sayıya çevrildi (parlaklık, std, kare farkı). Betikler scratchpad'de, sadece istatistik basar.
- Bu rapor şu bilgiyi **düzeltir:** "TOD tarayıcıda 720p 25 FPS" ([ikinci araştırma](2026-09-15-tod-4070-mobile-sigdirma-arastirmasi.md), forum kaynaklı). Ölçülen değer 1080p 50 FPS.

## 1. Sonuç özeti

| Soru | Cevap | Nasıl ölçüldü |
|---|---|---|
| TOD Chrome'da yakalanabiliyor mu? | **Evet, siyah değil.** | GDI BitBlt (mss) ve PrintWindow(PW_RENDERFULLCONTENT), video alanında ort. parlaklık ~79, std ~47, siyah piksel %7 |
| Çözünürlük | **1920x1080** (ABR, açılışta 640x360) | `chrome://media-internals` |
| Kare hızı | **50 FPS, gerçek 50p** | 144 Hz ekranda hızlı örnekleme: 5 sn'de 249 yeni kare (50,0), 6 sn'de 301 (50,3) |
| Video codec | H.264 High, level 4.2, 4:2:0, BT.709, LIMITED aralık | media-internals `info` |
| Ses | AAC, stereo, 48 kHz, CENC şifreli | media-internals `kAudioTracks` |
| DRM | Widevine (`com.widevine.alpha`), `use_hw_secure_codecs: false` (yazılım, L3) | media-internals `kSetCdm` |
| Video çözücü | `DecryptingVideoDecoder` (CDM içinde, CPU'da) | media-internals |
| Bitrate | **~4,8 Mbps** (Wi-Fi toplam indirme, video+ses+diğer, üst sınır) | 20 sn ve 30 sn adaptör sayacı: 4,86 ve 4,79 Mbps |
| Harici ekran | **Şu an bağlı değil** | WMI: tek aktif monitör AUO dahili panel |
| NVIDIA'daki 1920x1080 60 Hz ekran | Gerçek monitör değil, büyük ihtimalle Parsec sanal ekranı | `display_active: Disabled`, Parsec Virtual Display Adapter listede |
| Chrome hangi GPU'da? | NVIDIA'da işlem yok, **Intel iGPU'da** | `nvidia-smi` işlem listesi boş |

## 2. Ayrıntılar

### 2.1 Yakalama

- İlk denemede Chrome, baska bir pencerenin arkasındaydı. Ekran kopyası o pencereyi yakaladı, hareket 0 çıktı. Bu yanlış negatifti.
- Chrome öne alınınca hem BitBlt hem PrintWindow gerçek görüntü verdi, iki yöntemin istatistikleri neredeyse aynı (ort. 79,6 / 77,2).
- **Not:** OBS'in kullandığı Windows.Graphics.Capture (WGC) doğrudan denenmedi. BitBlt görüntü veriyorsa pencere "ekran yakalamayı engelle" (display affinity) bayrağı taşımıyor demektir. WGC de aynı bayrağa uyduğu için büyük ihtimalle çalışır (**tahmin**, Hat 1.1'de doğrulanacak).
- **Uyarı:** Chrome üstü tamamen kapalı pencerede video çizimini durdurabilir. Canlı hatta TOD penceresi görünür kalmalı (laptop ekranında, çıktı harici ekranda).

### 2.2 Kare hızı: gerçek 50p mi?

- 144 Hz ekranda örnekleme aralığı 6,94 ms. Yeni kare aralıkları 14,1 / 20,8 / 21,2 ms (2-3 örnek), ortalama 20 ms.
- 25p içeriğin iki kez gösterilmesi olsaydı ardışık farklardan biri hep çok küçük olurdu. Ölçüm: komşu fark oranı (min/max) medyan 0,87, 0,3'ün altı %0. 2 kare arası fark / 1 kare arası fark = 1,53 (gerçek hareket). **Sonuç: gerçek 50p.**
- Çift/tek indeks ortalama farkı 4,42 / 5,11, hafif asimetri. 50i yayının deinterlace edilmiş hali olabilir (**tahmin**).

### 2.3 Sıkıştırma yoğunluğu

- 1080p50 = saniyede 103,7 milyon piksel. 4,8 Mbps ile piksel başına ~0,046 bit.
- Karşılaştırma: eski beIN 1080p yayını ~6 Mbps (önceki rapor). TOD daha sıkı sıkıştırıyor. Bloklanma ve detay kaybı modelin ana düşmanı olmaya devam ediyor.
- ABR: oynatıcı 640x360 ile başlayıp 1080p'ye çıktı. Ağ zayıflarsa yayın ortasında çözünürlük düşebilir. Hat bu değişime dayanıklı olmalı.

## 3. Plana etkisi

| Konu | Eski varsayım (720p25) | Yeni gerçek (1080p50) |
|---|---|---|
| Büyütme oranı | x3 (0,92 MP → 8,29 MP) | **x2** (2,07 MP → 8,29 MP). Model işi daha kolay, ama girdi 2,25 kat daha çok piksel, hesap o kadar pahalı. |
| Ara kare | 60 çıktının 55'i üretilir, 12 faz (1/12 katları) | Çıktı k, kaynakta 5k/6. **10 kare/sn gerçek kareyle çakışır, 50 üretilir**, 6 faz (1/6 katları). Kareler arası 20 ms, hareket 40 ms'e göre yarı yarıya. **Ara kare çok daha kolay.** |
| Tampon (VRAM) | 720p kare ~2,8 MB | 1080p RGB uint8 ~6,2 MB, 1,5 sn = 75 kare ≈ 470 MB. Hâlâ sığar. NV12 tutulursa ~230 MB. |
| F14 (tam 1:1 yakalama) | Pencereyi 1280x720 yap | Laptop ekranı zaten 1920x1080: **video tam ekran = piksel piksel 1:1**. Çıktı harici ekranda. |
| F22 (Y/UV ayrı) | Renk 640x360 | Renk 960x540. Hâlâ geçerli. |
| F23 (zaman çizgisi) | 25 FPS'i 60/144 Hz ızgarasından çıkar | 50 FPS, 144 Hz'de kare başı 2,88 vsync. Hâlâ şart. |
| Hız bütçesi | Rahat sığar gibi | SR tarafı yeniden hesaplanmalı. EfRLFN x2 720p girdide 2080'de 271 FPS. 1080p girdide ~2,25 kat yavaş, **~120 FPS (tahmin)**. 50 gerçek kare/sn için yeterli görünüyor, ölçülecek. |
| Eğitim simülatörü (Hat 2.1) | 720p25 düşük bitrate | **1080p50, H.264 High, ~4-4,5 Mbps video, 4:2:0.** 4K CC klip → 1080p → x264 bu ayarlarla. |

## 4. Açık sorular

1. Diğer kanallar (beIN Sports 2-3-4, maç yayını vs. stüdyo) da 1080p50 mi? Tek kanal ölçüldü.
2. Gerçek video bitrate'i ne? Adaptör sayacı üst sınır. Segment boyutlarından ölçülebilir.
3. WGC (OBS'in yolu) doğrudan çalışıyor mu? Hat 1.1'de ilk iş.
4. Chrome çözme işini CPU'da yapıyor. 1080p50 yazılım çözme + WGC + bizim hat birlikte ne kadar CPU yiyor?
5. Chrome iGPU'da, işlem NVIDIA'da. Kareler Intel'den NVIDIA'ya kopyalanacak. Maliyeti ölçülecek.
6. Harici 4K ekran hangisi, ne zaman bağlanacak?

## Kaynaklar

- Ölçüm betikleri (bu oturum, scratchpad): `faz0_capture_test.py`, `faz0_capture_test2.py`, `faz0_fps.py`, `faz0_50p.py`, `faz0_cleanup.py`
- `chrome://media-internals`, TOD oynatıcısı (beIN Sports 1), 2026-09-15 09:32 UTC başlayan oturum
- `nvidia-smi`, `Get-NetAdapterStatistics`, `WmiMonitorBasicDisplayParams`, `Win32_VideoController`
