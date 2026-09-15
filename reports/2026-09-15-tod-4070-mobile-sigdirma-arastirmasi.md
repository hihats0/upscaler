# İkinci araştırma: TOD yayınını 4070 Mobile'a nasıl sığdırırız?

- Tarih: 2026-09-15
- Durum: **Kısmen eskidi.** "Senaryo A: laptop ekranı 1080p144" artık geçersiz. Hedef harici 4K ekranda **4K 60 FPS** (Senaryo B). **"TOD 720p 25 FPS" bilgisi yanlış çıktı:** Faz 0 ölçümünde beIN Sports 1 Chrome'da 1080p 50 FPS, ~4,8 Mbps. Bkz. [2026-09-15-faz0-kesif-tod-yakalama.md](2026-09-15-faz0-kesif-tod-yakalama.md). Donanım, lisans ve model hız bilgileri hâlâ geçerli, hız bütçesi yeniden hesaplanmalı.
- Kapsam: TOD'un gerçek yayın kalitesi, DRM, laptopun gerçek donanımı, hız bütçesi, yaratıcı fikirler, veri ve eğitim, riskler.
- Önceki rapor: [2026-09-15-ilk-arastirma-donanim-butcesi.md](2026-09-15-ilk-arastirma-donanim-butcesi.md). O raporun "50 FPS 1080p kaynak" varsayımı burada yanlış sanıldı, ama Faz 0 ölçümü onu **doğruladı** (1080p50).

## 1. Değişen gerçekler

### Kaynak: TOD

- Web ve uygulamada yayın **720p, 25 FPS** (kullanıcı şikayetleri ve forumlar). 1080p sadece "TV paketi" seçeneğiyle geliyor. Tarayıcıda en fazla 720p.
- Süper Lig için 4K yayın yok.
- Bitrate düşük, pikselleşme şikayetleri çok. Karşılaştırma: eski beIN yayınları 1080p ve yaklaşık 6000 kbps idi.
- Yani eldeki kaynak **küçük, düşük kare hızlı ve sıkıştırma hasarlı**. Hem çözünürlük hem kare hızı hem de sıkıştırma hasarı sorun.

### DRM

- TOD görüntüyü DRM ile koruyor. Görüntüyü mpv'ye doğrudan almak mümkün değil.
- Windows'ta Edge + PlayReady (donanım seviyesi) ekran yakalamayı siyah ekrana çeviriyor.
- Chrome + Widevine L3 yazılım seviyesinde çalışıyor. Birçok sitede ekran yakalama aracı görüntüyü alabiliyor. TOD için **denenmedi**.
- Tavır: DRM'i kırmayacağız, atlatmayacağız. İşletim sisteminin normal ekran yakalama API'si (Windows Graphics Capture) görüntü veriyorsa kullanırız. Siyah veriyorsa o yol kapanır.

### Laptopun gerçek donanımı (2026-09-15'te komutla okundu)

| Konu | Değer |
|---|---|
| GPU | RTX 4070 Laptop GPU, 8188 MiB |
| Güç sınırı | En fazla 140 W (şu an 138 W). 4070 Mobile'ın en güçlü sürümü. |
| Sürücü | 610.88 |
| Dahili ekran | 1920x1080, 144 Hz (AUO panel, yaklaşık 15.6"). Intel Iris Xe üzerinden bağlı (Optimus). |
| NVIDIA'nın gördüğü ekran | 1920x1080, 60 Hz. Harici bir monitör olabilir, doğrulanmadı. |
| Diğer | Parsec sanal ekran sürücüsü yüklü. |

**Sonuç:** Laptop ekranı 1080p. 4K çıktı bu ekranda **görünmez**, tekrar 1080p'ye küçültülür. 4K hedefi ancak harici bir 4K ekranla anlam kazanır. Yiğit'e soruldu.

## 2. Hız bütçesi (yeniden hesap)

Kaynak 720p olunca iş ciddi şekilde ucuzluyor. Hafif SR modellerinde maliyetin çoğu **girdi çözünürlüğünde** harcanır. Son büyütme katmanı (pixel shuffle) ucuzdur.

### Referans hızlar (başkalarının ölçümleri, bizim kartta değil)

| Model | İş | Kart | Hız | Kaynak |
|---|---|---|---|---|
| EfRLFN | 720p → 1440p (x2) | RTX 2080 | 271 FPS | ICLR 2026 makalesi |
| RLFN | 720p → 1440p (x2) | RTX 2080 | 225 FPS | aynı |
| SPAN | 720p → 1440p (x2) | RTX 2080 | 60 FPS | aynı |
| NVIDIA VSR | 720p → 1440p (x2) | RTX 2080 | 52 FPS | aynı |
| EfRLFN | x4 | RTX A6000 | 68 FPS | aynı |
| RIFE v4.25 TRT FP16 | 1080p ara kare | RTX 4060 Ti | 84 FPS | SVP forumu |

- Notebookcheck, 4070 Laptop'u en iyi güç ayarında 2080 Ti ile aynı sınıfta gösteriyor. Bizim 140 W sürüm üst sınıra yakın.
- **Tahmin (ölçülmedi):** RIFE 720p'de 1080p'ye göre yaklaşık 2.25 kat daha az piksel işler. Tam model ~180, lite model 250 FPS ve üstü olabilir.

### Senaryo A: laptop ekranı (720p25 → 1080p144)

| Adım | Saniyedeki iş | Tahmini durum |
|---|---|---|
| Ara kare (720p'de) | 144 çıktının ~119'u üretilmiş kare | Lite RIFE ile sığar, tam modelde sınırda |
| Büyütme 720p → 1080p | 144 kare | EfRLFN sınıfı modelle sığar |
| Toplam | | Sığar gibi, ölçülmeli |

### Senaryo B: harici 4K ekran (720p25 → 4K60)

| Adım | Saniyedeki iş | Tahmini durum |
|---|---|---|
| Ara kare (720p'de) | 60 çıktının ~55'i üretilmiş kare | Rahat sığar |
| Büyütme 720p → 4K (x3) | 60 kare | EfRLFN sınıfıyla sığar, SPAN sınıfı sığmaz |
| Toplam | | Sığar gibi, ölçülmeli |

**Ana bulgu:** Hız çözülebilir bir sorun. **Asıl savaş kalite.** 720p, 25 FPS ve düşük bitrate bir yayından "çok kaliteli" görüntü çıkarmak, hafif modellerin tek başına yapabileceği bir şey değil.

## 3. Kaliteyi bütçeyi aşmadan artırma fikirleri

Ayrıntılı fikir listesi ve durumları [notes/fikir-bankasi.md](../notes/fikir-bankasi.md) dosyasında. En güçlü adaylar:

1. **Ortak hareket haritası:** Ara kare modeli (RIFE) zaten kareler arası hareketi hesaplıyor. Aynı haritayı büyütmede de kullanırız, iki kez hesaplamayız. Düşük çözünürlükte hesaplanan akış ve maske yukarı örneklenip yeniden kullanılabilir (arXiv 2104.05778).
2. **Pahalı büyütme sadece gerçek karelere:** Saniyede 25 gerçek kareyi güçlü bir modelle büyütürüz. Ara kareleri büyütülmüş komşulardan hareketle kaydırıp küçük bir düzeltme ağıyla üretiriz. Böylece bütçenin 2.4 ile 5.8 katını gerçek karelere ayırabiliriz.
3. **Zamansal biriktirme (videoda DLSS mantığı):** Önceki çıktıyı hareketle kaydırıp yeni kareyle harmanlarız. Futbolda kamera sürekli kaydığı için ardışık karelerde alt piksel kaymaları olur. Bu **gerçek detay** kazandırır, uydurma değildir. Önceki tahmini kullanan tekrarlayan (recurrent) VSR yöntemleri literatürde var.
4. **Futbola ve TOD'a özel eğitim:** TOD'un bozulmasını (720p, 25 FPS, düşük bitrate H.264) taklit eden eğitim çiftleri. StreamSR çalışması, gerçek sıkıştırma hasarıyla eğitimin bicubic küçültmeye göre belirgin fark yarattığını gösteriyor. Futbol SR çalışması da futbol verisiyle eğitimin SoccerNet'te hafif iyileşme verdiğini buldu.
5. **Futbola özel parçalar:** Sahne kesmesi algılama, skor tabelası ve logo gibi sabit katmanları ayrı işleme, top için fizik tabanlı ara konum, kamera kaymasının ucuz hesabı.

## 4. Kullanılabilecek hazır parçalar (sadece yapı taşı ya da kıyas ölçütü)

| Parça | Ne işe yarar | Lisans ve not |
|---|---|---|
| EfRLFN | Hızlı SR, akış içeriği için ince ayarlı | MIT. x2 ve x4 ağırlıkları var. StreamSR veri seti paylaşılmış. |
| RIFE (Practical-RIFE, vs-rife) | Ara kare | vs-rife MIT. TensorRT desteği var. |
| RTX Video SDK | NVIDIA VSR'ı uygulamaya gömmek | RTX 20 ve üstü. DX11, DX12, Vulkan, CUDA. Lisans okunmadı. **Kıyas ölçütü** olarak değerli. |
| Lossless Scaling (LSFG 3) | Pencere yakalayıp kare üretme | Ticari. Kıyas ölçütü. Mimari örneği (yakala, işle, üstte göster). |
| TensorRT FP8/INT8 | Hızlandırma | FP8 Ada kartlarda destekli. Başka modellerde FP16'ya göre ~1.4 kat hız bildirilmiş. SR için ölçülmedi. |

## 5. Veri ve eğitim

- **TOD kaydı veri seti olamaz.** Yayını kaydedip saklamak telif hakkına ve TOD kullanım koşullarına aykırı. Yiğit'in "tüm Süper Lig maçlarını verelim" fikrinin ruhu doğru (alana özel eğitim işe yarıyor), ama kaynak olarak TOD kullanılamaz.
- **SoccerNet:** 550 maç, 720p ve 25 FPS, NDA gerekiyor. 4K için gerçek hedef görüntü (ground truth) olamaz, ama futbola özel ara kare ve x2 eğitimi için aday. NDA koşulları okunmalı.
- **StreamSR:** YouTube kaynaklı 5200 video, gerçek sıkıştırma hasarı. EfRLFN deposunda paylaşılmış. Lisansı kontrol edilmeli.
- **Kendi çekimimiz:** Telefonla 4K 60 FPS halı saha veya amatör maç çekimi. Hem SR hem ara kare için gerçek hedef görüntü verir (60 FPS'ten 25 FPS'e seyreltip aradaki gerçek kareleri hedef olarak kullanırız). Yayın görüntüsüne benzemez, bu bir dezavantaj.
- **Açık lisanslı stok videolar** (Pexels vb.): Futbol klipleri var. Makine öğrenmesi eğitimine izin verip vermediği kontrol edilmeli.
- **Eğitim donanımı:** EfRLFN boyutundaki küçük modeller 8 GB'ta eğitilebilir ama yavaş olur. Gerekirse bulutta GPU kiralanabilir.

## 6. Riskler

| Risk | Etki | Nasıl anlarız |
|---|---|---|
| TOD, Chrome'da da siyah ekran veriyor | TOD için gerçek zamanlı yol kapanır | OBS ile 2 dakikalık test |
| 25 FPS'te top kareler arasında çok yol alıyor | Ara karelerde top bozulur | Kendi çekimimizle ölçüm |
| 720p düşük bitrate'te detay yok | "Çok kaliteli 4K" beklentisi karşılanmaz, model detay uydurur | Kıyas testi, göz testi |
| Optimus: ekran Intel'de, işlem NVIDIA'da | Kareler iki kart arasında kopyalanır, gecikme artar | Ölçüm. MUX anahtarı var mı bakılmalı |
| Tarayıcı 720p videoyu tam ekranda 1080p'ye büyütüyor | Bulanık kare yakalarız | Pencereyi video tam 1280x720 olacak şekilde ayarlamak |
| Laptop ısınması, prizde olmama | Hız düşer | Uzun süreli ölçüm |

## 7. Açık sorular

1. Maç hangi ekranda izlenecek? Laptop ekranı (1080p 144 Hz) mı, harici 4K ekran mı?
2. TOD, Chrome'da OBS ile yakalanınca görüntü geliyor mu?
3. NVIDIA'nın gördüğü 1920x1080 60 Hz ekran ne? Harici monitör mü?
4. Laptopta MUX anahtarı veya Advanced Optimus var mı?

## Kaynaklar

- [Şikayetvar: TOD maksimum 720p](https://www.sikayetvar.com/tod-tv/tod-tv-maksimum-720p-olmasi)
- [Şikayetvar: TOD düşük çözünürlük](https://www.sikayetvar.com/tod-tv/todtv-dusuk-cozunurluk-yayin-yapiyor-full-hd-1080p-yayin-bekliyoruz)
- [Şikayetvar: TOD yayın kalitesi](https://www.sikayetvar.com/tod-tv/yayin-kalitesi)
- [Technopat: TOD uygulama ve TV görüntü kalitesi](https://www.technopat.net/sosyal/konu/tod-tv-uygulamada-ve-akilli-tvde-goeruentue-kalitesi-nedir.3731714/)
- [Technopat: TOD Smart TV seçeneği ve 1080p](https://www.technopat.net/sosyal/konu/tod-smart-tv-android-tv-secenegini-secmeden-hdmi-baglanti-ile-1080p-izlenir-mi.3365339/)
- [Screenify: DRM ve siyah ekran](https://www.screenify.studio/blog/2026-04-23-record-drm-protected-content)
- [Medium: Edge, PlayReady ve Widevine](https://medium.com/swlh/can-chromium-based-edge-solve-browsers-drm-issues-83479089d67d)
- [arXiv 2602.11339: Real-Time SR for Streaming Content (EfRLFN, StreamSR)](https://arxiv.org/html/2602.11339v2)
- [GitHub: EfRLFN](https://github.com/EvgeneyBogatyrev/EfRLFN)
- [arXiv 2104.05778: STVSR with Low-Resolution Flow and Mask Upsampling](https://arxiv.org/abs/2104.05778)
- [arXiv 2312.10890: Low-latency Space-time Supersampling](https://arxiv.org/abs/2312.10890)
- [arXiv 2402.00163: Football SR for object detection](https://arxiv.org/pdf/2402.00163)
- [arXiv 2503.11181: Football broadcast generative upscaler](https://arxiv.org/abs/2503.11181)
- [arXiv 2504.10686: NTIRE 2025 Efficient SR Challenge](https://arxiv.org/abs/2504.10686)
- [arXiv 1909.08080: Recurrent latent space propagation VSR](https://arxiv.org/pdf/1909.08080)
- [SoccerNet data](https://www.soccer-net.org/data)
- [NVIDIA RTX Video SDK](https://developer.nvidia.com/rtx-video-sdk)
- [NVIDIA blog: RTX Video SDK](https://developer.nvidia.com/blog/enhancing-low-resolution-sdr-video-with-the-nvidia-rtx-video-sdk)
- [NVIDIA blog: FP8 quantization with TensorRT](https://developer.nvidia.com/blog/model-quantization-turn-fp8-checkpoints-into-high-performance-inference-engines-with-nvidia-tensorrt/)
- [Notebookcheck: RTX 4070 Laptop vs 4060 Ti](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4070-Laptop-GPU-vs-NVIDIA-GeForce-RTX-4060-Ti-8G-vs-NVIDIA-GeForce-RTX-4060_11453_11592_11593.247598.0.html)
- [Notebookcheck: RTX 4070 Laptop vs 2080 Ti](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4070-Laptop-GPU-vs-GeForce-RTX-2080-Ti-Desktop_11453_9526.247598.0.html)
- [GitHub: vs-rife](https://github.com/HolyWu/vs-rife)
- [SVP forum: 4060 Ti RIFE](https://www.svp-team.com/forum/viewtopic.php?id=7204)
- [GitHub: mpv-AnimeJaNai](https://github.com/the-database/mpv-upscale-2x_animejanai)
