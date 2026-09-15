# Üçüncü araştırma: Yol haritası için modeller, lisanslar, veri, ölçüm ve donanım

- Tarih: 2026-09-15
- Durum: **Kısmen eskidi.** "Çevrimdışı referans hattı (tavan)" fikri kaldırıldı: her şey canlı, hedef canlı 4K 60 FPS. Bu rapordaki modeller artık sadece eğitimde öğretmen ya da kıyas adayı. Lisans, veri, metrik ve donanım bilgileri hâlâ geçerli. Güncel plan: [notes/yol-haritasi.md](../notes/yol-haritasi.md).
- Kapsam: Referans hattı ("tavan") model adayları ve lisansları, veri kaynakları, kalite metrikleri, yakalama kütüphaneleri, laptopun gerçek kısıtları.
- Önceki raporlar: [ilk araştırma](2026-09-15-ilk-arastirma-donanim-butcesi.md), [TOD ve 4070 Mobile](2026-09-15-tod-4070-mobile-sigdirma-arastirmasi.md)

## 1. Laptop (komutla okundu ve üretici bilgisiyle doğrulandı)

| Konu | Değer | Proje için anlamı |
|---|---|---|
| Model | ASUS TUF Gaming F15 FX507ZI4 | |
| CPU | Intel i7-12700H | Video çözme, kodlama ve veri hazırlama için yeterli |
| RAM | 16 GB | **Kısıt.** Ağır difüzyon modelleri düşük VRAM modunda 32 GB RAM öneriyor. |
| Disk | C: 111 GB boş (842 GB dolu) | **Ciddi kısıt.** 4K video verisi çok yer kaplar (bkz. 5. bölüm). |
| Görüntü çıkışı | 1x HDMI 2.1 FRL, 1x Thunderbolt 4 (DisplayPort) | Harici ekrana 4K 60 Hz rahat verir |
| MUX anahtarı | Var, NVIDIA Advanced Optimus ile | Harici ekran doğrudan NVIDIA'ya bağlanabilir, Intel üzerinden kopyalama derdi azalır |
| Python | 3.13.13 | Kütüphane uyumluluğu kontrol edilmeli |
| CUDA araç seti | 12.1 | Eski olabilir, sürücü 610.88 çok daha yeni |

## 2. Referans hattı (tavan) ve öğretmen adayları

### Büyütme (VSR)

| Model | Tür | Lisans | Donanım | Rol önerisi |
|---|---|---|---|---|
| SeedVR2 (3B / 7B) | Tek adımlı difüzyon, ByteDance | Apache-2.0 | 3B bile en az ~18 GB VRAM. FP8, INT8, NVFP4 sürümleri var. | Tavan adayı, **bulutta** |
| FlashVSR | Akış tipi difüzyon, Alibaba | Apache-2.0 | En az 8 GB. 4K için parça parça işleme (tiled) şart. 32 GB RAM önerilmiş. | Tavan adayı, **yerelde yavaş** çalışabilir |
| RVRT | Transformer VSR | Büyük kısmı CC-BY-NC | | Sadece kıyas, öğretmen olarak kullanılmaz |
| BasicVSR++ | Tekrarlayan VSR | Doğrulanamadı | | Kontrol edilecek |
| EfRLFN | Hafif SR | MIT | Gerçek zamanlı | Gerçek zamanlı motorun başlangıç modeli |

- SeedVR2, sıkıştırılmış girdide daha iyi. FlashVSR, uzun gerçek çekimlerde ve HD girdide daha iyi (üçüncü taraf karşılaştırma). TOD hem sıkıştırılmış hem gerçek çekim, ikisi de denenmeli.

### Ara kare (VFI)

| Model | Öne çıkan yanı | Lisans | Rol önerisi |
|---|---|---|---|
| EMA-VFI | Verimli, CNN ve Transformer karışımı | Apache-2.0 | Öğretmen adayı |
| BiM-VFI (CVPR 2025) | Düzensiz hareket (hızlanma, yön değişimi). LPIPS'te %26 iyileşme bildirilmiş. | Sadece araştırma ve eğitim | Kıyas. Futbola çok uygun, ama öğretmen olarak yayına giremez. |
| GIMM-VFI (NeurIPS 2024) | Keyfi zaman adımında ara kare | ComfyUI sarmalayıcısı ticari olmayan lisanslı. Orijinal lisans doğrulanmalı. | Kıyas |
| EDEN (2025) | Difüzyon, büyük hareket | Doğrulanmadı | Kontrol edilecek |
| RIFE | Hızlı, olgun | vs-rife MIT | Gerçek zamanlı başlangıç |

### Lisans önerisi (Yiğit'e sunulacak)

- Proje kodu: izin veren bir lisans (Apache-2.0 ya da MIT).
- **Yayınlanacak model ağırlıkları** yalnızca lisansı temiz öğretmen ve verilerden türetilir (Apache/MIT modeller + kendi çekimimiz).
- Ticari olmayan lisanslı modeller (RVRT, BiM-VFI vb.) sadece yerelde kıyas için kullanılır. Onlardan distillation yapılmaz.

## 3. Veri kaynakları

| Kaynak | İçerik | Lisans durumu | Karar |
|---|---|---|---|
| Pexels | Stok video | Kullanım koşulları makine öğrenmesi için veri toplamayı yasaklıyor | **Elendi** |
| SoccerNet | 550 maç, 720p 25 FPS | NDA | Kontrol edilecek. 4K hedef görüntü olamaz. |
| Inter4K | 4K, farklı kare hızları, 5 saniyelik klipler | Lisans bulunamadı | Kontrol edilecek |
| X4K1000FPS (XVFI) | 4K 1000 FPS, aşırı hareket | Lisans bulunamadı | Kontrol edilecek. Ara kare için çok değerli. |
| StreamSR | 5200 YouTube videosu | Lisans bilinmiyor | Kontrol edilecek |
| Kendi çekimimiz | Halı saha veya amatör maç, 4K 60 FPS | Tüm haklar bizde | **En temiz kaynak.** Demo videoları da buradan olur. |

## 4. Kalite ölçümü

| Ne ölçülür | Metrik | Not |
|---|---|---|
| Piksel doğruluğu | PSNR, SSIM | Hedef görüntü (ground truth) gerekir |
| Algısal kalite | LPIPS, DISTS | İnsan gözüne daha yakın |
| Zamansal tutarlılık (titreme) | tLPIPS, tOF | Video için şart |
| Yayın kalitesi | VMAF | Netflix'in metriği |
| Referanssız kalite | DOVER | Hedef görüntü olmadan (ör. canlı TOD çıktısı) |
| Göz testi | Yan yana video | Top, çizgiler, forma numaraları, skor tabelası |

## 5. Disk ve RAM hesabı

- 4K bir kare sıkıştırılmadan 3840 × 2160 × 3 bayt, yaklaşık 25 MB.
- **Tahmin:** PNG olarak kare başına 10-15 MB. 1 dakika 60 FPS = 3600 kare, yani **dakikada ~40-50 GB**. 111 GB boş alanla kareleri diske dökmek mümkün değil.
- Çözüm yolu: videoları sıkıştırılmış tutup eğitim ve ölçüm sırasında anlık çözmek, gerekirse harici SSD.
- 16 GB RAM, FlashVSR'ın düşük VRAM modu için önerilen 32 GB'ın altında. Tavan denemelerinin bir kısmı bulutta yapılmak zorunda kalabilir.

## 6. Gerçek zamanlı motor için teknik parçalar

| Parça | Seçenekler | Not |
|---|---|---|
| Pencere yakalama | Windows.Graphics.Capture (zbl: Rust ve Python, wincam: C++ ve Python), Desktop Duplication (D3DShot) | D3DShot doğrudan CUDA üzerinde PyTorch tensörü verebiliyor |
| GPU'da kalma | D3D11 ile CUDA arasında paylaşım | Kareyi CPU'ya indirmemek önemli |
| Ekrana çizme | CudaCanvas benzeri, D3D11/Vulkan | Sonra karar verilecek |
| Çıkarım | TensorRT, TensorRT for RTX (CUDA graphs) | |
| TOD'un gerçek akış bilgisi | `chrome://media-internals` | Oynatıcının çözünürlüğünü ve codec bilgisini gösterir. TOD'da denenmeli. |

## 7. Açık sorular

Güncel liste: [notes/soru-kuyrugu.md](../notes/soru-kuyrugu.md)

## Kaynaklar

- [ASUS TUF Gaming F15 (2023) teknik özellikler](https://www.asus.com/us/laptops/for-gaming/tuf-gaming/asus-tuf-gaming-f15-2023/techspec/)
- [LaptopMedia: TUF F15 FX507 2023](https://laptopmedia.com/highlights/specs-and-info-asus-tuf-gaming-f15-fx507-2023-supercharged-for-2023/)
- [Notebookcheck: TUF F15 FX507 incelemeleri](https://www.notebookcheck.net/Asus-TUF-Gaming-F15-FX507-Series.623521.0.html)
- [GitHub: SeedVR (SeedVR2)](https://github.com/ByteDance-Seed/SeedVR)
- [Hugging Face: SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B)
- [ComfyUI: SeedVR2 dokümanı](https://docs.comfy.org/tutorials/utility/seedvr2)
- [arXiv 2506.05301: SeedVR2](https://arxiv.org/pdf/2506.05301)
- [arXiv 2510.12747: FlashVSR](https://arxiv.org/pdf/2510.12747)
- [GitHub: FlashVSR-Pro](https://github.com/LujiaJin/FlashVSR-Pro)
- [Upsampler: SeedVR ve FlashVSR karşılaştırması](https://upsampler.com/blog/seedvr-vs-flashvsr-ai-video-super-resolution-2026)
- [GitHub: RVRT](https://github.com/jingyunliang/rvrt)
- [GitHub: BasicVSR++](https://github.com/ckkelvinchan/BasicVSR_PlusPlus)
- [GitHub: EMA-VFI](https://github.com/MCG-NJU/EMA-VFI)
- [GitHub: BiM-VFI](https://github.com/KAIST-VICLab/BiM-VFI)
- [BiM-VFI proje sayfası](https://kaist-viclab.github.io/BiM-VFI_site/)
- [GitHub: GIMM-VFI](https://github.com/GSeanCDAT/GIMM-VFI)
- [ComfyUI-GIMM-VFI lisansı](https://github.com/kijai/ComfyUI-GIMM-VFI/blob/main/LICENSE)
- [arXiv 2503.15831: EDEN](https://arxiv.org/abs/2503.15831)
- [GitHub: VFI sıralamaları](https://github.com/AIVFI/Video-Frame-Interpolation-Rankings-and-Video-Deblurring-Rankings)
- [Pexels: AI ve ML SSS](https://help.pexels.com/hc/en-us/articles/27292485713945-AI-and-ML-FAQ)
- [Pexels: Kullanım koşulları](https://www.pexels.com/terms-of-service/)
- [Inter4K](https://alexandrosstergiou.github.io/datasets/Inter4K/)
- [GitHub: XVFI (X4K1000FPS)](https://github.com/JihyongOh/XVFI)
- [arXiv 2511.16928: VSR değerlendirme metrikleri](https://arxiv.org/html/2511.16928)
- [GitHub: D3DShot](https://github.com/SerpentAI/D3DShot)
- [GitHub: zbl](https://github.com/modelflat/zbl)
- [PyPI: wincam](https://pypi.org/project/wincam/0.0.2)
- [PyPI: CudaCanvas](https://pypi.org/project/cudacanvas)
- [NVIDIA: TensorRT for RTX CUDA graphs](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/work-with-cuda-graphs.html)
- [Chromium Media Internals dokümanı](https://awesome.video/resource/188742)
