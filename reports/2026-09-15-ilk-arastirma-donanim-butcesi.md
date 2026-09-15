# İlk araştırma: 4070 Mobile'da canlı maçı 4K + 144 FPS izlemek

- Tarih: 2026-09-15
- Durum: **Kısmen eskidi.** 144 FPS hedefi eskidi (hedef 4K60). "50 FPS 1080p kaynak" varsayımı bir ara yanlış sanıldı (forumlar 720p25 diyordu), ama Faz 0 ölçümü **1080p50** olduğunu doğruladı: [2026-09-15-faz0-kesif-tod-yakalama.md](2026-09-15-faz0-kesif-tod-yakalama.md). Laptop ekranı 1080p 144 Hz. Güncel hali: [2026-09-15-tod-4070-mobile-sigdirma-arastirmasi.md](2026-09-15-tod-4070-mobile-sigdirma-arastirmasi.md). Model ve RTX VSR bilgileri hâlâ geçerli.
- Kapsam: Donanım bütçesi, model hızları, olası yollar, riskler, açık sorular.

## Hedef

Yiğit maç izlerken yayını 4K çözünürlükte ve çok yüksek FPS'te (144) izlemek istiyor. Çıktı yerel bir oynatıcıda (mpv türü) gösterilecek. Yeniden yayın gerekmiyor.

## Bilinenler

| Konu | Değer |
|---|---|
| GPU | RTX 4070 Mobile, 8 GB VRAM |
| Çıktı yolu | Yerel oynatıcı (mpv türü) |
| Model bilgisi | Yiğit modelleri bilmiyor, seçim bize kalıyor |

## Kavramlar (kısa)

- **Upscale:** 1080p görüntüyü 4K'ya büyütmek. 4K'da 4 kat piksel var, bu yüzden pahalı.
- **Interpolasyon:** İki gerçek karenin arasına tahmini kareler üretmek. RIFE bu işi yapan model ailesi.
- **TensorRT:** NVIDIA'nın modelleri kendi kartlarında hızlı çalıştırma motoru.
- **VRAM:** Ekran kartı belleği. Modeller ve kareler burada tutuluyor.

## Bulgular

### Kartın gücü

- RTX 4070 Mobile, masaüstü RTX 4060 Ti ile aynı çip sınıfında (AD106). Laptopun güç sınırına göre biraz daha yavaş olabilir.

### RIFE hızı

- 4060 Ti, RIFE v4.25, TensorRT FP16, 1080p: yaklaşık **84 kare/sn** (vs-mlrt v15.8, vs-rife v5.6.0 ile yapılan kullanıcı ölçümü). Lite modeller daha hızlı.
- SVP dokümanına göre 48 FPS gerçek zamanlı RIFE için gereken kartlar:
  - 1080p: RTX 2060 (TensorRT) / RTX 3070 (genel)
  - 4K: RTX 4080 (TensorRT) / RTX 4090 (genel)
- VRAM: 1440p için 6 ile 8 GB, 4K için en az 10 GB gerekiyor.
- Bir 4060 Ti kullanıcısı gerçek zamanlıda 720p'den 60 FPS'e çıkabilmiş, 1080p'de takılma yaşamış (SVP forumu).
- SVP dokümanında TensorRT'nin mpv gerçek zamanlı kullanımıyla ilgili kısıtlar olduğu yazıyor. Bizim ölçümümüzle doğrulanmalı.

### Büyütme

- **RTX Video Super Resolution (RTX VSR):** Sürücü içinde gelen, hafif çalışan AI büyütme. Chrome, Edge, VLC ve mpv destekliyor.
  - mpv'de örnek filtre: `--vf=d3d11vpp:scale=2:scaling-mode=nvidia`
  - VLC'de çıktı Direct3D11 olmalı, NVIDIA App'te VSR açılmalı.
- 144 kare/sn'yi ağır bir AI modeliyle (Real-ESRGAN, SPAN) 4K'ya büyütmek bu kartta mümkün değil.

### Kodsuz alternatif

- **Tarayıcı + RTX VSR** ile büyütme, üstüne **Lossless Scaling (LSFG 3)** ile kare üretimi.
- LSFG 3 herhangi bir pencerede çalışıyor, x2'den x20'ye kadar çarpan veriyor. En az 30 FPS kaynak istiyor, 60 ideal.
- Kalitesi RIFE kadar iyi değil ama bugün hiç kod yazmadan çalışır.

## Bütçe hesabı

Türkiye ve Avrupa'da maç yayınları genelde **50 FPS** (30 değil). 50'den 144'e çıkmak için saniyede yaklaşık 94 ara kare üretmek gerekiyor.

| Adım | Saniyedeki iş | 4070 Mobile'da |
|---|---|---|
| 50'den 144 FPS'e (RIFE, 1080p) | ~94 ara kare | Sınırda, lite model şart |
| 144 kareyi AI ile 4K'ya büyütmek | 144 × 4K kare | Sığmaz |
| İkisi birden | | Kesinlikle sığmaz |

**Sonuç:** Aynı kartta hem ağır AI ile 4K hem AI ile 144 FPS olmaz. Uygulanabilir sıralama şöyle:

1. İnterpolasyonu düşük çözünürlükte yap (1080p ya da altı).
2. Büyütmeyi ucuz yoldan yap (RTX VSR ya da GPU shader'ı).

## Maça özel riskler

- **Top:** Küçük ve çok hızlı. Interpolasyon modelleri en çok burada hata yapar (bulanıklaşma, çift görünme, kaybolma).
- **Skor tabelası ve alt yazılar:** Kamera hareket ederken sabit yazılar titreyebilir.
- **Gecikme:** Tampon kareler yüzünden yayın yaklaşık 100 ile 200 ms geriden gelir. İzlemek için sorun değil.
- **Oran:** 144, 50'ye tam bölünmüyor. Tam katlar (48×3, 72×2) daha temiz sonuç verir. Ekran yenileme hızı seçimi ayrıca konuşulmalı.
- **DRM:** beIN Connect, TOD, S Sport Plus, Exxen gibi platformlar görüntüyü kopya korumasıyla kilitler. Bu durumda görüntü mpv'ye alınamaz.

## Açık sorular

1. Maç kaynağı ne? (DRM'li platform / IPTV veya m3u8 link / indirilmiş dosya). mpv yolunun mümkün olup olmadığını belirler.
2. Ekran ne? Laptop ekranı büyük ihtimalle 4K değil. Harici 4K 144Hz monitör yoksa 4K'ya büyütmenin anlamı azalır.
3. Proje amacı "hedefi en kısa yoldan çalıştırmak" mı, yoksa "kendim yapıp öğrenmek" mi? Kodsuz alternatif bu yüzden masada.

## Kaynaklar

- [SVP: RIFE AI interpolation wiki](https://www.svp-team.com/wiki/RIFE_AI_interpolation)
- [SVP forum: RTX 4060 Ti'de RIFE](https://www.svp-team.com/forum/viewtopic.php?id=7204)
- [SVP forum: RTX 4070'te RIFE](https://www.svp-team.com/forum/viewtopic.php?id=7524)
- [SVP forum: RIFE filtre başlığı](https://www.svp-team.com/forum/viewtopic.php?id=6281&p=89)
- [mpv: NVIDIA super resolution tartışması](https://github.com/mpv-player/mpv/discussions/16513)
- [PCWorld: VLC'de RTX VSR](https://www.pcworld.com/article/1781216/nvidias-rtx-video-super-resolution-gets-vlc-support-for-offline-videos.html)
- [LSFG 3 (Lossless Scaling)](https://losslessscaling.com/lsfg-3/)
- [VideoCardz: Lossless Scaling 3](https://videocardz.com/newz/lossless-scaling-3-released-with-frame-generation-up-to-x20-no-shrooms-required)
