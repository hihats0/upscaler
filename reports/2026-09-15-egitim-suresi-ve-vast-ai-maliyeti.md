# Eğitim süresi ve vast.ai maliyeti

- Tarih: 2026-09-15
- Durum: Araştırma. Sayıların çoğu **tahmin**. Kesin değerler ilk küçük denemede ölçülecek.
- Kapsam: Vast.ai GPU fiyatları, öğretmen modellerin hızı, kendi modelimizin eğitim süresi ve maliyet senaryoları.

## 1. Vast.ai fiyatları

| GPU | Bellek | Saatlik (tipik, on-demand) | Kaynak ve tarih |
|---|---|---|---|
| RTX 4090 | 24 GB | **$0.39** (en ucuz ilanlar $0.14'ten başlıyor, bir sitede $0.53) | Thunder Compute, 2026-09-01. Vast.ai sayfası. ComputePrices, 2026-09-14. |
| RTX 5090 | 32 GB | $0.25'ten başlayan ilanlar var, tipik fiyat doğrulanamadı | Vast.ai sayfası |
| A100 40GB | 40 GB | ~$0.87 | ComputePrices, 2026-09-14 |
| A100 80GB | 80 GB | $0.94 ile $1.32 arası | ComputePrices, Thunder Compute |
| L40S | 48 GB | ~$0.80 | ComputePrices |
| H100 | 80 GB | **$2.13 ile $2.21 arası** | ComputePrices, Thunder Compute |
| H200 | 141 GB | ~$3.75 | ComputePrices |

**Gizli maliyetler ve uyarılar:**
- **Kesintili (interruptible) kiralama** aynı sunucuda %30-50 daha ucuz. Karşılığında iş yarıda kesilebilir, düzenli kayıt (checkpoint) almak şart.
- **Depolama:** Aylık GB başına ~$0.10-0.15. Makine durdurulsa bile ücret işlemeye devam eder, silinmedikçe bitmez. Örnek: 200 GB veri ayda ~$20-30.
- **Veri transferi:** Her sunucunun kendi ücreti var.
- **Sunucu kalitesi değişken.** Sadece "verified" ve güvenilirliği %99 üstü sunucular seçilmeli.
- Faturalama saniye bazında.

## 2. Öğretmen modellerin hızı

| Model | Hız | Bellek | Lisans | Not |
|---|---|---|---|---|
| SwiftVR (2026) | H100'de 4K 14 FPS, 1440p 31 FPS. RTX 5090'da 1080p 26 FPS. | 1440p'de 38 GB | Kod ve ağırlık yayında. Kod lisansı doğrulanmadı. | 4K'ya tek GPU'da çıkabilen tek üretken model olarak raporlanmış |
| FlashVSR | A100'de 768x1408 için ~17 FPS | 4K için parça parça işleme | Kod Apache-2.0 (önceki rapor) | |
| SeedVR2-7B | 720p'de 100 kare ~300 sn (~3 sn/kare) | 4K'da tek H100'e sığmıyor | Apache-2.0 | Pahalı ve 4K çıktı zor |

## 3. Kendi modelimizin eğitim süresi (tahmin)

**Dayanak:**
- RLFN: 0.317M parametre, 256x256 girişte 19.67G FLOPs, batch 64, 1000 epoch, üç aşama.
- SPAN (NTIRE 2024): 10⁶ iterasyon + ince ayarlar.
- Makalelerde duvar saati süresi verilmemiş. Aşağıdaki süreler FLOPs üzerinden kaba tahmin.
- 4070 Laptop, RTX 4090'dan kabaca 2-2.5 kat yavaş (tahmin).

| Senaryo | RTX 4090 | Maliyet ($0.39/sa) | Laptopta |
|---|---|---|---|
| A. **İnce ayar:** Hazır EfRLFN/RIFE ağırlıkları + futbol + TOD bozulması (100-200 bin iterasyon) | 5-12 saat | **$2-5** | 1-2 gece |
| B. **Sıfırdan küçük SR modeli** (~1M iterasyon) | 50-80 saat | **$20-30** | 5-7 gün kesintisiz (ısınma riski) |
| C. **Kendi video modelimiz:** Çok kareli, çift yönlü, ara kare dahil. Girdi kare sayısı kadar pahalı. | 150-400 saat (ya da H100'de daha kısa) | **$60-160** | Pratik değil |
| D. **Öğretmen verisi, SwiftVR:** 30 dk görüntü (25 FPS'te 45 bin kare) | H100'de ~1 saat | **~$2-3** | Sığmaz |
| D'. **Öğretmen verisi, SeedVR2-7B:** 45 bin kare, 720p | H100'de ~37 saat | **~$80** | Sığmaz |

**Deneme-yanılma çarpanı:** İlk eğitimler neredeyse her zaman hatalı ya da yetersiz çıkar. Gerçekçi toplam, tek bir koşunun **3-5 katı**.

**Tahmini ilk ciddi model bütçesi:** İnce ayarla başlanırsa $10-30. Kendi video modeli dahil **$50-200** (depolama hariç).

## 4. Önemli çıkarım: öğretmen her zaman gerekli değil

- Temiz **4K** CC klipler varsa hedef görüntü zaten elimizde. Onu doğrudan kullanmak öğretmen çıktısından daha iyidir.
- Öğretmen asıl şurada gerekiyor: CC 4K futbol görüntüsü azsa, bol bulunan **1080p** CC klipleri öğretmenle 4K "sahte hedefe" çeviririz. Bunun için en uygun aday SwiftVR (hızlı, 4K), maliyeti düşük.
- İnce ayarın ilk turu laptopta gece çalışarak **bedavaya** yapılabilir.

## Kaynaklar

- [Thunder Compute: Vast.ai fiyatları 2026](https://www.thundercompute.com/blog/vast-ai-vs-thunder-compute)
- [Vast.ai: RTX 4090](https://vast.ai/pricing/gpu/RTX-4090)
- [Vast.ai: RTX 5090](https://vast.ai/pricing/gpu/RTX-5090)
- [ComputePrices: Vast.ai](https://computeprices.com/providers/vast)
- [gpuperhour: Vast.ai](https://gpuperhour.com/providers/vastai)
- [arXiv 2510.12747: FlashVSR](https://arxiv.org/abs/2510.12747)
- [arXiv 2606.09516: SwiftVR](https://arxiv.org/abs/2606.09516)
- [SwiftVR proje sayfası](https://h-oliday.github.io/SwiftVR)
- [arXiv 2506.05301: SeedVR2](https://www.alphaxiv.org/abs/2506.05301v2)
- [GitHub: RLFN](https://github.com/bytedance/RLFN)
- [arXiv 2205.07514: RLFN](https://arxiv.org/pdf/2205.07514)
- [arXiv 2404.10343: NTIRE 2024 ESR raporu](https://arxiv.org/html/2404.10343)
- [GitHub: ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE)
