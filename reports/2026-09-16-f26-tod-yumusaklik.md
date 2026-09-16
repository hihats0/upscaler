# F26: Gerçek TOD yumuşaklığı, ölçüm ve ilk eğitim denemeleri

- **Tarih:** 2026-09-16
- **Durum:** araştırma (yarım; GAN denemesi sürüyor)
- **Kapsam:** Yiğit'in gözlemi ("logo keskin, kamera görüntüsü 1080p gibi değil") sayıyla doğrulandı, eğitim bozulması gerçeğe yaklaştırıldı, kayıp fonksiyonu denemeleri yapıldı. Önceki: [hat22 ilk ince ayar](2026-09-16-hat22-ilk-ince-ayar.md).

## 0. Özet

1. Canlı TOD kamera görüntüsü, eğitimde kullandığımız TOD simülasyonundan belirgin biçimde yumuşak.
2. Varsayılan model ft2, gerçek TOD'da **bicubic'ten bile az** ince detay üretiyor. Model sıkıştırmayı temizlerken detayı da siliyor.
3. Bozulmayı gerçeğe yaklaştırmak ve genlik spektrumu kaybı eklemek yönü doğru ama etki küçük: detay oranı (hf) 0,31'den en fazla ~0,35'e çıktı, sonra doydu. Çim dokusu hiçbir regresyon modelinde yok.
4. Sonraki adım GAN (çekişmeli eğitim). İlk deneme başladı, GPU 87 °C'ye çıkınca durduruldu.

## 1. Ölçüm yöntemi (`upscaler/sharpness.py`, `tools/sharpness_probe.py`)

- TOD kuralı: kareler sadece bellekte işlenir, diske sadece sayı yazılır.
- Sadece kadrajın ortası ölçülür (satırlar %22-78, sütunlar %12-88). Böylece logo, skor tabelası ve alt bant ölçüme girmez.
- **spk_50_75:** Radyal spektrumda Nyquist'in 0,5-0,75 bandındaki enerjinin 0,25-0,5 bandına oranı. Asıl ayırt edici ölçü bu.
  - Doğrulama (CC klip, 1080p): keskin 0,20, 720p'den büyütülmüş 0,08, 540p'den büyütülmüş 0,03.
- **kayip05:** Kareyi 0,5 oranında küçültüp geri büyütünce oluşan ortalama mutlak fark. Detay miktarını gösterir ama içeriğe çok bağlıdır.
- **oranXX:** Küçült-büyüt kaybı eğrisi. Ayırt ediciliği zayıf çıktı.
- Canlı ölçümde Chrome tam ekran olmalı (yakalama 1920x1080 ya da 1920x1079). Değilse video küçültülmüş olur ve ölçüm geçersiz sayılır.

## 2. Bulgular

### 2.1 Canlı TOD ve simülasyon

Canlı kaynak: Premier League tekrar yayını, beIN Sports 3.

| Kaynak | kayip05 | spk_50_75 | spk_75_100 |
|---|---|---|---|
| Eski sim (5bgF 200. sn) | 2,55 | 0,26 | 0,043 |
| CC keskin 1080p (63p 120. sn) | 2,55 | 0,20 | 0,018 |
| **TOD `tod_net1`** (180 sn, 350 örnek) | **0,71** | **0,137** | **0,036** |
| **TOD `tod_net2`** (300 sn, 296 örnek) | **0,90** | **0,132** | **0,038** |

### 2.2 Bozulma taraması (`tools/degrade_sweep.py`, `runs/sweep1`)

- 4 klip × 2 bölüm × 9 bozulma çeşidi. Her çeşitte ön küçültme p, 1080p'ye bicubic büyütme, unsharp ve 4,8 Mbps H.264 uygulandı.
- TOD'un spk_50_75 değeri (0,137), p=1 (0,161) ile p=0,75 (0,117) arasına düşüyor.
- TOD'un spk_75_100 değeri (0,036), bozulmasız varyantın (0,034) bile üstünde. Tahmin: sıkıştırma gürültüsü ya da yayıncı keskinleştirmesi.
- kayip05 bakımından hiçbir varyant TOD'a yaklaşmıyor (varyantlar 1,0-1,9, TOD 0,7-0,9). Sebep büyük ölçüde içerik farkı: yayında geniş çim planları çok.
- Tek bir ayar seçilmedi. Onun yerine eğitimde her bölüme rastgele bozulma uygulandı: **p ∈ [0,5; 1,0], unsharp ∈ [0; 0,8]**.

### 2.3 CC GT'nin de bir kısmı yumuşak (`tools/gt_sharpness.py`, `runs/gt_sharp.json`)

- 12 klipten 182 bölüm ölçüldü (4K karede spk_50_75). Klip medyanları 0,029 ile 0,202 arasında.
- Bazı bölümler, örneğin 63pDvVidZJA'nın 775. saniyesi, bozulma uygulanmadan 1080p'ye indirildiğinde bile yumuşak.
- Eşik 0,04 seçildi: 102 bölüm geçti, eğitim için 70 bölüm elendi.

### 2.4 Modellerin gerçek TOD'da 4K çıktısı (`sharpness_probe --sr`, 30 sn, 10 örnek)

| 4K çıktı | spk_50_75 |
|---|---|
| bicubic | 0,035 |
| hazır RT4KSR | 0,125 (sıkıştırma kusurlarını da büyütüyor) |
| **ft2** | **0,019** |
| Referans: keskin CC 4K bölümleri | ~0,04-0,06 (medyan) |

## 3. Veri ve eğitim

- **Kod:**
  - `tools/tod_sim.py --pre --sharp`
  - `tools/make_pairs.py --pre-range --sharp-range --gt-sharp --gt-min`
  - `tools/train_sr.py`: `--fft-weight --fft-mode complex|amp --gan-weight`, doğrulamada `hf` ölçüsü, `PatchDisc`
- **train2:** 10 klip, 56,3 bin yama, 6,1 GB. 5bgF-5I2P_M (canlı puan klibi) ve O3gD6n0zoik eğitimden hariç tutuldu.
- **hf:** Çıktının max(|fy|,|fx|) ≥ 0,5 Nyquist bandındaki enerjisinin GT'ninkine oranı. Yani 4K'da 1080p'nin taşıyamadığı detayın ne kadarının geri geldiğini ölçer. 1,0 GT kadar detay demek.
- **val2 hatası:** Tek bölüm, rastgele p=0,99 ve unsharp 0,3 düştü. TOD'u temsil etmediği için ft3/ft4'ün val2 sonuçları kullanılmadı.
- **val3:** O3gD6n0zoik ve 5bgF-5I2P_M'nin keskin bölümleri, sabit p=0,7 ve unsharp 0,4. Toplam 1596 yama. Canlı puan klipleriyle aynı bozulma (`data/sim/5bgF-5I2P_M_s{20,200}_l10_50p_tod.*`).

| val3 | PSNR-Y (dB) | hf |
|---|---|---|
| bicubic | 33,85 | 0,319 |
| hazır RT4KSR | 33,23 | 0,426 |
| ft2 | 33,99 | 0,314 |
| q_l1 (5 bin adım, L1) | 33,84 | 0,309 |
| q_amp05 (+ genlik 0,5) | 33,86 | 0,329 |
| q_amp2 (+ genlik 2) | 33,87 | 0,340 |
| q_amp8 | 33,84 | 0,347 |
| q_amp20 | 33,81 | 0,349 |

- ft3 (L1, val2) 10 bin adımda, ft4 (+ karmaşık FFT 0,1) 3 bin adımda durduruldu. İkisinde de hf düştü. Karmaşık FFT farkı faz hatasını cezalandırdığı için yine bulanık çıktıyı seçiyor.
- **Göz testi** (`tools/compare_patches.py`, `runs/karsilastirma_on.png`): GT'deki çim tanesi hiçbir modelde yok. Yazı ve kenarlarda modeller birbirine yakın.
- **GAN g01** (q_amp2'den başlatıldı, L1 + genlik 2 + GAN 0,01, lr 1e-4): 1000. adımda 33,89 dB / hf 0,338, henüz fark yok. Hız ~5 it/sn, 95 W, 86-87 °C, sürücü ısı kısıtlaması devrede. Durduruldu; g05 başlamıştı, o da durduruldu.

## 4. GPU sağlığı

- Bugün GPU yaklaşık 5,1 saat yük altındaydı (12:25-21:22 arası, `runs/` kayıtları).
- `train_sr` duraklama eşiği 88/78 °C'den **80/70 °C**'ye indirildi. GAN koşuları bu eşikle daha yavaş olacak.

## Açık sorular

- GAN, bu küçük ağda (24 kanal) çim dokusunu üretebilir mi? Titreme yapar mı? Canlı puanda zamansal titreme ölçülmeli.
- Algısal kayıp (VGG/LPIPS) için ağırlık indirmek gerekiyor, Yiğit'in izni lazım.
- hf ölçüsü iyi detayı büyütülmüş kusurdan ayırmıyor (hazır model 0,43). PSNR ve göz testiyle birlikte okunmalı.
- Yayındaki gerçek bozulma zinciri bilinmiyor. spk_75_100'ün yüksek çıkması keskinleştirmeye mi yoksa gürültüye mi bağlı?
- Yeni modellerin gerçek TOD'da `sharpness_probe --sr` ile ölçülmesi için TOD'da tekrar ya da özet açık olmalı. Harici 4K ekranda göz testi hâlâ yapılmadı.

## Kaynaklar

- Kayıtlar: `runs/tod_net1`, `runs/tod_net2`, `runs/sweep1`, `runs/gt_sharp.json`, `runs/pairs2.log`, `runs/val3.log`, `runs/train_{ft3,ft4,q_*,g01,g05}`, `runs/karsilastirma_on.png`, `runs/tod_uzun` (durdurulan uzun sınav).
- Wang vd. 2021, Real-ESRGAN (rastgele bozulma ve GAN ile görüntü büyütme): https://github.com/xinntao/Real-ESRGAN
- Jiang vd. 2021, Focal Frequency Loss (frekans uzayında kayıp): https://github.com/EndlessSora/focal-frequency-loss
