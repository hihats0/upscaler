# Hat 2.2: İlk ince ayar (RT4KSR x2, futbol + TOD bozulması)

- **Tarih:** 2026-09-16
- **Durum:** onaylı (ölçüldü). `rt4ksr-x2-ft2` watch varsayılanı oldu.
- **Kapsam:** Hazır RT4KSR x2'yi aynı mimaride, CC BY futbol klipleri ve TOD simülatörü bozulmasıyla laptopta ince ayar yapmak; TensorRT'ye derleyip canlı puanla (Hat 1.3) hazır modelle kıyaslamak. Bulut kullanılmadı. Önceki: [hat13 canlı ölçüm](2026-09-16-hat13-canli-olcum.md).

## Özet

| | Hazır RT4KSR | ft2 (ince ayarlı) | Fark |
|---|---|---|---|
| Doğrulama yamaları PSNR-Y (1800 yama, bicubic 34,34) | 33,65 dB | **34,51 dB** | +0,86 dB |
| Canlı, gerçek kare, segment 1 (n=91) | 33,86 / 0,8872 | **34,11 / 0,8948** | +0,25 dB, +0,008 SSIM (karelerin %100'ünde daha iyi) |
| Canlı, gerçek kare, segment 2 (n=88) | 33,64 / 0,9005 | **33,74 / 0,9081** | +0,11 dB, +0,008 SSIM (%83) |
| Canlı, ara kare, segment 1 (n=87) | 30,53 / 0,8417 | **30,64 / 0,8476** | +0,11 dB (%100) |
| Canlı, ara kare, segment 2 (n=85) | 31,79 / 0,8810 | 31,73 / **0,8847** | -0,06 dB, +0,004 SSIM (%62) |
| TensorRT SR 1920x1080 | 3,13 ms | 3,10 ms | aynı |
| Canlı çıkış (120 sn, fused-sr) | 59,92-60,0 FPS | 59,99-60,0 FPS, geç tik 0 | aynı |

**Sonuç:** Aynı hızda, gerçek karelerde PSNR ve SSIM'de ölçülebilir iyileşme. TOD benzeri girdide hazır model bicubic'i ancak yakalıyordu; ince ayarlı model gerçek karede bicubic'in 0,2-0,4 dB, SSIM'de 0,002-0,004 üstünde (üçlü tablo). Kazanç küçük: model çok küçük (24 kanal, 4 blok, 540p'de) ve sıkıştırma hasarını ancak biraz temizleyebiliyor.

## 1. Veri (C1)

- Havuz: 12 CC BY klip, 19,6 GB (`data/clips/manifest.json`). **Eğitim:** Fabio di Mauro Official'ın 10 amatör maçı (Eccellenza ve Interprovinciale, 2017-2022, iki farklı kamera dönemi). **Doğrulama:** 5bgF-5I2P_M (maç) ve O3gD6n0zoik (drone), eğitime hiç girmedi. Canlı puan segmentleri de doğrulama klibinden.
- Çift üretimi (`tools/make_pairs.py`): 45 sn'de bir 8 sn'lik segment -> `tod_sim` (4,8 Mbps H.264 High 50p, şeritsiz) -> sim karesi i ile GT karesi round(1,2·i) aynı anda çözülür. 4 karede bir, kare başına 6 rastgele yama; LR 128x128, HR 256x256 RGB; düz yamalar atılır (std > 6). Kareler diske yazılmaz, sim mp4'ler hemen silinir.
- **98.253 eğitim yaması** (123 shard, 11 GB sıkıştırılmış npz), klip başına 7-14 bin; üretim klip başına 6-13 dk (CPU). Doğrulama 1800 yama.
- İndirme: 8 ek klip, `-f 401/315` (AV1 ya da VP9 4K60), hız 2,7-20 MB/s arası oynadı.

## 2. Eğitim (C2)

`tools/train_sr.py`: RT4KSR eğitim biçimi (`rep=False`), hazır ağırlıktan başlar, bitince `rep_state_dict` ile tek 3x3'e birleşir (mimari ve hız aynı). L1 kaybı (RGB), Adam (0,9 / 0,99), kosinüs öğrenme oranı 2e-4 -> 2e-6, bf16 autocast, batch 16, yığın başına rastgele döndürme ve çevirme.

Laptop güvenliği: her 3 dk'da `weights/ft/<ad>/last.pt` (kaldığı yerden devam, EMA dahil), 50 adımda bir nvidia-smi sıcaklığı (>= 88 °C'de <= 78 °C'ye kadar duraklama), fiş çekilirse duraklama. İki eğitimde de duraklama olmadı (GPU en fazla 77 °C).

| Koşu | Veri | Adım | EMA | Süre | En iyi doğrulama |
|---|---|---|---|---|---|
| ft1 | 2 klip (19 bin yama) | 30 bin | yok | 23 dk | 34,42 dB (değerler ±0,15 dalgalı) |
| **ft2** | 10 klip (98 bin yama) | 60 bin | 0,999 | 31 dk (~32 it/sn) | **34,51 dB** (düzgün artış) |

Başlangıçta dokulu yamalarda hazır model bicubic'in 0,7 dB gerisindeydi (33,65 / 34,34): hazır model temiz bicubic küçültmeyle eğitilmiş, sıkıştırma bloklarını ve halkalarını "detay" sanıp büyütüyor (tahmin).

## 3. Derleme ve canlı ölçüm (C3)

- `tools/build_trt_sr.py --model rt4ksr-x2-ft2` ve `tools/build_trt_pipeline.py --sr rt4ksr-x2-ft2` (1920x1080 ve 1920x1020 tuvalleri). FP16 motor ile PyTorch farkı ort. 0,18/255. Gerçek kare motoru 3,00 ms, ara kare karma yol 11,14 ms (hazır modelle aynı).
- `load_sr` artık `rt4ksr-x2-<ad>` -> `weights/rt4ksr/<ad>_best.pth` okuyor; `watch --sr` ile seçilir.
- Canlı ölçüm: `tools/live_score.py <segment> --proc fused-sr --seconds 120 -- --sr <model>`, iki segment (5bgF-5I2P_M 200. ve 500. sn), CPU boşken. Puanlar ikili ortak GT karelerinde (`tools/score_table.py`).

Üçlü ortak kümede (daha az kare) ft1 de ölçüldü: segment 1 gerçek kare hazır 33,90 / ft1 34,14 / ft2 34,16 dB; segment 2 33,29 / 33,39 / 33,43 dB. ft2 her ikisinde en iyi.

Not: ft1'in ilk canlı ölçümü arka planda çift üretimi (CPU yükü) varken yapıldı ve geç tik %0,8-0,9 çıktı; CPU boşken aynı model geç tik 0 verdi. Canlı hat CPU yüküne duyarlı: maç sırasında ağır CPU işi çalıştırılmamalı.

## 4. Varsayılan

`WatchConfig.sr = "rt4ksr-x2-ft2"`. Ağırlık bu makinede yoksa `resolve_sr` hazır `rt4ksr-x2`'ye düşer ve `sr_yedek` olayı yazılır. Dayanıklılık sınavı yeni varsayılanla tekrarlandı (`runs/robust4_ft2`): 6/6 kontrol geçti, çökme 0.

Lisans: ft2 ağırlıkları RT4KSR (Apache-2.0) ağırlığından ve CC BY kliplerden türetildi. Yayınlanırsa klip atıfları (`data/clips/manifest.json` içindeki `atif`) README'ye eklenmeli.

## Açık sorular

- Kazanç küçük. Sonraki adımlar (fikir bankası): daha büyük ama aynı hızda model (ör. 32 kanal), Y/UV ayrı yol (F22), kayıpta SSIM/algısal terim, gerçek TOD'a daha yakın bozulma çeşitliliği (bitrate 3-6 Mbps, farklı GOP, çift sıkıştırma).
- Tek video çekimcisi: sahne ve kamera çeşitliliği sınırlı. Farklı CC BY kaynaklar aranmalı.
- Ara kare segment 2'de hafif düştü (-0,06 dB, karelerin %62'sinde iyi): uç kareler incelenmeli.
- Gerçek TOD'da göz testi (S tuşu bölünmüş ekran) Yiğit'le yapılmalı.

## Kaynaklar

- Zamfir vd. 2023, "Towards Real-Time 4K Image Super-Resolution" (RT4KSR), https://github.com/eduardzamfir/RT4KSR (Apache-2.0).
- Klipler ve lisanslar: `data/clips/manifest.json`.
- Kayıtlar: `runs/train_ft1`, `runs/train_ft2` (log.csv, olaylar.txt), `runs/c200_*`, `runs/c500_*`, `runs/k200_*`, `runs/k500_*`, `runs/robust4_ft2`.
- Kod: `tools/make_pairs.py`, `tools/train_sr.py`, `upscaler/models/sr.py`, `upscaler/watch.py`.
