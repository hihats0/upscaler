# Nasıl yapacağız

Durum: Güncellendi (2026-09-15). Faz ayrıntıları: [yol-haritasi.md](yol-haritasi.md). Kod yok.

## Temel kural: HER ŞEY CANLI

- Ürün her fazda canlı pencere (TOD) üzerinde gerçek zamanlı çalışır. 1-2 sn gecikme ve senkron ses kabul.
- Çevrimdışı ürün fazı yok. v1'deki "çevrimdışı referans hattı (tavan)" kaldırıldı. "Tavan" artık **laptopta canlı yetişebilen en iyi kalite** demek.
- Ölçüm de canlı: test klibi oynatıcı penceresinde oynatılır, aynı hattan geçer, orijinalle kıyaslanır.

## Hedef

- Harici 4K ekranda 4K 60 FPS. Kaynak TOD (Chrome, 720p 25 FPS).
- Önce laptopta en yüksek canlı kalite, sonra zayıf sistemlere iniş.
- GitHub'da açık kaynak.

## Boru hattı taslağı

```
Kaynak penceresi (TOD ya da test oynatıcısı) + uygulama sesi
  -> Zaman damgalı tampon (1-2 sn, gelecek kareler)       <- F15, F17
  -> Tekrar eden kareleri ayıkla, sahne kesmesi algıla     <- F6
  -> Hareket haritası (düşük çözünürlükte, bir kez)        <- F1
  -> Gerçek kareleri çift yönlü güçlü büyütme              <- F2, F3, F15
  -> Ara kareler: çok kareli, warp + düzeltme              <- F2, F16
  -> Sabit katmanlar (skor vb.)                            <- F7
  -> Kalite katmanı tampon durumuna göre seçilir           <- F19, F20
  -> Görüntü ve ses aynı saatle ekrana ve hoparlöre        <- F18
```

## Ölçüm prensipleri

- Kalite hem sayıyla (PSNR, SSIM, LPIPS, zamansal titreme) hem gözle ölçülür. Top, çizgiler, forma numaraları ve skor tabelasına bakılır.
- Hız ortalama verimle ölçülür. Tampon hiç boşalmamalı.
- Kıyas ölçütü: aynı klipte RTX VSR (+ gerekirse Lossless Scaling). Geçemiyorsak dürüstçe söyleriz.
