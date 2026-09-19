# TV modu ve sıkıştırma onarımı v0

- **Tarih:** 2026-09-19
- **Durum:** araştırma (ölçüldü; bölümler aşağıda tamamlandıkça dolar)
- **Kapsam:** Bu akşamki maç için laptoptan HDMI TV'ye 1080p 50 FPS canlı hat (A), TOD sıkıştırmasını onaran ilk model (B), modelin TV moduna takılması ve TOD'da kıyas (C). F27'nin ilk adımı. Önceki durum: [hat14](2026-09-16-hat14-mac-gunu-surumu.md), [F26](2026-09-16-f26-tod-yumusaklik.md).

## A. TV modu (`watch --tv`)

Ne yapıyor: TOD Chrome penceresi laptop panelinde (1920x1080, 144 Hz) yakalanır, TV'de (HDMI, 1920x1080, 50 Hz, birincil olmayan ekran) gösterilir. Ara kare yok, SR yok: çıkış hızı kaynak hızı (50), her tik gerçek kareye oturur. Ekran 50 Hz olduğu için sunucu vsync kilidinde (swap interval 1). Gecikmeli ses TV'nin HDMI çıkışından (`--audio-device NVIDIA`).

Kod:
- `upscaler/process.py`: `PassProcessor` (kare olduğu gibi, gerekirse 1080p'ye sığdırma), `RepairProcessor` (C için).
- `upscaler/present.py`: `pick_monitor(prefer="external")`. İki ekranın adı da "Generic PnP Monitor" olduğu için ad değil "birincil olmayan" kuralı. Ekran değişimi karşılaştırması artık konum ve Hz de içeriyor (aynı adlı iki ekranı ayırt etmek için).
- `upscaler/watch.py`: `--tv` ön ayarı (1080p doku, 50 FPS, `pass`, `external`), `--repair <ad>`.

| Koşu | Süre | Çıkış FPS | Geç tik | Geç sunum | Hattın eklediği A/V (p5 / p50 / p95) | Mod |
|---|---|---|---|---|---|---|
| Test klibi, ffplay, `runs/tv_kisa1` | 60 sn | 50,001 | 0 | %0,10 | -5,9 / 6,7 / 15,0 ms | lock, interval 1 |
| **Test klibi, ffplay, `runs/tv_10dk`** | **10 dk** | **50,001** | **0** | %0,03 | **-12,9 / -2,1 / 11,1 ms** | lock |
| **TOD, `runs/tv_tod2`** | **2 dk** | **50,0** | **0** | 0 | ölçülemez (DRM, test deseni yok) | lock |
| Masaüstü kısayolu, TOD, `runs/kisayol_ham` | 20 sn | 50,002 | 0 | | | lock |

- 10 dk: işlem p50 0,2 ms (sadece kare seçimi), uçtan uca p95 21 ms (swap vsync'i bekliyor), VRAM 1210 MB baştan sona sabit, ses sert atlama 0, eksik 0.
- Doğrusal eğim "görüntü gecikmesi -37 ms/saat" verdi ama dakika ortalamaları 1497-1507 ms arasında, eğilim yok. Eğimi ilk dakikadaki tek aykırı ölçüm (1721 ms) çekiyor. Kayma birikmiyor.
- Gerçek kare tik oranı %93-95: kalan tiklerde kare seçimi zaman oranından yapılıyor (TV'nin 50 Hz'i ile kaynağın 50'si tam aynı değil). `pass` harman yapmaz, en yakın kareyi verir.
- TOD: giriş 6017 benzersiz kare / 120 sn = 50,1 FPS, saat periyodu 20,000 ms, ses TV çıkışında (aygıt 10, gecikme 22 ms).
- TOD'da A/V ölçülemez; hattın eklediği A/V içerikten bağımsız, test klibiyle ölçülen değer geçerli. TV'nin kendi ses/görüntü işleme gecikmesi (TV ayarı) hattın dışında.

Kısayol: masaüstünde `Maç TV ham (upscaler).bat` = `watch.bat --tv --audio-device NVIDIA`. Eski `Maç izle (upscaler).bat` değişmedi.
