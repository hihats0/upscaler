# Yol haritası (taslak v4: sıfır bütçe, iki paralel hat)

**AMAÇ: Canlı maçı (TOD) harici 4K ekranda 4K 60 FPS izlemek, 4070 Laptop'ta.**

## Şu an neredeyiz (2026-09-19 22:00, /goal AI yeniden çizim: 4/5 kriter geçti)

- ✅ ai_v4 (48x8x3) + güç + harman + renk motorda (9,0 ms), **ön işleme** (`AiAheadProcessor`). Kısayollar: "Maç TV AI" (güç 1,3, harman 0,85), "Maç TV AI kıyas", "Maç TV AI agresif" (1,45 / 0,9). Doğrulama: spk 0,390 / 0,427 (hedef 0,379), titreme 1,04 / 1,02, LPIPS 0,070 (ham 0,111).
- ✅ 87 dk canlı TOD maçı: 50,0 FPS, geç tik 0, AI hazır %99,06, VRAM sabit.
- ✅ TV takılmasının bizim tarafı düzeltildi (TV penceresi tam boy; 16:24'ten beri 144 Hz'de sunuyordu).
- ✅ Kriter 1: canlı TOD 2 dk spk AI 0,299 / ham 0,135 (maç sonrası yayın). ⏳ Yiğit 3 anlama sorusunu cevaplayacak (sorular: 540p yeniden çizim neden kötü, ön işleme takılmayı neden bitirdi, güç neden 1,6 -> 1,45).
- Sonra: `--dump-timing` ile bizim takılma payımız, F28 (TOD aksamasını RIFE ile doldur), daha büyük model (ön işleme sayesinde sığar).
- Rapor: `reports/2026-09-20-ai-yeniden-cizim.md`.

## Önceki durum (2026-09-19 akşam, /goal: AI yeniden çizim "S24 Ultra gibi")

Goal: canlı TOD TV'de AI kareyi yeniden çizsin (keskin çim, canlı renk, titremesiz), 1080p 50 FPS. Rapor: `reports/2026-09-20-ai-yeniden-cizim.md`.
- ✅ A tarama (`tools/ai_eval.py`, `runs/ai_eval/tarama1.json`): "küçült + GAN ile yeniden çiz" (540p/270p) PSNR -4..-8 dB, titreme 1,5-1,8: elendi. En iyi algısal: Real-ESRGAN x2 1080p'de + geri küçültme (LPIPS 0,100, spk 0,31) ama 2 sn/kare. Hazır hiçbiri canlıya sığmıyor + iyi değil.
- ✅ TRT hız (`runs/ai_bench.json`): AiNet 48x8x3 kare 11,9 ms, 48x6x3 9,7 ms, 32x6x3 6,5 ms.
- ✅ B veri: `data/pairs/ai_train_a/b` (~29 bin örnek, 10 klip), `ai_val` (932, 5bgF + O3gD, sabit 0,7/0,4). Girdi t-1..t+2, hedef t, t+1 (keskin 1080p).
- ⏳ C eğitim `tools/train_ai.py --name ai_v1` (48x8x3, L1 + LPIPS-VGG + GAN 0,05 + zamansal). Kayıt `runs/train_ai_v1`.
- Sonra: ai_eval ile tam kare ölçüm, `tools/build_trt_ai.py --name <ad>`, `watch --tv --ai <ad>` (işlemci `AiProcessor` hazır), TOD canlı ölçüm, kısayollar "Maç TV AI" ve "Maç TV AI kıyas".
- ❗ **MAÇTAN SONRA İLK İŞ (Yiğit, 2026-09-19 20:10): TV'de takılma.** AI ile "baya fark ediyor" dedi ama görüntü arada takılıyor; normal izlerken de (TOD'un kendisinde) oluyormuş. Veri: `runs/watch_20260919_*` (AI agresif, 7,5 dk): giriş tam 50 benzersiz kare/sn, geç tik 0, AMA çıkış TV'nin 50 Hz'ine kilitli değil (swap beklemiyor, ~110-140 sunum/sn), GPU 85-89 °C ve ısı kısıtlaması (0x20). Adaylar: (1) sunumu 50 Hz'e oturtmak (lock neden bloklamıyor: 16:xx TV testlerinde bloklıyordu), (2) ısı, (3) TOD'un kendi kare/bitrate dalgalanması. Önce ham modda ölç, sonra AI.
  - 20:15 KÖK SEBEP BULUNDU VE DÜZELTİLDİ: 16:24 DPI düzeltmesinden sonra TV penceresi 1920x1079 oldu, DWM birincil panelin 144 Hz'inde sundu (ham TV bile 144,06). Birincil olmayan ekranda pencere artık tam boy (`present.py`): TV kilidi geri geldi. Keskinlik probu (--probe-sharp) canlıda saniyede 2 tik kaçırıyordu, maç kısayolundan çıkarıldı. Kalan: 10 sn'de 1-4 kaçan vsync (AI 12-15 ms, ısıda saat 1935 MHz). **Çözüm: AI'yı kare gelince önceden çalıştır (1,5 sn tampon), sunum sadece hazır kareyi gösterir.**
  - 20:40 AI ön işleme devreye girdi (`AiAheadProcessor`): TV'ye 50,0 FPS, geç sunum ~0. Yiğit hâlâ "50 gibi değil" dedi. **Kaynak ölçümü** (`tools/content_fps.py`, Chrome penceresi, sadece sayı): 15 sn 50,06 FPS temiz; 60 sn 49,5 FPS, 2969 yeni karede 10 tekrar (~40 ms), 8 gecikme (25-35 ms), 2 uzun boşluk (en uzun 302 ms). **TOD/Chrome'un kendisi dakikada ~20 aksama + ara ara 0,3 sn donma.** Bizim payımız ölçülmedi: SIRADAKİ `--dump-timing` ile TV tikinde tekrar/atlama sayısı (kaynak 50,06 Hz vs TV 50,00 Hz faz kayması şüphesi). Fikir F28: kaynak aksamasını 1,5 sn tampon + RIFE ara kareyle doldurmak. TV çıkış penceresi tam ekran yolunda olduğu için WGC ile dışarıdan ölçülemiyor (1 kare).
- Devam komutu: aynı goal. Önce `runs/train_ai_v1/olaylar.txt`'e bak.

## Önceki durum (2026-09-19, /goal: bu akşamki maç için TV modu + onarım v0)

Hedef: laptop -> HDMI TV (1080p, 50 Hz) 1080p 50 FPS canlı hat + TOD sıkıştırmasını onaran ilk model (F27). Rapor: `reports/2026-09-19-tv-modu-ve-onarim-v0.md`.
- ✅ A0 commit.
- ✅ A TV modu `watch --tv`: test klibi 10 dk 50,001 FPS, geç tik 0, hattın eklediği A/V -2,1 ms; TOD 2 dk 50,0 FPS, geç tik 0, kilit modu (TV 50 Hz). Kısayol `Maç TV ham (upscaler).bat`.
- ✅ B onarım v0 (32x6, 5,9 ms): rep_v0 +0,31 dB ama hf düşüyor (detay siliyor); rep_v0b (genlik 8) +0,20 dB, hf 0,857 > 0,828.
- ✅ C: TOD canlı 5 dk onarımlı 50,0 FPS, geç tik 0, VRAM sabit. TOD keskinliği: onarımlı hamdan yumuşak (v0 -%21, v0b -%10). Kısayollar v0b.
- **Sıradaki:** model keskinleştirmiyor. Çok kareli onarım (F15/F27) ya da ince bandı koruyan kayıp / GAN son kat (F26). 4K (F27 adım 2) sonra.

## Önceki durum (2026-09-16 18:20, /goal A -> B -> C bitti; TOD canlı koşusu Yiğit'e bağlı)

Büyük hedef (Yiğit, 2026-09-16): A) Hat 1.4'ü bitir, B) Hat 1.3 canlı ölçüm + 2.0 mini veri + 2.1 TOD simülatörü, C) Hat 2.2 ilk ince ayar. Sırayla.

**A (Hat 1.4): ✅ bitti** (TOD canlı koşusu hariç). Rapor: `reports/2026-09-16-hat14-mac-gunu-surumu.md` (4K kontrol listesi içinde).
- ✅ A1 A/V: kayma kaynaktandı (ffplay). 60 dk: hattın eklediği 5,4 ms, kayma -0,7 ms/saat.
- ✅ A2 dayanıklılık: robust3 6/6 (tampon 1080p üstünü küçültür, yeniden ayırmada VRAM önbelleği boşalır).
- ✅ A3 kilit modu: 144 Hz'de 48 FPS interval 3, geç sunum 0.
- ✅ A4 (test klibiyle) 60 dk `--dump-timing --info`: 59,999 FPS, geç tik 5, VRAM sabit. ⏳ **TOD 60-100 dk: Yiğit TOD açınca** (`notes/soru-kuyrugu.md`).

**B (Hat 1.3 + 2.0 + 2.1): ✅ bitti.** Rapor: `reports/2026-09-16-hat13-canli-olcum.md`.
- ✅ B1: 4 CC BY 4K60 klip (4,19 GB), lisans sayfada doğrulandı, `data/clips/manifest.json`. Eğitim: 63pDvVidZJA, u5w_du22wSQ. Doğrulama/puan: 5bgF-5I2P_M, O3gD6n0zoik (eğitime girmez).
- ✅ B2: `tools/tod_sim.py` (4800 kbps High 50p, hizalama 40/40).
- ✅ B3: `tools/live_score.py` / `watch --score`: 60 FPS korunuyor, kaydırma testi -10,5 dB.
- ✅ B4: bicubic 33,70 / RT4KSR 33,84 dB gerçek kare; RIFE ara kare +4,6 dB. Segmentler: `data/sim/5bgF-5I2P_M_s200_l10_50p.json`, `..._s500_...`.

**C (Hat 2.2 ilk ince ayar): ✅ bitti.** Rapor: `reports/2026-09-16-hat22-ilk-ince-ayar.md`.
- ✅ C1: 12 klip 19,6 GB; 98 bin eğitim yaması (11 GB), doğrulama 1800 (ayrı klipler).
- ✅ C2: ft2 (10 klip, 60 bin adım, EMA 0,999, 31 dk, maks 77 °C): doğrulama 34,51 dB (hazır 33,65).
- ✅ C3: TensorRT aynı hız (3,10 ms). Canlı gerçek kare +0,25 / +0,11 dB, SSIM +0,008; 60 FPS, geç tik 0. **Watch varsayılanı `rt4ksr-x2-ft2`** (ağırlık yoksa hazır modele düşer). robust4_ft2 6/6.

**Sıradaki (2026-09-16 akşam):** F26 devam: GAN denemesi (g01/g05, 80/70 °C sınırıyla), sonra canlı puan (`data/sim/*_tod`) + titreme, TOD'da `sharpness_probe --sr`. Rapor: `reports/2026-09-16-f26-tod-yumusaklik.md`.

**Önceki öneri:** TOD'da 60-100 dk canlı sınav (Yiğit TOD açınca), 4K ekran kontrol listesi, sonra Hat 2.2b: daha çeşitli CC kaynak + bozulma çeşitliliği + aynı hızda biraz büyük model, ya da Hat 2.3 (kendi video modelimiz). Fikirler: `notes/fikir-bankasi.md`.

## Önceki durum (2026-09-15 18:20, Hat 1.4 yarıda: kullanım limiti %91, durduruldu)

Görev: tek komutla TOD canlı maçını 4K60, senkron sesle, 90+ dk kesintisiz izlemek. Paketler:

1. ✅ **Boyut değişimi:** 1920x1080 kaynaşık motorlar (gerçek kare 2,99 ms, ara kare 11,13 ms). `process.plan_input` + `fit_bgra`, `GpuFrameRing` generation. Test `tests/test_reconfigure.py`. Commit efb3184.
2. ✅ (4K ekran hariç) **GPU sunucu** `upscaler/present.py`: glfw + GL + CUDA-GL interop (doku birebir doğrulandı), araç penceresi, 1 px kısa (Chrome örtülme), Esc/S/I, ekran değişiminde yeniden açma. TOD watch 180 sn: uçtan uca p95 13,89 ms, sunum p95 1,72 ms, geç sunum %0,075, geç tik 0. Laptop 144 Hz: "timer" modu. ⏳ Kilit modu donanımda denenmedi: `--out-fps 48` ile (144/3) dene.
3. ⏳ **Ses** `upscaler/audio.py`: çalışıyor (TOD 180 sn: hata p99 3,4 ms, sert atlama 0). A/V 3 dk (`tools/av_sync_test.py --seconds 180 --run-name av_3dk`): hattın eklediği fark medyan 2,8 ms, AMA çıkış A/V 3 dk'da 36,9 -> 58,6 ms kaydı (eğim 680 ms/saat). **Sıradaki ilk iş:** `runs/av_3dk` kaydıyla bunun gerçek kayma mı ölçüm hatası mı olduğunu ayır (AvProbe eşleştirme penceresi 2 sn flaş periyoduna göre dar olmalı, goruntu/ses gecikmesi -491 ms yanlış eşleşme). Olası gerçek sebep: `lag_ema` (sunum gecikmesi) ya da CaptureClock alt zarfı. Sonra 60 dk A/V koşusu (drift <= 20 ms).
4. ✅ (ölçüm kaldı) `python -m upscaler watch`, `watch.bat`, masaüstü `Maç izle (upscaler).bat`, README. İlk kare 7,1 sn (süreç başlangıcından). Kapanış temizliği 0,09 sn; süreç çıkışı dahil ölçülmedi (olaylarda `unix` alanı var).
5. ⏳ **Dayanıklılık:** kod watch.py'de (yönetici iş parçacığı, pencere/yakalama/ses/sunucu toparlama, donma boşluğunda tutma `schedule.GAP_PERIODS`, runs/ kayıtları, nvidia-smi). Sınav aracı yazıldı, **koşulmadı:** `tools/robustness_test.py --run-name robust1`.
6. ⏳ 60-100 dk canlı TOD sınavı (watch --dump-timing --info). GPU 180 sn'de 81 °C'ye çıktı, kısıtlama kolonu izlenmeli. RSS ~3 MB/dk arttı (istatistik listeleri array'e çevrildi, uzun koşuda tekrar bak).

Sonra: `reports/2026-09-15-hat14-mac-gunu-surumu.md`, vault notu, commit. 4K kontrol listesi raporda.

## Önceki durum (2026-09-15 17:00, Hat 1.2 çıkış kriteri karşılandı)

- ✅ **Gerçek SR modeli hatta:** RT4KSR x2 (Apache-2.0), TensorRT 1080p 3,1 ms, doğal görüntüde bicubic'ten +2,9 dB. EfRLFN elendi (1080p'de 105 ms).
- ✅ **TOD canlı RIFE + RT4KSR 4K60** (`--proc fused-sr`, 70 sn): çıkış 59,97 FPS, geç tik %0,05, işlem p95 13,85 ms, VRAM 1,45 GB. Rapor: `reports/2026-09-15-hat12-sr-modeli.md`.
- ✅ `--split` (sol SR, sağ bicubic; `fused-sr --split` önizlemesiz 60,0 FPS, p95 15,5 ms), `--dump-timing` + `tools/analyze_timing.py`.
- ✅ Düzeltilen hatalar: alpha artık sıra numarasından (gerçek kare tik 0,118 -> 0,167); saat 48 Hz tuzağı (kilitten sonra 6 sn'lik kayan periyot doğrulaması, canlıda doğrulandı).
- ⚠️ 44,6 FPS anomalisi 4 koşuda tekrarlanmadı, kayıt aracı hazır.
- **Sıradaki:** gecikmeli ses çalma (Chrome çıkış yönlendirmesi sorusu bekliyor), 1920x1080 (tam ekran) motorları, harici 4K ekranda sunum maliyeti, Hat 1.3 (canlı ölçüm) ya da Hat 2.0 (CC veri, TOD simülatörü; RT4KSR'nin H.264 hasarındaki kalitesi).

## Önceki durum (2026-09-15 akşam, limit nedeniyle durduruldu)

- **Hat 1.1 iskelet kodu var, Hat 1.2 başladı.** Rapor: `reports/2026-09-15-hat11-canli-iskelet-ve-hiz.md`.
- Canlı hat v0 TOD'da 50→60 FPS 4K (bicubic) akıyor. Ses yakalama çalışıyor, gecikmeli çalma yok.
- ✅ Zaman çizgisi "144 Hz yetişme karesi" hatası çözüldü: tampon "tekrar" kararını bir kare erteliyor, sıra atlarsa boşluğu dolduruyor (`ring._resolve_tentative`). Tekrar oynatma max 0,59 ms, canlı desen p99 0,29 / 0,17 ms.
- ✅ **rife-flow-trt: ara kare 11,9 ms, 50→60 bütçe 629 ms/sn: SIĞIYOR** (%26 pay). Motorlar: `weights/trt/rifeflow_4.25_960x576_fp16.engine` (1080p kaynak), `..._960x512_...` (1920x1020 pencere). Doğrulama: çıktı farkı ≤0,004/255.
- ✅ **TOD canlı, rife-flow-trt (RIFE ara kare + bicubic 4K): çıkış 59,95 FPS, 20 sn'de 1 geç tik, işlem p50 13,6 / p95 14,9 ms, VRAM 1,35 GB.** Pay dar: çıkış tikleri kaynak ızgarasına hizalı değil, 60 tikin hepsi RIFE çalıştırıyor.
- ✅ Çıkış 1/300 sn ızgarasına hizalandı + saat kilitlenmeden çıkış yok (`live.py`, `--no-snap` ile kapatılır). Desen: p99 0,46 ms, max 0,53 ms, gerçek kare tik 0,165. TOD (rife-flow-trt): çıkış 60,0, 0 geç tik, işlem p50 6,5 / p95 14,4 ms.
- ⚠️ Açık: o TOD koşusunda giriş 44,6 FPS ve gerçek kare tik oranı 0,75 (beklenen ~0,17). Yayın takılması mı, hizalama/tutma etkileşimi mi? `--dump-timing` benzeri kayıtla incelenmeli (live.py'de sadece LiveConfig.dump_timing var, CLI bayrağı yok).
- **Sıradaki:** yukarıdaki anomali, render-ahead/batch, gecikmeli ses çalma, SR modeli (EfRLFN x2), tam ekran TOD ölçümü. Kod commit edilmedi.
- Kod commit edilmedi.

## Önceki durum (2026-09-15 12:45)

- Plan v4 ve bulut kuralları **onaylandı**. Kod yok (sadece scratchpad ölçüm betikleri).
- **Faz 0 (Hat 1.0 keşif) ölçümleri bitti** (agent yaptı). Rapor: `reports/2026-09-15-faz0-kesif-tod-yakalama.md`.
  1. ✅ TOD Chrome'da yakalanabiliyor, siyah değil.
  2. ✅ Kaynak **1080p 50 FPS** (720p25 değil), H.264 High, ~4,8 Mbps, Widevine L3.
  3. ⏳ Kalan: harici 4K ekran bağlanıp test edilecek. Yazılım ortamı kararı (Yiğit ile).
- **Etkisi:** Büyütme x3 değil x2, ara kare 50→60 (çok daha kolay). SR hız bütçesi 1080p girdiyle yeniden hesaplanacak. proje notundaki kaynak satırı düzeltildi.
- Sonra: Hat 1.1 (iskelet) tasarımı.
- Brainstorming süreci: yol haritası onaylandı. Bir sonraki adım faz tasarımı, sonra spec ve plan.

Durum: Taslak (2026-09-15). Yiğit: "Çok param yok." Plan **0 TL** varsayımıyla kuruldu. Kural hâlâ: **her şey canlı.** Kod yok, onay bekleniyor.

## Bütçe prensipleri

1. **Varsayılan harcama 0.** Yiğit az dolarla buluta açık, **ama sadece iyi sonuç çıkacaksa** (2026-09-15). Kurallar (**onaylandı**, 2026-09-15):
   - **Kanıt kapısı:** Bir fikir önce laptopta ya da Kaggle'da küçük ölçekte umut vermeli (rakibi ölçülebilir şekilde geçmeli). Kanıt yoksa buluta para yok.
   - **Koşu tavanı:** Tek bulut koşusu en fazla $10 (öneri).
   - **Toplam tavan:** Yiğit yeni onay verene kadar toplam en fazla $30 (öneri).
   - Her koşudan önce tahmini süre ve tutar Yiğit'e söylenir. Koşu bitince veri silinir.
2. **Tüm yazılımlar bedava:** Python, PyTorch, TensorRT, ffmpeg, OBS.
3. **Ana eğitim cihazı laptop** (gece çalışır). **Yardımcılar:** Kaggle (haftada ~30 saat, arka planda çalışır) ve Colab (yedek).
4. **Model zaten küçük olmak zorunda.** 4070 Laptop'ta canlı 4K60 çalışacak bir modeli eğitmek de aynı laptopa sığar. Pahalı olan tek şey dev öğretmen modellerdi. **Şimdilik öğretmen yok**, cevap anahtarı temiz 4K Creative Commons klipler.
5. **Önce ince ayar, sonra sıfırdan.** Hazır ağırlıklarla başlamak eğitimi günlerden gecelere indirir.
6. **Para yerine zaman harcıyoruz.** Bulutta saatler süren iş, laptopta birkaç gece sürer.

## Neden canlı hat hâlâ gerekli?

- Model tek başına maç göstermez. Yakalama, tampon, ses ve 4K60 çıktı olmadan canlı çalışamaz.
- Hazır modeller (RIFE, EfRLFN) rakibimiz. Onları görmeden "bizimki daha iyi" diyemeyiz.

## Hat 1: Canlı hat (sahne), 0 TL

| Adım | Ne yapılır | Çıkış kriteri |
|---|---|---|
| 1.0 Keşif | OBS ile TOD siyah ekran testi, `chrome://media-internals`, harici 4K ekran, yazılım ortamı | TOD yakalanabiliyor mu, cevabı belli |
| 1.1 İskelet | Pencere ve ses yakalama, tampon, 4K60 çıktı, senkron ses. Basit büyütme. | Maç 1-2 sn gecikmeyle, sesle, takılmadan akıyor |
| 1.2 Rakipler | RIFE + EfRLFN hatta. İkiye bölünmüş ekran. RTX VSR kıyası. | Canlı 4K60, rakip kalitesi görülüyor |
| 1.3 Canlı ölçüm | Test klibi oynatıcıda, aynı hattan geçer, puan ekranda | Canlı puan |

## Hat 2: Kendi modelimiz (mutfak), hemen başlar, 0 TL

| Adım | Ne yapılır | Nerede | Çıkış kriteri |
|---|---|---|---|
| 2.0 Veri | YouTube Creative Commons futbol klipleri, önce 4K. **Toplam ≤ ~40 GB**, sıkıştırılmış tutulur. | Laptop | İlk temiz klip havuzu |
| 2.1 TOD simülatörü | Klipler 720p25 ve düşük bitrate H.264'e çevrilir. Değerler Hat 1.0'da ölçülenlere göre ayarlanır. | Laptop | Çıktı gözle TOD'a benziyor |
| 2.2 İlk ince ayar | Hazır EfRLFN/RIFE + futbol + TOD bozulması | Laptop gece (tahmin 2-4 gece) + Kaggle | Rakipten ölçülebilir iyi |
| 2.3 Kendi küçük video modelimiz | Gelecek karelere bakan, çok kareli, ara kareyi de yapan model (F1, F2, F3, F15, F16). Canlı 4K60'a sığacak boyutta. | Laptop geceleri + Kaggle (haftalar sürebilir) | Aynı hızda ince ayarlı modelden iyi |
| 2.4 (Kanıt kapısından sonra) Bulut. **Maç sırasında bulut yok:** bulut sadece modeli hazırlar, çıkan model laptopta canlı 4K60 çalışır. | Laptopta umut veren modeli bulutta büyütmek (daha çok veri, daha uzun eğitim) ya da SwiftVR öğretmeniyle 1080p klipleri 4K hedefe çevirmek (~$2-3). | Vast.ai RTX 4090 (~$0.39/sa) | Koşu başına ≤ $10, harcanan her dolar ölçülebilir kalite artışı getirmeli |

## Buluşma

- Hat 2'nin her modeli Hat 1'e takılır, rakiple canlı ölçülür. **Şart: canlı 4K60.**
- Sonra: zayıf sistemler ve GitHub yayını.

## Laptop sağlığı (gece eğitimleri için)

- Prize takılı çalışır.
- Sıcaklık ve güç kaydı tutulur (`nvidia-smi` ile). Belirlenen sıcaklığın üstünde eğitim kendini duraklatır.
- Soğutucu altlık önerilir. Gerekirse GPU güç sınırı düşürülür (ayarı Yiğit yapar).
- Eğitim her birkaç dakikada kayıt (checkpoint) alır. Kesilirse kaldığı yerden devam eder.

## Bilinen kısıtlar

- Disk 111 GB boş, RAM 16 GB.
- Yayınlanacak ağırlıklar sadece lisansı temiz kaynaklardan türetilir.
- Maliyet ayrıntısı (paralı senaryolar, referans için): [reports/2026-09-15-egitim-suresi-ve-vast-ai-maliyeti.md](../reports/2026-09-15-egitim-suresi-ve-vast-ai-maliyeti.md)
