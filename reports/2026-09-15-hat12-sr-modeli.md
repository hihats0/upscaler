# Hat 1.2: Gerçek SR modeli canlı hatta (RIFE + RT4KSR, TOD 4K60)

- Tarih: 2026-09-15 (akşamüstü, agent otonom, Yiğit dışarıda)
- Durum: **Ölçüm (onaylı veri)**
- Kapsam: SR modeli seçimi ve hız ölçümü, canlı hatta entegrasyon, TOD canlı koşuları, zaman çizgisi hizalama hatası, giriş FPS anomalisi, bölünmüş ekran kıyas modu, kaynaşık TensorRT motoru.
- Önceki: [2026-09-15-hat11-canli-iskelet-ve-hiz.md](2026-09-15-hat11-canli-iskelet-ve-hiz.md)

## 1. SR modeli seçimi (1080p girdi -> 4K, 4070 Laptop)

Bütçe: 60 çıkış/sn, ara kare yolunda akış (~5 ms) + warp (~3 ms) zaten var. SR için kare başına ~7 ms kalıyor.

| Model | Lisans | Nerede hesaplar | Eager fp16 | TensorRT fp16 | Karar |
|---|---|---|---|---|---|
| EfRLFN x2 | MIT | tam 1080p, 52 kanal, 6 blok | 306 ms | 105 ms | **Elendi** (15 kat yavaş) |
| RT4KSR x2 (Rep) | Apache-2.0 | 540p (PixelUnshuffle), 24 kanal, 4 blok | 22 ms | **3,13 ms** (1020p: 2,90) | **Seçildi** |
| RVSR | Apache-2.0 | x4, 540p girdi için | ölçülmedi | ölçülmedi | 1080p kaynağı yarıya indirmek bilgi kaybı |

- EfRLFN makalesindeki 271 FPS, 360x480 girdiyle ölçülmüş (1080p'nin 1/12'si). Tam çözünürlükte çalışan hafif ağlar bellek bant genişliğine takılıyor.
- RT4KSR NTIRE 2023 Real-Time 4K SR yarışmasının resmi taban modeli. Checkpoint eğitim biçiminde (1x1-3x3-1x1 + kısa yollar). Tek 3x3'e birleştirme (reparam) uygulandı, TF32 kapalıyken fark < 1e-4 (`tests/test_sr.py`).
- Kalite (doğal görüntü, RT4KSR teaser, 2 kat bicubic küçültüp geri büyütme): **RT4KSR 35,62 dB, bicubic 32,67 dB, bilinear 31,24 dB.** TOD sıkıştırma hasarı içermez; gerçek TOD kalitesi Hat 2'de ölçülecek.
- FP16 tuzağı: LayerNorm `eps=1e-6` FP16'da sıfıra yuvarlanır, düz bölgede (siyah bant) 0/0 = NaN. Norm float32'de hesaplanıyor.

## 2. Kod

| Dosya | Ne yapar |
|---|---|
| `third_party/rt4ksr/arch.py` | RT4KSR mimarisi + `rep_state_dict` (Apache-2.0, değişiklikler dosya başında) |
| `third_party/efrlfn/arch.py` | EfRLFN (MIT), kıyas için duruyor |
| `upscaler/models/sr.py` | SR kayıt/yükleme (`efrlfn-x2`, `rt4ksr-x2`) |
| `tools/build_trt_sr.py` | SR -> ONNX -> TensorRT, eager ile kıyas |
| `upscaler/process.py` | `SrUpscaler`, `SrProcessor` (ara kare kaynak çözünürlükte, sonra SR; `split`), `FusedSrProcessor` |
| `upscaler/models/fused.py`, `tools/build_trt_pipeline.py` | Kaynaşık motor (bölüm 6) |
| `upscaler/live.py` | `--proc rife-flow-trt-sr / fused-sr`, `--sr`, `--split`, `--dump-timing` (kare + tik CSV) |
| `tools/analyze_timing.py` | Zaman kaydı analizi |

Ağırlıklar `weights/` altında (gitignore): `efrlfn/EfRLFN-2x.pt` (yazarın Google Drive linki), `rt4ksr/rt4ksr_x2.pth` (GitHub deposu).

## 3. İlk canlı TOD koşusu (ayrı parçalar)

`--proc rife-flow-trt-sr`, TOD beIN Sports 1 (1920x1020 pencere), 70 sn, gecikme 1,5 sn, önizleme yok:

| Ölçüt | Değer | Hedef |
|---|---|---|
| Çıkış FPS (aktif) | 60,0 (4098 kare) | >= 59,9 |
| Geç tik | 0 | <= %1 |
| İşlem p50 / p95 / max | 14,34 / **15,83** / 18,38 ms | p95 < 16 |
| VRAM tepe | 1,44 GB | < 7 GB |
| Giriş | 49,96 FPS | |

Hedefi karşıladı ama p95 payı 0,17 ms. Profil (1920x1020, ara kare, senkronlu): BGRA->RGB 0,5 + küçültme 0,5 + akış TRT 5,0 + warp 2,8 + SR TRT 3,2 + çıktı uint8 2,1 = 14,1 ms (senkronsuz 12,1).

### 3b. Karma yol canlı TOD koşusu (çıkış kriteri koşusu)

`--proc fused-sr` (bölüm 6), hizalama düzeltmesi (bölüm 4) dahil, aynı kanal, 70 sn, önizleme yok. Kayıt: `runs/tod_fused_sr_70s.csv` (sadece zaman sayıları).

| Ölçüt | Değer | Hedef |
|---|---|---|
| Çıkış FPS (aktif) | **59,97** (4095 kare) | >= 59,9 |
| Geç tik | **2** (%0,05; tek bir 60 ms sıçraması) | <= %1 |
| İşlem p50 / p95 / p99 / max | 11,36 / **13,85** / 14,30 / 60,18 ms | p95 < 16 |
| Ara kare p50 / p95, gerçek kare p50 | 12,07 / 13,91, 4,45 ms | |
| VRAM tepe | **1,45 GB** | < 7 GB |
| Giriş | 50,34 FPS, saat 20,0 ms | |
| Gerçek kare tik oranı | 0,167 (beklenen 10/60) | |
| Beş bölümde işlem p95 | 14,15 / 13,82 / 13,73 / 13,78 / 13,67 ms | kararlı |

**Hat 1.2 çıkış kriteri karşılandı:** TOD canlı yayını RIFE ara kare + RT4KSR SR ile 1080p50 kaynaktan 4K 60 FPS, p95 payı 2,15 ms.

## 4. Zaman çizgisi hizalama hatası (bulundu, düzeltildi)

`--dump-timing` analizi: gerçek kare tik oranı 0,118 (50->60 ızgarasında beklenen 10/60 = 0,167). `(s - a_t) * 300` küsuratı p50 0,09, p99 0,38.

- Sebep: kare damgası `t`, kare geldiği andaki ofset tahminiyle yazılıp sabit kalıyor. Hizalama ise o anki `clock._offset`'i kullanıyor. Ofset kaydıkça alpha 1/6 ızgarasından kayıyor, gerçek kare olması gereken tikler küçük alpha'yla RIFE'dan geçiyor.
- Düzeltme (`live.py`): ardışık iki karede alpha, zaman farkı yerine sıra numarasından hesaplanır: `u = round((s - origin)/period * 6)/6`, `alpha = u - a.index`.
- Doğrulama: gerçek kare tik oranı 0,118 -> **0,157** (koşu A, `rife-flow-trt`, 25 sn) ve **0,167** (bölüm 3b, 70 sn, beş bölümde 0,16-0,17).

## 5. Giriş FPS anomalisi ve saat kilidi tuzağı

Önceki rapordaki koşu: giriş 44,6 FPS, gerçek kare tik 0,75. O koşunun zaman kaydı yoktu.

Bu oturumdaki TOD koşuları (hepsi `--dump-timing` ile):

| Koşu | İşlemci | Önizleme | Giriş FPS | Saat | Tekrar / boşluk | Gerçek kare tik |
|---|---|---|---|---|---|---|
| 3 | rife-flow-trt-sr | yok | 49,96 | 20,0 ms | 5 / 0 | 0,118 (hizalama hatası) |
| A | rife-flow-trt | yok | 50,31 | 20,0 ms | 1 / 0 | 0,157 |
| **B** | rife-flow-trt-sr | **açık** | 49,86 | **20,833 ms** | **59 / 32** | 0,199 |
| 3b | fused-sr | yok | 50,34 | 20,0 ms | 32 / 0 | 0,167 |

**44,6 FPS tekrarlanmadı.** Ama B'de aynı aileden, tekrarlanabilir bir hata çıktı: **saat 50 FPS kaynağı 48 Hz'e kilitledi.**

- Mekanizma: 144 Hz ekranda 50 FPS kaynak 3-3-...-2 vsync düzeniyle basılır, WGC zamanları 6,94 ms ızgarasına yapışır ve 13,9 / 27,8 ms titrer. `estimate_period` 1 sn'lik ısınmada faz tutarlılığını ölçer. B'nin ilk 49 karesinde skorlar: **48 Hz 0,277, 60 Hz 0,183, 50 Hz 0,078.** Aynı kaydın ilk 300 karesinde: 50 Hz 0,45, 48 Hz 0,08.
- Etkisi: yanlış periyotla sıra numaraları her ~25 karede çakışır: 59 "tekrar", 32 "boşluk doldurma", sıra farkı 0 (91 kez) ve 2 (46 kez), alpha ızgarası bozulur.
- Çevrimdışı deney (ilk kilit anındaki pencere): B 1,5 sn'de hâlâ 48, 2-3 sn'de 50. Diğer 3 TOD kaydı ve 6 test kaydı her pencerede 50.
- Denenen ve elenenler:
  - Isınmayı 2 sn'ye uzatmak: tuzağı çözdü ama 6 birim testi kaldı. Kilitten önce titrek damgayla giren kare sayısı artıyor, ring tekrar oynatmasında bir karelik (20 ms) hatalar çıktı.
  - "Sadece tam katsa uzun periyodu seç" kuralı: 60 Hz ekranda 25 FPS'i 59,94'e kilitledi.
- **Düzeltme: kilit sonrası kayan periyot doğrulaması** (`SourceClock`): ilk kilit 1 sn'de kalır. Her 100 kabul edilen karede son 6 sn'lik ham zamanlarla periyot yeniden tahmin edilir. **Arka arkaya iki** uyuşmazlıkta saat yeniden kilitlenir (`relocks`), geçmiş yeniden numaralanır, `GpuFrameRing` tampondaki damgaları yeniden kurar.
- İlk deneme (3 sn'lik tek seferlik doğrulama) B'yi çevrimdışı düzeltti ama canlı split koşusunda (önizleme açık) yine 48'de kaldı. O kayıtta 3 sn'lik kayan pencereler 57 denemenin 6'sında 48 dedi. 6 ve 10 sn'lik pencereler bütün kayıtlarda her seferinde 50 dedi.
- Çevrimdışı tekrar: B 8,13 sn'de, canlı split kaydı 12,52 sn'de 20,0 ms'e düzeldi. 3 temiz TOD kaydında yeniden kilit yok.
- **Canlı doğrulama** (split + önizleme, 60 sn): saat 20,0 ms ile bitti, `yeniden_damgalama` 2 (ilk kilit + yeniden kilit), gerçek kare tik 0,168.
- Regresyon testi: `tests/test_clock_lock.py`. İki tuzak kaydı `tests/data/tod_window_144hz_48hz_trap{,2}.csv` (sadece sayı): temizlerde 0, tuzaklarda 1 yeniden kilit.

44,6 FPS için olası sebepler (**tahmin**, doğrulanmadı):
1. `capture.py` art arda aynı görünen kareleri atar (8 pikselde bir örnek). Durağan içerik (sabit reklam görüntüsü, grafik, siyah geçiş) giriş FPS'ini düşürür.
2. TOD oynatıcı takılması. Bu oturumda bir kez "Bir hata oluştu (SA-...)" ekranı görüldü, o sırada WGC 0,06 FPS verdi.
3. Yanlış saat kilidi gerçek kare tik oranını saptırır (B'de 0,199). 0,75'i tek başına açıklamıyor.

Artık her koşu `--dump-timing` ile kaydedilebiliyor, `tools/analyze_timing.py` bölüm bölüm giriş FPS'i, sıra farklarını ve tik oranlarını gösteriyor. Anomali tekrar görülürse kaynak bu kayıtla ayırt edilecek.

## 6. Kaynaşık TensorRT motoru

Amaç: p95 payını açmak. `tools/build_trt_pipeline.py`, 1920x1020 -> 4K, sentetik ama doğal görüntüye benzeyen kareler (3-6 px kaydırma), senkronlu ölçüm:

| Yol | Ayrı parçalar | Motor | Çıktı farkı |
|---|---|---|---|
| Tam kaynaşık ara kare (BGRA -> küçültme -> akış -> warp -> SR -> 4K uint8) | 12,12 ms | 12,60 ms | ort. 0,46/255 |
| Gerçek kare `StillSR` (BGRA -> SR -> 4K uint8) | 4,87 / 7,40 ms | 2,82 / 3,13 ms | ort. 0,26/255 |
| **Karma ara kare**: akış TRT + warp PyTorch, `RgbSR` motoru (SR + BGR + uint8 + siyah bant) | 12,14 ms | **10,24 ms** | 0,000 |

- Tam kaynaşık motor hızlanma vermedi (akış + warp kısmı TensorRT'de PyTorch'tan hızlı değil, motor 142 MB). Kullanılmıyor, `--full` ile derlenebilir.
- Kazanç çıktı tarafında: 4K float -> uint8, kanal çevirme ve siyah bant kopyası motorun içinde kaynaşıyor.
- Seçilen: `--proc fused-sr` (`FusedSrProcessor`): ara karede karma yol, gerçek karede `StillSR`. Motor yoksa ayrı parçalara düşer.
- Önizleme açıkken (pygame, 4K -> 540p + CPU kopyası) işlem ~3-4 ms artıyor (bölüm 5, koşu B). Ölçümler önizlemesiz.

## 7. Bölünmüş ekran kıyas modu

`--split`: sol yarı SR, sağ yarı aynı karenin bicubic büyütmesi, ortada 4 px beyaz çizgi. `process.overlay_bicubic_right` ortak yardımcı: SR çıktısının üstüne sağ yarıyı yazar, sadece sağ yarının kaynak sütunlarını büyütür (tam 2x ölçekte tam görüntüyle birebir aynı). Birim testleri: `test_split_left_sr_right_bicubic_with_divider`, `test_sr_letterbox_keeps_aspect_and_black_bars`.

TOD canlı (1920x1020):

| Sürüm | Önizleme | Çıkış FPS | Geç tik | İşlem p50 / p95 | Saat |
|---|---|---|---|---|---|
| v1: ayrı parçalar, tam 4K bicubic, 4K float alan küçültmeli önizleme | açık, 60 sn | 51,2 | 512 | 20,6 / 21,7 ms | 20,833 (tuzak) |
| v2: sağ yarı bicubic, seyreltmeli önizleme | açık, 60 sn | 59,27 | 42 | 18,0 / 19,1 ms | 20,0 (canlı yeniden kilit) |
| **v3: karma kaynaşık yol** (`--proc fused-sr --split`) | **yok**, 30 sn | **60,0** | **0** | 13,97 / **15,46 ms** | 20,0 |
| v3 | açık, 30 sn | 59,86 | 4 (%0,24) | 15,94 / 17,25 ms | 20,0 (canlı yeniden kilit) |

Kıyas modu önizlemesiz hedefin içinde. Önizleme (pygame + CPU kopyası) hâlâ ~2 ms ekliyor. Gerçek 4K çıkış yolu GPU'da kalınca bu maliyet değişecek.

## Açık sorular

- RT4KSR TOD'un H.264 hasarında ne kadar iyi? (Hat 2: TOD simülatörü + CC klipler, ince ayar.)
- Harici 4K ekranda sunum (present) maliyeti ölçülmedi. Ölçümler önizlemesiz, `torch.cuda.synchronize()` ile. pygame önizlemesi ~3-4 ms ekliyor; gerçek çıkış yolu (mpv türü, GPU'da kalan sunum) gerekli.
- Ses gecikmeli çalma hâlâ yok (Chrome çıkış yönlendirmesi sorusu `notes/soru-kuyrugu.md`).
- Kaynaşık motorlar sadece 1920x1020 (ekranı kaplayan pencere) için derlendi. Tam ekran TOD 1920x1080 için `tools/build_trt_pipeline.py --h 1080 --w 1920` gerekli (motor yoksa ayrı parçalara düşer, p95 ~15,8 ms).
- 44,6 FPS anomalisi tekrarlanmadı. Görülürse `--dump-timing` kaydıyla ayırt edilecek.
- Önizleme açıkken tuzak kaydında sıra farkı 0/2 değerleri kalıyor (20/21). Gerçek yakalama düzensizliği mi, saat kuralları mı, ayrılmadı.

## Kaynaklar

- RT4KSR: https://github.com/eduardzamfir/RT4KSR (Apache-2.0), makale "Towards Real-Time 4K Image Super-Resolution" (CVPRW 2023)
- NTIRE 2023 RTSR: https://github.com/eduardzamfir/NTIRE23-RTSR
- EfRLFN: https://github.com/EvgeneyBogatyrev/EfRLFN (MIT), makale https://arxiv.org/abs/2602.11339
- RVSR (AIS 2024): https://github.com/huai-chang/RVSR (Apache-2.0)
- AIS 2024 RTSR anketi: https://arxiv.org/abs/2404.16484
