# Hat 1.1 canlı iskelet + Hat 1.2 ilk hız ölçümleri

- Tarih: 2026-09-15 (öğleden sonra)
- Durum: **Ölçüm (onaylı veri), iş yarıda.** Kullanım limiti nedeniyle oturum burada durduruldu.
- Kapsam: yazılım ortamı, WGC yakalama, zaman çizgisi (F23), GPU tampon, 60 Hz çıkış, ses yakalama, RIFE hızları.
- Önceki: [2026-09-15-faz0-kesif-tod-yakalama.md](2026-09-15-faz0-kesif-tod-yakalama.md)

## 1. Yazılım ortamı (agent seçti, Yiğit "sen seç" dedi)

| Parça | Seçim |
|---|---|
| Dil / ortam | Python 3.13, `uv venv --system-site-packages .venv` |
| GPU | PyTorch 2.11 + CUDA 12.6 (sistemde kurulu) |
| Yakalama | windows-capture 2.0.1 (WGC, MIT) |
| Ses | proc-tap 1.1.1 (WASAPI process loopback, MIT), pycaw (oturum kontrolü) |
| Hızlandırma | tensorrt-cu12 11.3.0.99 (NVIDIA lisansı, yerel kullanım; repoya girmez). venv 281 MB → 3,6 GB |
| Test | pytest. `-m live` canlı testler ayrı |

## 2. Kod (repo)

| Dosya | Ne yapar |
|---|---|
| `upscaler/capture.py` | WGC kaynağı, 1 ms MinUpdateInterval, aynı kare ayıklama |
| `upscaler/timeline.py` | SourceClock: faz tutarlılığıyla periyot, ideal zaman damgası, tekrar/geç kare kuralları. `pick_frames` |
| `upscaler/ring.py` | GPU halka tampon, slot kilidi, kilitlenince yeniden damgalama |
| `upscaler/process.py` | baseline (harman+bicubic), rife, rife-flow işlemcileri |
| `upscaler/models/rife.py`, `rife_flow.py` | RIFE 4.25 sarmalayıcı, CUDA graph, akış+maske (F1/F2) |
| `upscaler/live.py` | 60 Hz çıkış döngüsü, sabit gecikme, istatistik, zaman kaydı |
| `upscaler/testpattern.py`, `tools/test_pattern_window.py` | Barkodlu 50 FPS test deseni |
| `tools/probe_capture.py`, `probe_audio.py`, `bench_rife.py`, `bench_processors.py` | Ölçüm araçları |
| `third_party/vsrife/` | vs-rife IFNet 4.25 + warplayer (MIT, LICENSE içinde) |
| `weights/rife/` | flownet_v4.25(.lite).pkl (gitignore) |
| `tests/` | 32 birim + 1 canlı test, `tests/data/*.csv` (sadece zaman sayıları) |

Çalıştırma:
```
.venv/Scripts/python.exe -m pytest -q            # birim
.venv/Scripts/python.exe -m pytest -m live -q -s # canlı desen testi
.venv/Scripts/python.exe -m upscaler.live --title "TOD - Google Chrome" --foreground --delay 1.5 --seconds 20 --no-preview --proc baseline
.venv/Scripts/python.exe tools/bench_processors.py
```

## 3. Ölçümler

### Yakalama
- Chrome üstü kapalıyken WGC 5 sn'de 1 kare: **TOD penceresi görünür olmalı.**
- **WGC MinUpdateInterval varsayılanı 144 Hz'de yakalamayı 48 Hz'e kısıyor** (50 FPS'te her 25 karede 1 kayıp). 1 ms ile desen 50,03 FPS, 0 kayıp. TOD 50,09 FPS.

### Canlı hat v0 (baseline)
- Desen (1280x720): giriş 49,9, çıkış 60,0 FPS, 0 geç tik, işlem p50 4,5 ms, VRAM 0,54 GB. Çıkış zaman hatası iyi koşularda p99 0,15-0,75 ms.
- TOD penceresi (1920x1020, 20 sn): giriş 50, çıkış 60,0 FPS, 0 geç tik, işlem p50 4,9 / p95 7,5 ms, VRAM 1,2 GB. O koşuda saat yanlışlıkla 12,27 ms'e kilitlendi (pencere içi fazladan güncellemeler), sonra kurallar düzeltildi, tekrar oynatma testi TOD kaydında 50 FPS'e kilitleniyor.

### Ses
- ProcTap Chrome sesini yakalıyor: 48 kHz, 10 ms parça, geliş p90 11,3 ms.
- Karıştırıcıda sessize alınan Chrome'un yakalaması da sessiz (-180 dBFS). Oturum sesi 0,1 → ~-15 dB. **Yakalama oturum sesinden sonra.** Orijinal sesi kapatıp gecikmeli sesi çalmak için Chrome'u kullanılmayan bir çıkışa yönlendirmek gerekiyor (Windows ayarı, Yiğit).

### RIFE (1080p, FP16, PyTorch)
| İşlemci | Ara kare | 50→60 bütçe (ms/sn) |
|---|---|---|
| baseline | 4,3 ms | 248 |
| rife-flow 4.25 (akış 540p, warp 1080p) | **18,0 ms** | **934 (sığmıyor, eşik 850)** |
| rife-flow lite | 19,2 ms | 996 |
| rife 4.25 tam | 49,0 ms | 2483 |
- Tam RIFE profili: kopya/tip dönüşümü ~%35, konv. %34, cat %15, ters konv. %14, grid_sample %12. GPU %99, 85 W, kısıtlama yok. CUDA graph sadece %5 kazandırdı.

## 4. Açık hata (sıradaki ilk iş)

**Canlı desen testi kararsız** (3 koşu: kaldı / geçti / kaldı; p99 0,75 ile 7,9 ms arası). Kök neden `tests/data/live_pattern_fail1.csv` tekrar oynatmasıyla bulundu:
- 50 FPS kaynak 144 Hz ekranda 3-3-3-...-2 vsync düzeniyle basılıyor (144/50 = 2,88). Kareler 20,83 ms aralıkla gelip her karede ~0,04 yuva gecikme biriktiriyor, sonra bir kare **6,9 ms** sonra gelip yetişiyor.
- Örnek: barkod 413 yuva 413,15'te, 414 ise 413,50'de (aralık 6,9 ms). Mevcut kural 414'ü "tekrar" sayıp 413'ün yerine yazıyor → o girdi ~20 ms yanlış zamanda.
- `fix_prev` kuralı tetiklenmiyor çünkü 414 ne 413 yuvasına (0,35 P) ne kendi yuvasına yakın, tam ortada.
- Öneri: "tekrar" kararı için izgara tahmini yerine vsync düzeni hesaba katılmalı; yuva ortasına düşen, önceki kare geç birikmişken gelen kare yeni kare sayılmalı. Sentetik test `test_late_frame_then_on_time_frame_are_both_real` de aynı durumu kapsıyor (şu an kalıyor). Yeni kural `live_pattern_fail1.csv` ve `pattern_144hz.csv`, `tod_window_144hz.csv` tekrar oynatmalarında denenmeli.
- Tam ekran TOD'da (pencere arayüzü yok) fazladan güncelleme az olacağından tekrar kuralı gevşetilebilir.

## 4b. Güncelleme (aynı gün, akşam)

- **Bölüm 4'teki hata çözüldü:** `GpuFrameRing` "tekrar" kararını bir kare erteliyor (`_tentative`). Sonraki kare sıra atlarsa bekletilen kare boşluğa yerleşir. `tests/test_ring_replay.py`: `live_pattern_fail1.csv` max 0,59 ms, desen 0,40 ms. Canlı desen p99 0,29 / 0,17 ms.
- **TensorRT:** `tools/build_trt_flow.py` (ONNX opset 17, strongly typed FP16). 960x576: 11,2 → 4,6 ms; 960x512: 10,7 → 4,1 ms. Gerçekçi girdide çıktı farkı ≤0,004/255 (`tests/test_trt_flow.py`).
- **İşlemci bütçesi (1080p→4K):** rife-flow-trt ara kare 11,9 ms → 50→60 bütçe 629 ms/sn (sığar).
- **TOD canlı, `--proc rife-flow-trt`, 20 sn:** çıkış 59,95 FPS, 1 geç tik, işlem p50 13,6 / p95 14,9 / max 18,2 ms, tampon 1,48 sn, VRAM 1,35 GB, 11 tekrar güncellemesi (pencere arayüzü).
- Not: çıkış tikleri kaynak ızgarasına hizalı olmadığı için neredeyse her tik ara kare üretiyor (bütçe hesabı 10 gerçek kare varsaymıştı). Çözüm: 1/300 sn ızgarası.
- **Izgara hizalama uygulandı** (`live.py` `_snap_grid`, varsayılan açık) + saat kilitlenmeden çıkış yok. Desen: çıkış zaman hatası p99 0,46 / max 0,53 ms, gerçek kare tik 0,165, 0 geç tik. TOD rife-flow-trt 20 sn: çıkış 60,0, 0 geç tik, işlem p50 6,5 / p95 14,4 / max 17,6 ms, VRAM 1,35 GB. **Anomali:** giriş 44,6 FPS, gerçek kare tik 0,75. İncelenmedi (limit).

## 5. Sıradakiler

1. Zaman çizgisi hatası (yukarıda), testler yeşil.
2. rife-flow akış ağını TensorRT motoruna çevir (ONNX, 544x960 sabit, FP16), bütçeyi 850 altına indir. `trt.Builder.platform_has_fast_fp16` TRT 11'de yok (önemsiz).
3. Render-ahead + batch (F17): çıkış karelerini tampondan önceden üretip kuyruğa koymak.
4. Ses: gecikmeli çalma (sounddevice), Chrome çıkış yönlendirmesi kararı.
5. TOD tam ekran (1920x1080, 1:1) ile canlı ölçüm.

## Kaynaklar

- windows-capture: https://github.com/NiiightmareXD/windows-capture (MIT)
- ProcTap: https://github.com/m96-chan/ProcTap (MIT)
- vs-rife: https://github.com/HolyWu/vs-rife (MIT), ağırlıklar GitHub release `model`
- Practical-RIFE: https://github.com/hzwer/Practical-RIFE (MIT)
- EfRLFN: https://github.com/EvgeneyBogatyrev/EfRLFN (MIT, x2/x4 Google Drive)
- Microsoft process loopback örneği: https://learn.microsoft.com/en-us/samples/microsoft/windows-classic-samples/applicationloopbackaudio-sample/
