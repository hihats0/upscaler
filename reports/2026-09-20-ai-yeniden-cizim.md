# AI yeniden çizim ("S24 Ultra gibi")

- **Tarih:** 2026-09-19 akşam (goal adı 2026-09-20)
- **Durum:** araştırma (sürüyor)
- **Kapsam:** Canlı TOD 1080p karesini AI ile keskin (çim dokusu), titremesiz yeniden çizmek, TV modunda 1080p 50 FPS. A: hazır model taraması, B: keskin hedefli veri, C: ince ayar, D: TV moduna takma. Önceki: [TV modu ve onarım v0](2026-09-19-tv-modu-ve-onarim-v0.md), [F26](2026-09-16-f26-tod-yumusaklik.md).

## A. Hazır model taraması

### Lisanslar

| Model | Lisans | Durum |
|---|---|---|
| Real-ESRGAN (realesr-general-x4v3, -wdn-, realesr-animevideov3, RealESRGAN_x2plus) | BSD-3-Clause (repo LICENSE indirildi, `third_party/realesrgan/LICENSE`) | canlıda kullanılabilir, öğretmen olabilir |
| SwinIR-M real x2 GAN | Apache-2.0 (`third_party/swinir/LICENSE`) | kullanılabilir ama çok yavaş |
| RealBasicVSR | Apache-2.0 (mmagic) | ölçülmedi: mmcv kurulumu ve 4x modelin 1080p girdi maliyeti; yeniden çizim yolu zaten elendi |
| SeedVR2 3B/7B | Apache-2.0 | ölçülmedi: resmi test ~18 GB VRAM, 8 GB'ta topluluk FP8/GGUF ile dakikalar/kare; 16 GB RAM |
| SwiftVR (tek adımlı difüzyon, 2026) | makale CC BY 4.0, kod/ağırlık lisansı doğrulanmadı | RTX 5090'da 1080p 26 FPS: 4070 Laptop'ta canlı olamaz |
| FlashVSR (tek adımlı difüzyon) | açık | A100'de 768x1408 ~17 FPS: canlı olamaz |

Difüzyon öğretmeni gerekmedi: hedef zaten gerçek (4K CC karenin keskin 1080p hali). Öğretmen sadece gerçek hedef olmayan yerde işe yarardı.

### Kalite (simüle TOD klibi, `tools/ai_eval.py`, `runs/ai_eval/tarama1.json`)

Girdi: 3 doğrulama klibi (5bgF-5I2P_M 20. ve 200. sn, O3gD6n0zoik 20. sn; eğitimde yok), TOD simülasyonu (ön küçültme 0,7 + unsharp 0,4 + 4,8 Mbps H.264, canlı TOD spk'sine yakın). Hedef: aynı anın 4K karesinden **keskin 1080p** (lanczos, yumuşatma yok). 10 karede bir, kadrajın ortası. Hedef spk_50_75 **0,379**, spk_75_100 0,082.

| Aday | Yol | PSNR-Y | LPIPS ↓ | spk_50_75 | spk_75_100 | Titreme (çıkış/hedef) |
|---|---|---|---|---|---|---|
| ham (TOD sim) | | **34,46** | 0,111 | 0,161 | 0,033 | 0,98 |
| bicubic-540 | 540p'ye küçült + bicubic | 32,29 | 0,211 | 0,055 | 0,006 | 0,89 |
| esr-gen-270 | 270p + general-x4v3 | 26,66 | 0,269 | 0,131 | 0,023 | 0,86 |
| esr-wdn-270 | 270p + general-wdn | 27,69 | 0,198 | 0,206 | 0,052 | 1,81 |
| esr-anime-270 | 270p + animevideov3 | 27,62 | 0,329 | 0,063 | 0,031 | 0,98 |
| esr-gen-540d | 540p + general x4 -> 4K -> 1080p | 29,60 | 0,171 | 0,246 | 0,068 | 0,92 |
| esr-anime-540d | 540p + anime x4 -> 4K -> 1080p | 30,74 | 0,196 | 0,154 | 0,031 | 1,08 |
| **esrx2-540** ("yeniden çizim") | 540p + RealESRGAN_x2plus | 30,43 | 0,125 | 0,184 | 0,053 | **1,54** |
| **esrx2-1080d** | 1080p + x2plus -> 4K -> 1080p | 32,83 | **0,100** | **0,309** | **0,096** | 1,65 |

**Sonuç:**
- "Küçült, GAN ile yeniden çiz" yolu elendi: 540p'ye inmek girdideki gerçek detayı atıyor (bicubic-540 bile 2,2 dB kaybediyor), GAN bunu uydurmayla dolduruyor. PSNR 4 dB düşüyor, LPIPS hamdan kötü, titreme 1,54.
- Algısal olarak en iyi hazır sonuç RealESRGAN_x2plus'ın 1080p'yi 4K'ya çizip geri küçültmesi (LPIPS 0,100, spk 0,31): gözle "keskin" ama titreme 1,65 ve **2,1 sn/kare** (PyTorch).
- Hazır modellerin hiçbiri hem canlıya sığmıyor hem hamdan iyi değil. Karar: kendi küçük ağımız, keskin hedefle (L1 + algısal + GAN + zamansal) eğitilecek.

### Hız (TensorRT FP16, 1080p çıkış, sessiz sistem, `runs/ai_bench.json`)

| Ağ | Parametre | GPU p50 / p95 (ms) |
|---|---|---|
| AiNet 32x6, 1 kare | 63 bin | 5,0 / 5,1 |
| AiNet 48x6, 1 kare | 136 bin | 8,3 / 8,5 |
| AiNet 48x8, 1 kare | 177 bin | 10,4 / 10,6 |
| AiNet 64x6, 1 kare | 236 bin | 10,9 / 11,1 |
| AiNet 64x8, 1 kare | 310 bin | 13,7 / 13,9 |
| AiNet 32x6, 3 kare | 70 bin | 6,4 / 6,5 |
| AiNet 48x6, 3 kare | 146 bin | 9,6 / 9,7 |
| **AiNet 48x8, 3 kare** | **188 bin** | **11,8 / 11,9** |
| AiNet 64x6, 3 kare | 250 bin | 12,2 / 12,3 |
| esr-anime-270 | 621 bin | 4,7 / 4,7 |
| esr-gen-270 | 1,2 M | 8,5 / 9,4 |
| esrx2-540 | 16,7 M | 170 / 171 |

Bütçe: 50 FPS = 20 ms/kare; hattın geri kalanı ~1 ms, ısınma kısıtlamasına pay için ağ ≤ ~12 ms.

**AiNet** (`upscaler/models/ai.py`, kendi kodumuz): 1080p kare 2x2 katlanır (12 kanal, 540p, bilgi kaybı yok), gövde 540p'de conv + PReLU (SRVGGNetCompact tarzı), çıkış 2x2 açılıp girdiye eklenir. 3 kare modunda t-1, t, t+1 kanal olarak girer.

## B. Veri

`tools/make_pairs.py --target 1080sharp --temporal`: girdi tod_sim zinciri (rastgele ön küçültme 0,5-1,0 + unsharp 0-0,8 + 4,8 Mbps H.264 50p), hedef aynı anın 4K karesinden lanczos 1080p (yumuşatma yok). Sadece `runs/gt_sharp.json`'da 4K spk ≥ 0,04 olan segmentler. Yama 128x128, girdi kareleri t-1..t+2, hedef t ve t+1.

| Küme | Klipler | Örnek | Boyut |
|---|---|---|---|
| ai_train_a | 9kP6NZmq4NE, l3bLab5HPR0, Xd2Vsa9-cVg, JbQxmnn2MtM, VVYPUcKhPiQ | ~20 bin | 3,1 GB |
| ai_train_b | D7tVERw5XFo, QdLKQ3SnHmc, q8DsjL7iWXA, 63pDvVidZJA, u5w_du22wSQ | ~9 bin | 1,4 GB |
| ai_val | 5bgF-5I2P_M, O3gD6n0zoik (eğitimde yok), sabit 0,7 / 0,4 | 932 | 0,13 GB |

Doğrulama yamalarında girdinin hf'si (≥ 0,5 Nyquist enerjisi) hedefin 0,49'u, LPIPS 0,097.

## C. İnce ayar

`tools/train_ai.py`: L1 + LPIPS-VGG + GAN (PatchDisc, hinge) + zamansal (ardışık kare farkı hedefinkine L1), EMA 0,999, laptop, 80/70 °C duraklama, watch açılınca 5 sn içinde durur.

| Koşu | Ayar | Adım | Yama doğrulama: PSNR / LPIPS / hf / titreme |
|---|---|---|---|
| girdi (ham) | | | 34,66 / 0,0966 / 0,493 / 1,09 |
| ai_v1 | 48x8x3, 5 bin GAN'sız + GAN 0,05 | 12 bin | 34,49 / **0,0722** / **0,667** / 1,36 |
| ai_v2 | v1'den, zamansal 5 | 3 bin (durduruldu) | 34,60 / 0,0757 / 0,619 / 1,33 (detay düştü, titreme az düştü) |
| ai_v3 | v1'den, GAN 0,15, LPIPS 2, L1 0,5 | sürüyor | |

**Tam kare (ai_eval, 3 doğrulama klibi):**

| Aday | PSNR | LPIPS | spk_50_75 (hedef 0,37-0,38) | titreme |
|---|---|---|---|---|
| ham | 34,46 | 0,110 | 0,162 | 0,90-0,98 |
| ai_v1 (12 bin) | 34,40 | 0,084 | 0,241 | **1,235** |
| ai_v1 + harman 0,6 | 34,40 | 0,085 | 0,241 | 0,961 |
| ai_v1 + harman 0,85 | 34,40 | 0,085 | 0,241 | 0,799 |
| **ai_v1 + güç 1,8 + harman 0,6** | 33,79 | **0,079** | **0,331** | **1,094** |
| ai_v1 + güç 2,5 + harman 0,6 | 32,93 | 0,092 | 0,433 | 1,224 |

- **Durgun bölge harmanı** (`StaticBlend`): girdinin t ile t-1 farkı 5x5 ortalamada < 2/255 olan pikselde çıkış önceki çıkışla %60 karışır. Titremeyi ~%22 düşürdü, kalite aynı. Zamansal kaybı büyütmek (v2) detayı da düşürdü, harman daha iyi.
- **Güç** (AI'nın eklediği farkın çarpanı): 1,8'de spk hedefin %11 altında, LPIPS en iyi, titreme sınır içinde. PSNR -0,7 dB (keskin doku PSNR'ı düşürür, beklenen).

## D. TV modu

`watch --tv --ai <ad> [--ai-guc 1.8] [--ai-renk 1.2] [--ai-kontrast 1.05] [--ai-blend 0.6] [--split] [--probe-sharp 2]`. `AiProcessor`: son 3 yeni kare tamponda, çıkış 1 kare (20 ms) geriden. Güç, renk (BT.709 lumaya göre doygunluk, orta griye göre kontrast) ve harman TensorRT motorunun içinde (`build_trt_ai.py --blend --gain --sat --con`): 48x8x3 hepsi dahil **GPU p50 9,45 ms** (harman PyTorch'tayken canlıda işlem p50 ~19 ms ve 47-50 FPS'ti).

- TOD ilk canlı (18:00, harman PyTorch'ta, eğitim ilk 30 sn GPU'yu paylaştı): 47-50 FPS, işlem p50 ~19 ms, GPU 84-86 °C. Yetmez.
- TOD agresif (18:06, motor içinde): girişte 50 FPS, işlem p95 13,9 ms, geç tik 0, GPU 70 °C. Ama çıkış 142 FPS sayıldı: TV penceresi odak dışıyken swap vsync'i beklemedi, döngü serbest döndü. Düzeltme: lock modunda tik aralığı çıkış periyoduna bağlandı (`vsync_bloklamiyor` olayı).
- TV: Windows 1920x1080 @ 50 Hz, TV'nin en yüksek modu 1080p 60 (EnumDisplaySettings).
- Yiğit'in göz testi (ai_v1, güç 1): "çok fark yok". Güç 1,8 + renk sonrası geri bildirim bekleniyor.

## Açık sorular

- RealBasicVSR ve difüzyon modelleri ölçülmedi (sebepler yukarıda).

## Kaynaklar

- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN (BSD-3)
- SwinIR: https://github.com/JingyunLiang/SwinIR (Apache-2.0)
- SeedVR2: https://huggingface.co/ByteDance-Seed/SeedVR2-3B (Apache-2.0), 8 GB notu: https://seedvr2.net/blog/tutorials/seedvr2-comfyui-low-vram-guide-2026
- SwiftVR: https://arxiv.org/abs/2606.09516
- FlashVSR / tek adımlı difüzyon VSR: https://arxiv.org/abs/2601.20308
- LPIPS: https://github.com/richzhang/PerceptualSimilarity (BSD-2)
