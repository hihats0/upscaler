# Öğrenilenler

Doğrulanmış ya da kaynağı olan bilgiler. Tahminler ayrıca "tahmin" diye işaretlenir. Ayrıntı ve kaynaklar `reports/` altında.

## Yiğit'in gereksinimleri

- 2026-09-15: **HER ŞEY CANLI.** Yiğit sert şekilde vurguladı. Çevrimdışı ürün fazı önerilmez, ölçüm de canlı hattan geçer.
- 2026-09-15: İzleme harici 4K ekranda olacak. Hedef **4K 60 FPS**.
- 2026-09-15: **1-2 saniye gecikme kabul.** Ses görüntüyle senkron gitmeli. Bu, gelecek kareleri görebilen (lookahead) modellerin önünü açıyor.
- 2026-09-15: Önce **en yüksek kalite** görülsün, sonra optimize edilip **zayıf sistemlere** kadar indirilsin.
- 2026-09-15: Proje **GitHub'da açık kaynak** olarak paylaşılacak. Model, veri ve ağırlık lisansları önemli hale geldi.

## Kaynak (TOD)

- ~~2026-09-15: TOD tarayıcıda 720p 25 FPS (forumlar).~~ **Yanlış çıktı.** Ölçüm aşağıda.
- 2026-09-15 (**ölçüldü**, beIN Sports 1, Chrome): **1920x1080, gerçek 50p**, H.264 High L4.2, 4:2:0, BT.709 LIMITED. Ses AAC stereo 48 kHz. ABR: açılışta 640x360, sonra 1080p. Rapor: `reports/2026-09-15-faz0-kesif-tod-yakalama.md`.
- 2026-09-15 (ölçüldü): Bitrate ~4,8 Mbps (Wi-Fi toplamı, üst sınır). Piksel başına ~0,046 bit, sıkı sıkıştırma. Modelin sıkıştırma hasarını da temizlemesi gerekiyor.
- 2026-09-15 (ölçüldü): Widevine `use_hw_secure_codecs: false` (L3), `DecryptingVideoDecoder` (CPU'da çözme). **Chrome penceresi yakalanabiliyor, siyah değil** (BitBlt ve PrintWindow). WGC doğrudan denenmedi (tahmin: çalışır).
- 2026-09-15 (ölçüldü): Chrome üstü kapalıyken yakalama yanıltıcı olur. TOD penceresi görünür olmalı.
- 2026-09-15 (ölçüldü, `tools/probe_capture.py`): **WGC TOD'u yakalıyor**: 1920x1020, 49,8 benzersiz kare/sn. **Chrome'un üstü kapalıyken WGC 5 sn'de sadece 1 kare aldı**: Chrome çizmeyi bırakıyor. Canlı hatta TOD penceresi görünür olmalı. WGC `timespan` (100 ns) ile `perf_counter` aynı saat, fark ~3 ms.
- 2026-09-15 (ölçüldü, test deseni barkodu): **WGC `MinUpdateInterval` varsayılanı yakalamayı 144 Hz'de 48 Hz'e kısıyor** (50 FPS kaynakta her 25 karede 1 kayıp, saat yanlışlıkla 48'e kilitleniyor). **1 ms verince: 50,03 FPS, 0 kayıp, saat sıra hatası 0**, ideal aralık hatası ≤ 0,27 ms. TOD'da da 50,09 FPS.
- 2026-09-15 (ölçüldü): GPU'da 1080p→4K bicubic FP16 ~335 kare/sn (2,98 ms). 1080p BGRA pinned yükleme 0,65 ms.
- 2026-09-15 (ölçüldü, `tools/probe_audio.py`): **Chrome sesi ProcTap (WASAPI process loopback, süreç ağacı dahil) ile yakalanıyor**: 48 kHz stereo float32, 480 örnek/10 ms parça, geliş p90 11,3 ms. Chrome sesi ayrı bir ses servisi sürecinden çalıyor (ağaç dahil etmek şart).
- 2026-09-15 (ölçüldü): **Karıştırıcıda Chrome sessize alınınca yakalama da sessiz (-180 dBFS).** Chrome sesi 0,1 → yakalama ~-15 dB, 0,01 → ~-25 dB. Yakalama oturum sesinden SONRA. "Orijinali kıs, gecikmeli sesi biz çal" temiz değil (yankı). Temiz yol: Chrome'u kullanılmayan bir ses çıkışına yönlendirmek (Windows ayarı, Yiğit yapar) ya da sanal ses kablosu.
- 2026-09-15 (ölçüldü): **RIFE 4.25 PyTorch FP16 1080p: ~21 kare/sn** (47 ms). scale=0,5 ve lite de 20-24. Darboğaz hesap değil çekirdek çağrı yükü (tahmin). Hedef: saniyede 50 ara kare. CUDA graphs / TensorRT şart.
- 2026-09-15 (ölçüldü): **rife-flow** (RIFE akış+maske 540p, gerçek kareleri 1080p'de warp): ara kare 18 ms, tam RIFE 49 ms. 50→60 bütçe 934 ms/sn, henüz sığmıyor. CUDA graph %5.
- 2026-09-15 (ölçüldü, tekrar oynatma): 50 FPS kaynak 144 Hz'de 3-3-...-2 vsync düzeniyle geliyor; biriken gecikme sonrası bir kare 6,9 ms aralıkla "yetişiyor". Bu kare yuva ortasına düşüyor ve saat onu tekrar sanıyor. Canlı test kararsızlığının kök nedeni.
- 2026-09-15 (ölçüldü): **rife-flow akış ağı TensorRT FP16 (960x576): 11,2 ms → 4,6 ms (x2,44).** ONNX 15,8 MB, motor 125 MB, kurulum 34 sn (`tools/build_trt_flow.py`, strongly typed, `onnx` paketi gerekli). Rastgele girdide akış farkı ort. 0,17 px, maske logit farkı 1,85; **gerçekçi girdide (desen, kayan kare, doku) akış farkı ≤0,003 px, sigmoid maske ≤0,004, 1080p çıktı farkı ort. ≤0,004/255: doğrulandı.**
- 2026-09-15: TensorRT 11.3 venv'e kuruldu (+3,3 GB). TRT 11'de `platform_has_fast_fp16` yok.
- 2026-09-15 (ölçüldü): Canlı hat v0 (bicubic + doğrusal harman) TOD'da: 50 FPS giriş, 60,0 FPS çıkış, 0 geç tik, işlem p50 4,9 ms, VRAM 1,2 GB. Pencere yakalamada video dışı güncellemeler (kontroller, imleç) fazladan "kare" üretiyor.
- Süper Lig'de 4K yok (forumlar). Edge + PlayReady donanım DRM'i yakalamayı siyah yapar.

## Donanım (komutla okundu, 2026-09-15)

- RTX 4070 Laptop GPU, 8188 MiB, güç sınırı en fazla 140 W. En güçlü 4070 Mobile sürümü.
- Sürücü 610.88.
- Dahili ekran 1920x1080 144 Hz, Intel iGPU üzerinden bağlı (Optimus). 4K çıktı bu ekranda görünmez.
- NVIDIA tarafında görünen 1920x1080 60 Hz ekran gerçek monitör değil (`display_active: Disabled`), büyük ihtimalle Parsec sanal ekranı (2026-09-15).
- 2026-09-15: Harici ekran bağlı değil. Chrome Intel iGPU'da çalışıyor (NVIDIA'da işlem yok). Windows ölçekleme %125.

- 2026-09-15: Laptop ASUS TUF Gaming F15 FX507ZI4. i7-12700H, 16 GB RAM, C: 111 GB boş. HDMI 2.1 FRL, Thunderbolt 4 (DP), MUX anahtarı ve Advanced Optimus var. Python 3.13.13, CUDA araç seti 12.1.
- Çıkarım: Harici 4K60 sorun değil. **Disk ve RAM asıl kısıt.** 4K PNG kareleri dakikada ~40-50 GB tutar (tahmin), diske dökülmez.

## Lisanslar (2026-09-15)

- Apache-2.0: SeedVR2, FlashVSR, EMA-VFI. MIT: EfRLFN, vs-rife.
- Ticari olmayan / sadece araştırma: RVRT (CC-BY-NC), BiM-VFI (araştırma ve eğitim), GIMM-VFI (sarmalayıcı ticari olmayan, orijinal doğrulanmadı).
- Pexels, kullanım koşullarında ML için veri toplamayı yasaklıyor.
- Inter4K, X4K1000FPS, StreamSR, BasicVSR++ lisansları bulunamadı.

## Hukuk ve veri (2026-09-15, avukat görüşü değil)

- YouTube kullanım koşulları: YouTube'un kendi indirme düğmesi yoksa ya da hak sahibinin izni yoksa içerik indirilemez. İstisnalar: Creative Commons lisanslı videolar, kamu malı içerik, kendi yüklediğin videolar.
- YouTube aramasında "Creative Commons" filtresi var. CC BY lisansı, atıf yapmak şartıyla yeniden kullanıma izin veriyor.
- Türkiye'de ABD'deki gibi açık uçlu bir "adil kullanım" (fair use) kuralı yok. FSEK istisnaları dar ve sınırlı sayıda. Yapay zeka eğitimi için açık bir istisna yok. Yargıtay izinsiz çoğaltmayı hak ihlali sayıyor. Eğitim için lisans ve bedel öngören bir kanun teklifi var.
- Sonuç: TOD'u kaydetmek ve CC olmayan YouTube maç videolarını indirmek olmaz. CC lisanslı videolar kullanılabilir.
- Veri neden gerekli: (1) Ölçüm için "cevap anahtarı" (aynı sahnenin temiz 4K60 hali), (2) aynı klibin tekrar tekrar kullanılabilmesi, (3) tavan modelleri canlı çalışamayacak kadar yavaş ve TOD kaydedilemediği için dosya lazım, (4) kendi modelimizin eğitimi.

## Bulut ve eğitim maliyeti (2026-09-15)

- Vast.ai tipik saatlik: RTX 4090 $0.39, A100 80GB $0.94-1.32, H100 $2.13-2.21. Kesintili kiralama %30-50 ucuz. Depolama aylık GB başına $0.10-0.15, makine dursa da işler.
- SwiftVR: H100'de 4K 14 FPS, RTX 5090'da 1080p 26 FPS. 1440p'de 38 GB bellek. Kod ve ağırlık yayında, kod lisansı doğrulanmadı. En ucuz 4K öğretmen adayı.
- FlashVSR: A100'de 768x1408 için ~17 FPS.
- Tahmin: İnce ayar 4090'da 5-12 saat ($2-5). Sıfırdan küçük SR 50-80 saat ($20-30). Kendi video modelimiz 150-400 saat ($60-160). Deneme-yanılma için 3-5 ile çarp.
- Temiz 4K hedef görüntü varsa öğretmen gereksiz. Öğretmen, 1080p klipleri 4K sahte hedefe çevirmek için lazım.
- **Yiğit'in bütçesi çok kısıtlı (2026-09-15).** Plan 0 TL varsayımıyla kuruldu.

## Bedava GPU (2026-09-15)

- Kaggle: Haftada ~30 saat (talebe göre değişir). P100 16 GB ya da 2x T4 (toplam 32 GB). Oturum en fazla 9 saat. "Save & Run All" ile tarayıcı kapalıyken arka planda çalışır.
- Google Colab bedava: Haftada ~15-30 saat T4 16 GB. Oturum en fazla 12 saat ama garanti yok. 90 dakika etkileşim olmazsa bağlantıyı keser.
- Tahmin: T4 ve P100, 140 W 4070 Laptop'tan yavaş. Değerleri: daha fazla VRAM (16 GB) ve laptop maç açarken eğitimin sürebilmesi.

## Tavan modelleri

- SeedVR2-3B bile en az ~18 GB VRAM istiyor, bizim laptopa sığmaz, bulut gerekir.
- FlashVSR en az 8 GB VRAM ile çalışır, 4K için parça parça işleme şart, 32 GB RAM öneriliyor.

## Modeller ve hız

- Hafif SR modellerinde maliyetin çoğu girdi çözünürlüğünde harcanır. 720p kaynak, 1080p kaynağa göre çok daha ucuz.
- EfRLFN: 720p x2'de RTX 2080'de 271 FPS, SPAN 60 FPS, NVIDIA VSR 52 FPS. Kullanıcı testinde EfRLFN, NVIDIA VSR'a karşı %77 tercih edildi. MIT lisanslı, x2 ve x4 ağırlıkları var.
- RIFE v4.25 TRT FP16, 1080p'de 4060 Ti'de 84 FPS.
- Tahmin: 4070 Laptop 140 W, 2080 Ti ile 4060 Ti arasında bir yerde.
- Gerçek sıkıştırma hasarıyla eğitim, bicubic küçültmeyle eğitimden belirgin şekilde daha iyi (StreamSR çalışması).
- Futbol verisiyle eğitilen SR, genel veriyle eğitilene göre futbolda biraz daha iyi (arXiv 2402.00163).
- Difüzyon tabanlı futbol büyütücüler var (arXiv 2503.11181), ama gerçek zamanlı değiller.
- SR ve ara kareyi ayrı ayrı zincirlemek israf. Hareket haritası ve maskeyi düşük çözünürlükte bir kez hesaplayıp paylaşmak daha verimli (arXiv 2104.05778).
- FP8, Ada kartlarda (4070 dahil) TensorRT ile destekleniyor. SR modellerindeki kazancı ölçülmedi.

## Veri

- SoccerNet: 550 maç, 720p 25 FPS, NDA gerekiyor. 4K hedef görüntü olarak kullanılamaz.
- StreamSR: 5200 YouTube videosu, EfRLFN deposunda. Lisansı kontrol edilecek.
- TOD kaydını veri seti olarak kullanamayız (telif ve kullanım koşulları).

## Çıkarımlar

- Hız çözülebilir bir sorun. Asıl zorluk kalite: 720p, 25 FPS ve düşük bitrate kaynaktan iyi görüntü çıkarmak.

## SR modelleri canlı hatta (2026-09-15 akşamüstü, 4070 Laptop, 1080p -> 4K)

- EfRLFN x2 (MIT): ağ tam 1080p'de 52 kanal. Eager fp16 306 ms, TensorRT 105 ms. **Canlı 4K60'a sığmaz, elendi.** Makalenin 271 FPS'i 360x480 girdiyle (12 kat az piksel).
- RT4KSR x2 (Apache-2.0, NTIRE23 RTSR taban modeli): 2x PixelUnshuffle ile 540p'de 24 kanal. Checkpoint eğitim biçiminde, reparam (tek 3x3) birebir doğru. **TensorRT 1080p: 3,13 ms, 1020p: 2,90 ms.** Doğal görüntüde x2 PSNR 35,62 dB, bicubic 32,67 dB.
- İşlemci ölçümü: rife-flow-trt + RT4KSR ara kare 13,1 ms, gerçek kare 5,2 ms, 50->60 bütçe 707 ms/sn (sığar). Bicubic'li rife-flow-trt 13,0 ms: SR, bicubic kadar ucuz.
- 4K'da warp + harman 14 ms (1080p'de 2,7 ms): ara kareyi 4K'da üretmek pahalı, önce ara kare sonra SR.
- LayerNorm FP16'da eps=1e-6 sıfıra yuvarlanır, düz bölgede (siyah bant) 0/0 olur: norm float32'de hesaplanıyor.
- cuDNN TF32 konvolüsyonu float32 karşılaştırmalarda ~1e-3 fark üretir.
- TOD oynatıcısı bazen "Bir hata oluştu (SA-...)" ile durur; WGC o zaman neredeyse kare vermez (0,06 FPS). Canlı koşudan önce yayının aktığını kontrol et.
- Kaynaşık TensorRT (2026-09-15, 1920x1020): tam kaynaşık motor (BGRA -> akış -> warp -> SR -> 4K uint8) ayrı parçalardan hızlı değil (12,6 / 12,1 ms). Kazanç, çıktı dönüşümünü (RGB->BGR, uint8, siyah bant) SR motoruna katmakta: ara kare 10,2 ms, gerçek kare 3,1 ms. TOD canlı p95 15,8 -> 13,9 ms.
- Önizleme penceresi (pygame, 4K -> 540p + CPU kopyası) işlem süresine ~3-4 ms ekliyor.
- Saat kilidi: 144 Hz ekranda 1 sn ısınma bazen 50 FPS kaynağı 48 Hz'e kilitliyor (3-3-2 vsync düzeni, 48 Hz skoru 0,28, 50 Hz 0,08). 3 sn'lik pencere bütün kayıtlarda 50 veriyor. Isınmayı uzatmak testleri bozdu; çözüm kilitten sonra her ~2 sn'de son 6 sn'lik geçmişle periyodu doğrulamak, arka arkaya 2 uyuşmazlıkta yeniden kilit. 3 sn'lik pencere canlıda 57 denemenin 6'sında yine 48 dedi.
- Kare damgaları farklı ofset tahminleriyle yazılıyor: çıkış ızgarasına hizalamada alpha zaman farkından değil sıra numarasından hesaplanmalı.

## Hat 1.4 maç günü sürümü (2026-09-15 akşam)

- 1920x1080 kaynaşık motorlar derlendi: gerçek kare 2,99 ms, ara kare (karma) 11,13 ms (ayrı parçalar 5,20 / 13,12).
- TOD ABR 720p'ye düşse de **pencere boyutu değişmez** (Chrome videoyu pencereye büyütür). Hattın gördüğü boyut değişimi tam ekran <-> pencere geçişi. Motoru olmayan boyut en yakın motor tuvaline sığdırılıyor (`process.plan_input`, `fit_bgra`).
- GL bağlamı bu laptopta ortam değişkeni olmadan da NVIDIA'da açılıyor (`NVIDIA GeForce RTX 4070 Laptop GPU/PCIe/SSE2`, GL 4.6).
- CUDA-GL interop: `cudaGraphicsSubResourceGetMappedArray(cudaArray_t*, resource, index, mip)` ve `cudaGraphicsResourceGetMappedPointer(void**, size_t*, resource)`. Argüman sırası çoğu örnekten farklı; ters sırada kayıt "başarılı" görünür ama eşleme 400 verir.
- **Chrome örtülme (ölçüldü, WGC benzersiz kare/sn):** açık 54; ekranı birebir kaplayan ve çizen üst pencere altında **35** (araç penceresi de normal pencere de); aynı pencere çizmeyince 50; **pencere 1 px kısa olunca 50,0** (araç ya da normal). Sebep tahmin: DWM tam ekran yolu. Sunucu penceresi 1 px kısa açılıyor.
- Process loopback (ProcTap) Microsoft belgesine göre belirli bir ses uç noktasına bağlı değil: Chrome başka çıkışa yönlendirilse de yakalanır. Bu laptopta ikinci çıkış aygıtı olmadığı için deneyle doğrulanmadı.
- Watch TOD 180 sn (1920x1020 pencere, laptop 144 Hz, timer modu): çıkış 60,0, geç tik 0, işlem p95 12,82 ms, **sunum dahil uçtan uca p95 13,89 ms**, sunum (yükleme 0,9 + çizim + swap) p95 1,72 ms, geç sunum %0,075, nvidia-smi VRAM 2533 MB sabit, torch reserved 1514 MB sabit, GPU 66 -> 81 °C, 72-81 W. İlk kare süreç başlangıcından 7,1 sn.
- İlk denemede kaynak/ses yönetimi (ProcTap başlatma, öne getirme) çıkış döngüsünde 7 sn bloke etti (444 geç tik): yönetim ayrı iş parçacığına alındı.
- ProcTap parça gelmeden `alive()` yanlış "ölü" diyordu: başlangıç payı eklendi.
- A/V 3 dk (ffplay flaş+bip klibi): ffplay'in kendi A/V farkı medyan 46 ms, çıkış 49 ms, hattın eklediği 2,8 ms. Çıkış farkı 3 dk'da 37 -> 59 ms kaydı: incelenmedi (tahmin: ölçüm eşleştirmesi ya da sunum gecikmesi EMA'sı).

## A/V kayması incelemesi (2026-09-16)

- **3 dk'daki 37 -> 59 ms "kayma" hattın değil kaynağın (ffplay).** İkinci koşuda (`runs/av_3dk_b`, ham kayıt `av_ham.json`) girdideki A/V (ffplay'in kendi farkı) t≈110 sn'de 63 -> 42 ms **basamak** yaptı (klip 60 sn'de bir başa sarıyor). Çıkış A/V aynı basamağı izledi. Doğrusal eğim bunu -600 ms/saat "kayma" diye raporluyordu.
- Hattın gecikmeleri sabit: görüntü (ham flaş -> swap) medyan 1509,5 ms, ses (tahmini yakalama -> DAC) 1512,3 ms. Yakalama saati c0 3 dk'da en fazla 1,8 ms oynadı, fs 47999,7-48000,2.
- Olay başına hattın eklediği A/V `(ses çıkış - ses giriş) - (görüntü çıkış - görüntü giriş)`: medyan 3,2 ms, p5-p95 -10..+14 ms (çıkış tik ızgarası 16,7 ms). Asıl ölçüt bu (kaynağın kendi A/V'sinden bağımsız).
- Eski özetteki görüntü/ses gecikmesi -491 ms eşleştirme hatasıydı: olaylar 2 sn periyotlu, gecikme 1,5 sn; "en yakın" olay -0,5 sn'deki. Eşleştirme artık beklenen gecikme etrafında.
- Tahmin: gerçek maçta Chrome'un kendi A/V farkı da ölçülemez (DRM'li içerikte flaş yok); hat bu farkı olduğu gibi taşır (+3 ms).
- tracemalloc (25 kare iz) RSS'i dakikada ~180 MB şişirir: bellek sızıntısında RSS için kullanılmaz, sadece Python nesne farkı için.
- `PresentStats` listeleri sınırsızdı (her karede 4 float, ~0,5 MB/dk): sınırlı deque yapıldı.
- 60 dk A/V (`runs/av_60dk`, test klibi, ölçülü): hattın eklediği A/V medyan 5,4 ms (p5-p95 -8..+16), kayma -0,7 ms/saat; görüntü/ses gecikmesi kayması 0,5 / -0,2 ms/saat. Çıkış 59,994 FPS, geç tik 20/215857 (%0,009), geç sunum %0,23, ses sert atlama 0. RSS 1280 -> 1163 MB (büyüme yok), VRAM smi 2595 MB sabit. GPU ort. 85,5 °C, maks 88 °C, ~86 W; zamanın %9'unda ısıl yavaşlama bayrağı (0x20). Geç tik kümeleri (620-750 sn, 1850-1860 sn) bu bayrağa denk gelmedi (10 sn çözünürlükte).

## Dayanıklılık (2026-09-16)

- Windows %125 ölçekte DPI farkında olmayan araç `SetWindowPos(1920, 1080)` ile pencereyi fiziksel 2400x1350 yapıyor. Sınav aracı artık `winutil.dpi_aware()` çağırıyor.
- **4K ekran riski:** tampon kaynak boyutunda 150 kare tutuyordu; 2400x1350 pencere 2 GB, 3840x2160 (4K ekranda tam ekran Chrome) ~5 GB olurdu. Tampon artık 1920x1080'i aşan kareyi GPU'da küçültüp saklıyor (`GpuFrameRing.MAX_HW`).
- Tampon yeniden ayrılırken işlemdeki seçim eski slotları tutuyor, `empty_cache` bırakamıyordu (her boyut değişiminde +1 GB). Ana döngü, nesil değişen seçimi bıraktıktan sonra önbelleği bir kez boşaltıyor. robust3: VRAM +16 MB, tüm kontroller geçti.
- Boyut değişiminde tampon sıfırlanır, 1,5 sn gecikme dolana kadar son kare tutulur (o 10 sn penceresinde ~51 FPS görünür, işlem yavaşlığı değil). Sunucu yeniden açılması (ekran değişimi) ~1,2 sn'de 75 geç tik yapar (tek seferlik).

## Kilit modu (2026-09-16, `runs/lock48`)

- 144 Hz laptop panelinde `--out-fps 48`: mod lock, interval 3. Çıkış 48,03 FPS, geç tik 0, geç sunum 0, sunum aralığı p1/p50/p99 16,5 / 20,8 / 25,2 ms, sunum gecikmesi p99 4,4 ms. Swap vsync beklediği için sunum süresi p50 7,3 ms. Hattın eklediği A/V 1,5 ms. 4K 60 Hz ekranda aynı yol interval 1 ile çalışacak (denenmedi, ekran yok).

## Canlı ölçüm ve veri (2026-09-16, rapor: reports/2026-09-16-hat13-canli-olcum.md)

- Fabio di Mauro Official (YouTube): kendi çektiği İtalyan amatör maçları, 4K 59,94 FPS, CC BY (sayfada doğrulandı). Arama filtresi CC+4K: `sp=EgQwAXAB`. "CC" etiketli yıldız derlemeleri geçersiz (yükleyen hak sahibi değil).
- YouTube "60 FPS" klipleri 59,94 (60000/1001). tod_sim kareleri sırayla 60 sayar.
- tod_sim eşleşmesi 40/40 (fps=50:round=near kare seçer), 4800 kbps H.264 High 50p doğrulandı.
- Canlı puan: 4K SSIM GPU'da bitişik olmayan kırpıkta 54-65 ms, bitişikte 13 ms; ama GPU'da (ayrı akışta bile) TensorRT işini bekletip işlem p99'u 71 ms yaptı. CPU'da (cv2 2 iş parçacığı) ~350 ms/kare, hatta etkisi yok.
- Çıkış tiklerinin fazı klip numarasına göre rastgele: GT ızgarası için faz `-(klip no - sıra) mod 5`.
- **TOD benzeri girdide hazır RT4KSR, bicubic'ten sadece +0,10 dB (SSIM -0,004).** Temiz girdide +2,9 dB idi. RIFE ara kare doğrusal harmandan +4,6 dB. Gerçek kare 33,7 dB, ara kare 30,4-31,6 dB (fused-sr).
- 1 GT kare kaydırma: 33,8 -> 23,3 dB (ölçüm hizalı).

## İnce ayar hazırlığı (2026-09-16)

- Yama çiftleri: LR 128x128 (1080p ölçek), HR 256x256, düz yamalar atılır (std > 6). 800 yama = ~80-90 MB npz. İki eğitim klibi (16 + 15 segment, 8 sn) ~19 bin yama, 2,1 GB, klip başına ~8 dk (CPU).
- Doğrulama (val1): 5bgF-5I2P_M 1600 + drone 200 yama. Başlangıç: **hazır RT4KSR 34,82 dB, bicubic 35,82 dB** (dokulu yamalarda hazır model bicubic'in gerisinde). 300 adımlık deneme 35,39'a çıktı.
- Eğitim hızı: batch 16, bf16 autocast, ~24 it/sn (4070 Laptop), GPU ~64 °C.
- YouTube indirmesi bu oturumda 20 MB/s'den 2,7 MB/s'ye düştü (tahmin: kısıtlama ya da CPU yükü).
- Fabio di Mauro kanalının yeni videoları futsal vlogu; 2017-2022 Eccellenza/Interprovinciale maçları işe yarıyor. Kanal URL'si: channel/UCjSoXQgnoKUJVUEmu08Ebwg (@fabiodimaurofficial 404).

## İnce ayar sonucu (2026-09-16, rapor: reports/2026-09-16-hat22-ilk-ince-ayar.md)

- ft2 (10 klip, 98 bin yama, 60 bin adım, EMA 0,999, L1, lr 2e-4 kosinüs): doğrulama 34,51 dB (hazır 33,65, bicubic 34,34). EMA'sız ft1 doğrulaması ±0,15 dB dalgalanıyordu.
- Canlı: gerçek kare ft2 - hazır = +0,25 / +0,11 dB, SSIM +0,008; ara kare +0,11 / -0,06 dB. TensorRT 3,10 ms (aynı hız). 60 FPS, geç tik 0.
- Canlı hat CPU yüküne duyarlı: arka planda ffmpeg çift üretimi varken geç tik %0,8-0,9 (boşta 0).
- Eğitim sırasında GPU en fazla 77 °C (canlı hatta 88 °C).
- 2026-09-16 (runs/tod_uzun): Durağan TOD sayfasında Chrome kare göndermeyi kesebiliyor (yayin_dondu 1-1,2 sn, yakalama yeniden başlatınca döndü). Sebebi tahmin: içerik durağan ya da upscaler penceresi Chrome'u örttüğü için kısılma. 1080p ekranda 4K çıktı küçültülerek gösterilir, 4K farkı görünmez.
- 2026-09-16 (Yiğit gözü, TOD): beIN logosu keskin, kamera görüntüsü 1080p gibi değil. Yani kaynağın etkili çözünürlüğü 1080p'nin altında (sebep tahmin: yapım zinciri, deinterlace, 4,8 Mbps sıkıştırma). tod_sim.py ise temiz 4K'yı keskin 1080p'ye indiriyor: eğitim bozulması gerçeğe göre fazla temiz. F26.
- 2026-09-16 F26 ölçüm (`tools/sharpness_probe.py`, sadece sayı): canlı TOD tekrar yayını (Premier League, beIN 3) kadraj ortası kayıp(0,5) 0,71-0,90, spk_50_75 0,13-0,14, spk_75_100 0,036-0,038 (tod_net1 180 sn, tod_net2 300 sn). Bizim eski sim: 2,55 / 0,26 / 0,043. Ayırt eden asıl ölçü spk_50_75 (CC keskin 1080p 0,20, 720p'den büyütülmüş 0,08, 540p 0,03). kayip05 içeriğe çok bağlı.
- 2026-09-16: CC 4K kliplerin bazı bölümleri de yumuşak (63pDvVidZJA 775. sn bozulmasız 1080p'de spk 0,12). GT bölüm bölüm ölçülmeli (`tools/gt_sharpness.py`).
- 2026-09-16 F26: canlı TOD karesi modelden geçirilip 4K çıktı ölçüldü (`sharpness_probe --sr`, 30 sn, 10 örnek): 4K spk_50_75 bicubic 0,035, hazır RT4KSR 0,125, **ft2 0,019** (bicubic'ten bile düz). Keskin 4K CC bölümleri ~0,04-0,06. ft2 sıkıştırmayı temizlerken detayı da siliyor: Yiğit'in "çamurlu" gözlemiyle uyumlu.
- 2026-09-16 F26 eğitim denemeleri (val3: O3g + 5bgF keskin bölümler, sabit TOD benzeri bozulma pre 0,7 unsharp 0,4, 1596 yama). Taban: bicubic 33,85 dB / hf 0,319; hazır 33,23 / 0,426; ft2 33,99 / 0,314. 5 bin adım, rastgele bozulmalı train2 (56 bin yama): L1 33,84/0,309; +genlik 0,5 33,86/0,329; +2 33,87/0,340; +8 33,84/0,347; +20 33,81/0,349. **Regresyon kayıplarıyla hf ~0,35'te doyuyor**; göz testinde çim dokusu hiçbir modelde yok (`tools/compare_patches.py`).
- Karmaşık FFT kaybı (0,1) etkisiz; faz hatasını cezalandırdığı için yine bulanığı seçer. Genlik kaybı doğru yönde.
- val2 tek bölümdü ve rastgele p0,99 düştü: TOD'u temsil etmedi, ft3/ft4 kararları ona göre verilmemeli. Doğrulamada bozulma sabit ve TOD benzeri olmalı.
- GAN (kendi PatchDisc, spektral norm) eğitimi ~5 it/sn, GPU 83 °C.
- 2026-09-16: Bugün GPU ~5,1 saat yüklü (12:25-21:22 arası, runs/ kayıtlarından). GAN eğitimi 95 W, 86-87 °C ve sürücü ısı kısıtlaması (sw_thermal_slowdown) açıldı. Yiğit GPU sağlığını sordu: train_sr duraklama eşiği 88/78'den **80/70 °C**'ye indirildi.
