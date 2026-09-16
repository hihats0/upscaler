# upscaler

Bir video penceresini (tarayıcı, oynatıcı) gerçek zamanlı olarak 4K 60 FPS'e çıkaran açık kaynak hat.
Pencere Windows'un normal ekran yakalama API'siyle (Windows.Graphics.Capture) alınır. Kareler ve ses sadece
bellekte işlenir, diske yazılmaz. DRM atlatmaz; yakalanamayan (siyah gelen) içerik işlenmez.

- Ara kare: RIFE 4.25 akışı (TensorRT) + kaynak çözünürlükte kaydırma
- Büyütme: RT4KSR x2 (Apache-2.0), futbol + yayın sıkıştırmasıyla ince ayarlı (CC BY klipler), TensorRT kaynaşık motor
- Sunum: CUDA-OpenGL interop, kare CPU'ya inmez
- Ses: WASAPI process loopback, görüntü gecikmesine kilitli çalma

## Maç nasıl izlenir

1. Chrome'da yayını aç, videoyu tam ekran yap (F11 ya da oynatıcının tam ekran düğmesi).
2. Masaüstündeki **Maç izle (upscaler)** dosyasına çift tıkla (ya da `.venv\Scripts\python.exe -m upscaler watch`).
3. Birkaç saniye içinde 4K60 görüntü seçilen ekranda tam ekran açılır; ses 1,5 sn gecikmeyle görüntüyle birlikte gelir.
4. Kısayollar: **Esc** çıkış, **S** yarı SR / yarı bicubic kıyas, **I** bilgi katmanı (FPS, gecikme, geç tik, VRAM).
5. Çift ses duyarsan: Windows ses ayarlarında Chrome'u kullanılmayan bir çıkışa yönlendir (ayrıntı: `notes/soru-kuyrugu.md`). Kayıtlar `runs/` altında.

## Geliştirme

```
.venv/Scripts/python.exe -m pytest -q                         # birim testleri
.venv/Scripts/python.exe -m upscaler.present --seconds 10 --verify   # sunum ölçümü
.venv/Scripts/python.exe tools/build_trt_pipeline.py --h 1080 --w 1920  # motorlar
```

Ayrıntılı ölçümler ve kararlar `reports/` ve `notes/` altında.
