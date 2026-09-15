# Fikir bankası

Durumlar: **aday** (güçlü, erken denenecek) / **deneysel** (umut var, risk yüksek) / **çılgın** (belki bir gün) / **elendi** (sebebiyle).

| No | Fikir | Neden işe yarayabilir | Durum |
|---|---|---|---|
| F1 | **Ortak hareket haritası.** RIFE'nin hesapladığı akışı büyütmede de kullan. | Aynı hesap iki kez yapılmaz. arXiv 2104.05778 bunu gösteriyor. | aday |
| F2 | **Pahalı büyütme sadece gerçek karelere.** Ara kareler, büyütülmüş komşulardan hareketle kaydırılıp küçük bir ağla düzeltilir. | 25 gerçek kareye ~2.4 kat daha fazla bütçe ayırır (25'ten 60 FPS'e). Kaliteyi en çok bu artırabilir. | aday |
| F3 | **Zamansal biriktirme (videoda DLSS mantığı).** Önceki çıktıyı kaydır, yeni kareyle harmanla. | Kamera kaydıkça alt piksel kaymaları gerçek detay verir. Titremeyi de azaltır. | aday |
| F4 | **TOD bozulmasını taklit eden eğitim.** 720p, 25 FPS, düşük bitrate H.264 ile eğitim çiftleri. | Gerçek sıkıştırma hasarıyla eğitim çok daha iyi sonuç veriyor (StreamSR). | aday |
| F5 | **Futbola özel ince ayar.** Yiğit'in fikri. | Dar alan, küçük modelin gücünü tek işe odaklar. | aday (veri kaynağı yasal olmalı) |
| F6 | **Sahne kesmesi algılama.** Kamera değişince ara kare üretme, biriktirmeyi sıfırla. | Kesmelerde RIFE ve biriktirme saçma kareler üretir. Ucuz ve şart. | aday |
| F7 | **Sabit katmanlar** (skor tabelası, TOD logosu, saat). Algılayıp ara kareden hariç tut, keskin büyütüp üstüne koy. | Yazıların titremesini ve erimesini önler. | deneysel |
| F8 | **Top farkındalıklı ara kare.** Topu bul, yörüngesini fizikle (parabol) tahmin et, ara karelerde doğru yere koy. | RIFE'nin en zayıf olduğu yer küçük ve hızlı nesneler. 25 FPS'te top kareler arasında çok yol alıyor. | deneysel |
| F9 | **Kamera kaymasını ucuz hesapla.** Genel hareketi homografiyle bul, ağ sadece oyuncuların hareketini çözsün. | Futbolda hareketin çoğu kamera kaymasından geliyor. | deneysel |
| F10 | **Bölgeye göre hesap.** Çim ve seyirci ucuz yoldan, oyuncular, top ve çizgiler pahalı yoldan. | Ekranın büyük kısmı çim, orada pahalı model israf. | deneysel |
| F11 | **Saha çizgilerini geometriyle yeniden çiz.** Kamera kalibrasyonuyla çizgileri net çiz. | Çizgilerin düzeni biliniyor. | çılgın |
| F12 | **FP8/INT8 niceleme** (TensorRT). | Ada kartlar FP8 destekliyor. Başka modellerde ~1.4 kat hız bildirilmiş. | aday (ölçülecek) |
| F13 | **Difüzyon modelini öğretmen olarak kullan.** Çevrimdışı çok kaliteli hedef kareler üretip küçük modeli onlarla eğit (distillation). | Gerçek zamanlı model hızlı kalır, öğretmenin kalitesinden faydalanır. | deneysel |
| F14 | **Tarayıcı penceresinde videoyu tam 1280x720 boyutunda yakala.** | Tam ekranda tarayıcı önce bulanık büyütür, sonra biz yakalarız. Bu bilgi kaybıdır. | aday (pratik detay) |
| F15 | **İleriye bakan pencere (lookahead).** 1-2 saniyelik tampon, 25-50 gelecek kare demek. Çift yönlü VSR, hem geçmiş hem gelecek kareleri kullanır. | Çift yönlü yöntemler tek yönlülerden belirgin kaliteli. Gecikme izni bunu mümkün kılıyor. | aday |
| F16 | **Çok kareli ara kare.** 2 kare yerine 4 kare (t-1, t, t+1, t+2) kullanılır. Doğrusal olmayan hareketi (topun eğrisi, hızlanan oyuncu) yakalar. | Literatürde "quadratic video interpolation" fikri var. 25 FPS'te kareler arası süre uzun, doğrusal tahmin yetmez. | aday |
| F17 | **Kare başı süre sınırı yerine ortalama verim.** Tampon sayesinde GPU'nun ortalamada saniyede 60 kare üretmesi yeterli. Toplu (batch) işlem yapılabilir. | Toplu işlem GPU'yu çok daha verimli kullanır. Anlık yavaşlamaları tampon emer. | aday |
| F18 | **Ses senkronu.** Tarayıcının sesini uygulama bazlı yakala (Windows process loopback), görüntüyle aynı zaman damgasıyla geciktirip çal, orijinal sekmeyi sessize al. | Gecikmeli görüntüye ses de aynı gecikmeyle eşlik eder. | aday |
| F19 | **Tampon azalınca kaliteyi otomatik düşür.** Daha küçük model katmanına geçilir, tampon dolunca geri dönülür. | Donma yerine anlık kalite düşüşü olur. Zayıf sistemlerde de işe yarar. | aday |
| F20 | **Model katmanları (XL → L → M → S → XS).** En büyük model öğretmen olur, küçükler ondan öğrenir (distillation). | "Önce en yüksek kalite, sonra zayıf sistemler" hedefinin doğrudan karşılığı. | aday |
| F21 | **En alt katmanı shader'a gömülü mini ağ yapmak.** Anime4K ve FSRCNNX gibi GLSL/HLSL. | TensorRT ya da CUDA gerekmez, her GPU'da, hatta iGPU'da çalışır. | deneysel |
| F22 | **Parlaklık (Y) ve renk (UV) ayrı yol.** Pahalı ağ sadece Y kanalına, renk kanalları Y rehberliğinde ucuz büyütülür. | Göz detayı parlaklıkta görür. TOD H.264 4:2:0, renk zaten 640x360. Hesap ~%30-60 düşebilir (tahmin). | aday |
| F23 | **25 FPS zaman çizgisini yakalamadan geri kurmak.** Chrome 25 FPS videoyu 60/144 Hz ekrana 2-3 (ya da 5-6) vsync tutarak basar. Yakalanan tekrarlar atılır, yeni karelere ideal 40 ms ızgarasında zaman damgası verilir (PLL benzeri). | Yanlış zaman damgası ara karelerde titreme (judder) yapar. Ucuz ve şart. | aday |
| F24 | **En alt katman (T0) klasik hareket vektörü.** iGPU'da compute shader ile blok eşleme ara kare + shader SR (F21). AI yok. | SVP benzeri yöntemler zayıf donanımda canlı çalışıyor. "Her makinede açılır" tabanı. | deneysel |
