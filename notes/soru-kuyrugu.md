# Soru kuyruğu

Yiğit'e her mesajda tek soru sorulur. Sıradaki sorular burada bekler. Cevaplanınca "Cevaplananlar"a taşınır.

## Sıradakiler (öncelik sırasıyla)

1. **Ses yönlendirme:** Gecikmeli sesi biz çalacağız, Chrome'un kendi sesi duyulmamalı. Karıştırıcıda sessize almak yakalamayı da susturuyor. Chrome'u Windows'ta kullanılmayan bir çıkışa (ör. bağlı olmayan HDMI / sanal kablo) yönlendirmek uygun mu? (Windows ayarı, Yiğit yapar.)
- ~~Yazılım ortamı~~ Cevaplandı 2026-09-15: Yiğit "sen seç" dedi. Python 3.13 + PyTorch + WGC + ProcTap + TensorRT seçildi.
2. **Öğrencilik:** Öğrenci misin? (GitHub Student Developer Pack gibi programlarda bulut kredisi olabilir, araştırılacak.)
3. **Disk:** Harici SSD var mı ya da alınabilir mi? (C: 111 GB boş)
4. ~~Harici ekran~~ Cevaplandı 2026-09-15: **4K ekran yok.** Hat 1.4 laptop ekranında (1080p 144 Hz) doğrulanacak, 4K tarama ekran gelince.
5. **Zaman:** Haftada bu projeye kabaca kaç saat ayırabilirsin? (Faz sürelerini buna göre konuşuruz.)
6. **Lisans:** Öneri: kod Apache-2.0, ağırlıklar sadece temiz kaynaklardan. Uygun mu?
7. **GitHub:** Repo baştan herkese açık mı olsun, yoksa ilk çalışan sürümde mi açılsın?
8. **İsim:** Projenin GitHub adı ne olsun?

## Yiğit'in yapacağı Windows ayarları (talimat, agent değiştirmez)

**Çift ses (Chrome'un kendi sesi + bizim 1,5 sn gecikmeli sesimiz):**
1. Karıştırıcıda Chrome'u sessize ALMA: yakalama da susuyor (ölçüldü).
2. Ayarlar > Sistem > Ses > Ses karıştırıcı > uygulamalar listesinde Chrome > Çıkış aygıtı: **kullanılmayan bir çıkış** seç.
   - 4K ekran HDMI ile bağlıyken: Chrome'u ekranın HDMI ses çıkışına (hoparlörü yoksa sessiz kalır) ya da tersine yönlendir; upscaler varsayılan çıkıştan çalar (`--audio-device "Hoparlör"` ile ad parçasıyla seçilebilir).
   - Tek çıkış varsa (sadece laptop hoparlörü): ücretsiz sanal kablo (VB-CABLE) kurup Chrome'u "CABLE Input"a yönlendir. Sürücü kurulumu senin kararın.
3. Kontrol: `watch --info` açıkken bilgi katmanında "ses: çalıyor" ve hata ms değeri akıyorsa yakalama yönlendirmeden sonra da çalışıyor. (Microsoft belgesi: process loopback belirli bir çıkışa bağlı değil; bu laptopta ikinci çıkış olmadığı için denenemedi.)

**Ekran:** 4K ekranda NVIDIA/Windows ayarında 3840x2160 **60 Hz** seçili olmalı (144 Hz laptop panelinde 60'a vsync kilidi yok, hat kendi saatiyle akar).

**Chrome:** Yayın tam ekran (1920x1080) olsun. Başka pencere Chrome'u tamamen kapatmasın (watch bulunca Chrome'u öne alıyor).

## Bekleyen canlı işler (Yiğit'e bağlı)

- ⏳ 2026-09-16: **TOD'da 60-100 dk canlı watch sınavı** (`watch --dump-timing --info --run-name tod_uzun`). Yiğit TOD'u açmadı; uzun koşu test klibiyle yapıldı (`runs/uzun60_info`). Maç ya da canlı yayın açıkken koşulacak. 2026-09-16 18:13 denemesi (`runs/tod_uzun`): ekranda maç yoktu, 88 sn'de Yiğit durdurdu. Hat 60 FPS, geç tik 0; TOD kare akışı 39. ve 85. sn'de kesildi (yakalama 6 sn'de toparlandı), 88. sn'de TOD penceresi simge durumuna geçti. Laptop ekranı 1080p olduğu için 4K hissi yok. **Tekrar: maç varken ve harici 4K ekran bağlıyken.**
- ⏳ 4K ekran gelince 5 dk kontrol listesi: `reports/2026-09-16-hat14-mac-gunu-surumu.md`.

## Keşif testleri (Faz 0)

- ✅ 2026-09-15: Yakalama testi (agent yaptı, Yiğit "sen yap" dedi): görüntü geliyor, siyah değil.
- ✅ 2026-09-15: media-internals (agent yaptı): 1920x1080, 50p, H.264 High, ~4,8 Mbps, Widevine L3.
- ⏳ Harici 4K ekran bağlanınca: çıkış NVIDIA'dan mı gidiyor, 4K 60 Hz açılıyor mu?
- ⏳ Başka kanallar (beIN 2-3-4) da 1080p50 mi?

## Cevaplananlar

- 2026-09-15: Kaynak ne? **TOD.**
- 2026-09-15: Hangi ekran? **Harici 4K ekran (B), hedef 4K60.**
- 2026-09-15: Gecikme? **1-2 saniye kabul, ses senkron olmalı.**
- 2026-09-15: Yaklaşım? **A: öğretmen-öğrenci (tavan + gerçek zamanlı motor).**
- 2026-09-15: 4K60 çekim imkanı? **C: yok.** Yiğit "veriye ne gerek var, TOD'dan canlı açarım ya da YouTube adil kullanım olmaz mı?" diye sordu. Veri ihtiyacının sebebi ve YouTube'un hukuki durumu anlatıldı. Veri kaynağı olarak YouTube'daki Creative Commons lisanslı videolar önerildi.
- 2026-09-15: Canlı göz testi erkene alınsın mı? Yiğit: **"HER ŞEY CANLI LAZIM, UNUTMA."** Yol haritası canlı öncelikli v2'ye çevrildi, çevrimdışı tavan fazı kaldırıldı. proje notuna kesin çizgi olarak yazıldı.
- 2026-09-15: Dev modeller eğitimde öğretmen olarak kullanılsın mı? **Evet (A).** Şart: eğitim bitince küçük model canlı maçı 4K 60 FPS verebilmeli.
- 2026-09-15: v2 onayı yerine Yiğit sordu: "Direkt kendi modelimize niye geçmiyoruz? Kaç saat sürer? Vast.ai fiyatlarını araştır." Maliyet raporu yazıldı, yol haritası iki paralel hatta (v3) çevrildi.
- 2026-09-15: Bütçe? Yiğit: **"Çok param yok."** Plan 0 TL varsayımıyla v4'e çevrildi (laptop geceleri + Kaggle, öğretmen ertelendi).
- 2026-09-15: Yiğit: **"Az dolarla buluta OK'yim, güzel iş çıkacaksa."** Kanıt kapısı ve harcama tavanları önerildi.
- 2026-09-15: Hedef hâlâ canlı 4K60 mı? Yiğit iki kez sordu. **Evet.** Eski çelişkiler temizlendi, proje notuna "AMAÇ" satırı eklendi. Bulut sadece eğitimde, maç sırasında yok.
- 2026-09-15: Plan v4 + bulut kuralları (kanıt kapısı, koşu ≤ $10, toplam ≤ $30) onayı? Yiğit: **"He iyi tm."** Onaylandı. Oturum kapandı, yarın Faz 0.
- ⏳ 2026-09-16: Algısal kayıp (VGG/LPIPS) için hazır ağırlık indirilsin mi (~500 MB, pytorch.org)? GAN tek başına yetmezse sorulacak.
- ⏳ 2026-09-16: F26 modellerini gerçek TOD'da ölçmek için TOD'da bir tekrar ya da özet tam ekran açık olmalı (`sharpness_probe --window ... --sr rt4ksr-x2-ft2 rt4ksr-x2-<yeni>`).

**2026-09-19 15:25 (goal, TOD tıkanması):** TOD oynatıcısı önce duraklatılmıştı (agent Play'e bastı). Sonra oynatıcı "oynuyor" gösterdiği halde görüntü akmadı (300 sn'de 412 kare, 52 donma; ham modda da aynı). Onarımlı modun TOD'da 5 dk testi ve keskinlik ölçümü bunun için bekliyor. Yiğit: TOD'da oynayan bir yayın/tekrar aç, tam ekran bırak.

## 2026-09-19 gece (AI yeniden çizim)
- TOD açıkken 2 dk canlı keskinlik ölçümü (kriter 1) için yayın açılsın; agent `--probe-sharp 2` ile ölçecek.
- Hangi kısayol: "Maç TV AI" (güç 1,3) mı "agresif" (1,45) mi? Gözle seç.
