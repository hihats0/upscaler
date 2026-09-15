# Soru kuyruğu

Yiğit'e her mesajda tek soru sorulur. Sıradaki sorular burada bekler. Cevaplanınca "Cevaplananlar"a taşınır.

## Sıradakiler (öncelik sırasıyla)

1. **Ses yönlendirme:** Gecikmeli sesi biz çalacağız, Chrome'un kendi sesi duyulmamalı. Karıştırıcıda sessize almak yakalamayı da susturuyor. Chrome'u Windows'ta kullanılmayan bir çıkışa (ör. bağlı olmayan HDMI / sanal kablo) yönlendirmek uygun mu? (Windows ayarı, Yiğit yapar.)
- ~~Yazılım ortamı~~ Cevaplandı 2026-09-15: Yiğit "sen seç" dedi. Python 3.13 + PyTorch + WGC + ProcTap + TensorRT seçildi.
2. **Öğrencilik:** Öğrenci misin? (GitHub Student Developer Pack gibi programlarda bulut kredisi olabilir, araştırılacak.)
3. **Disk:** Harici SSD var mı ya da alınabilir mi? (C: 111 GB boş)
4. **Harici ekran:** Hangi 4K ekran (model, HDMI 2.1 var mı, yenileme hızı)?
5. **Zaman:** Haftada bu projeye kabaca kaç saat ayırabilirsin? (Faz sürelerini buna göre konuşuruz.)
6. **Lisans:** Öneri: kod Apache-2.0, ağırlıklar sadece temiz kaynaklardan. Uygun mu?
7. **GitHub:** Repo baştan herkese açık mı olsun, yoksa ilk çalışan sürümde mi açılsın?
8. **İsim:** Projenin GitHub adı ne olsun?

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
