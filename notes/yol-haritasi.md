# Yol haritası (taslak v4: sıfır bütçe, iki paralel hat)

**AMAÇ: Canlı maçı (TOD) harici 4K ekranda 4K 60 FPS izlemek, 4070 Laptop'ta.**

## Şu an neredeyiz (2026-09-15 akşam, limit nedeniyle durduruldu)

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
