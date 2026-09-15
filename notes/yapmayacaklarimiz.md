# Yapmayacaklarımız

Her maddenin bir sebebi var. Sebep ortadan kalkarsa madde yeniden tartışılabilir.

## Kesin çizgiler

- **DRM kırmak ya da atlatmak yok.** Sebep: yasa dışı ve TOD kullanım koşullarına aykırı. İşletim sisteminin normal ekran yakalama API'si siyah veriyorsa o yol kapanır, çevresinden dolaşmayız.
- **TOD yayınını kaydetmek, saklamak, veri seti yapmak yok.** Sebep: telif ve kullanım koşulları. İşleme bellekte, anlık yapılır ve diske yazılmaz.
- **Başkasının programını paketleyip "yaptık" demek yok** (SVP, Lossless Scaling, RTX VSR). Sebep: Yiğit'in amacı yeni bir şey yapmak. Bunlar sadece kıyas ölçütü olarak kullanılır: "Bizimki bunlardan iyi mi?"

## Teknik olarak elenenler

- **Gerçek zamanlı yolda difüzyon modeli yok.** Sebep: canlı 4K 60 FPS için çok yavaş. Çevrimdışı "öğretmen model" olarak (eğitim verisi üretmek için) düşünülebilir.
- **Büyük transformer SR modelleri yok** (SwinIR, HiT-SR vb.). Sebep: gerçek zamanlıya uygun değiller.
- **Yüksek çözünürlükte ara kare üretimi yok.** Sebep: pahalı. Hareket, düşük çözünürlükte hesaplanır.
- ~~Laptop ekranında 4K çıktı yok.~~ Çözüldü (2026-09-15): İzleme harici 4K ekranda olacak, hedef 4K60.
- **Milisaniye seviyesinde düşük gecikme için uğraşmak yok.** Sebep: Yiğit 1-2 saniye gecikmeyi kabul ediyor. Kare başına süre sınırı yerine ortalama verime bakılır.

- **Creative Commons olmayan YouTube videolarını (ör. Süper Lig özetleri) indirmek yok.** Sebep: YouTube kullanım koşulları izinsiz indirmeyi yasaklıyor. Türkiye'de açık uçlu bir "adil kullanım" kuralı yok (2026-09-15). CC lisanslı videolar serbest, atıf yapılır.
- **Pexels videolarıyla eğitim yok.** Sebep: Pexels kullanım koşulları ML için veri toplamayı yasaklıyor (2026-09-15).
- **4K kareleri PNG olarak diske dökmek yok.** Sebep: dakikada ~40-50 GB (tahmin), diskte 111 GB boş yer var. Videolar sıkıştırılmış tutulur, anlık çözülür.
- **Ticari olmayan lisanslı modellerden (RVRT, BiM-VFI vb.) distillation yok.** Sebep: Türetilen ağırlıklar GitHub'da yayınlanamaz. Bu modeller sadece yerel kıyas için kullanılır.

## Para

- **Yiğit'in açık onayı olmadan paralı bulut (vast.ai vb.) kullanmak yok.** Sebep: bütçe çok kısıtlı (2026-09-15). Varsayılan harcama 0. Yiğit az dolara açık, ama sadece iyi sonuç çıkacaksa.
- **Kanıtsız bulut harcaması yok.** Bir fikir önce laptopta ya da Kaggle'da rakibi geçmeli. Koşu başına ≤ $10, toplam ≤ $30 (onaylandı, 2026-09-15). Maç sırasında bulut yok.
- **Bulutta veri bırakmak yok.** Sebep: Vast.ai'da depolama ücreti makine dursa da işler, unutulursa fatura çıkar.
- **Dev öğretmen modelleri (SeedVR2, SwiftVR) şimdilik yok.** Sebep: laptopa ve bedava GPU'lara sığmıyorlar. Temiz 4K klipler cevap anahtarı olarak yeterli.

## Açık kaynak (GitHub) kuralları

- **Lisansı yeniden dağıtıma izin vermeyen model ya da veriyle eğitilmiş ağırlık yayınlamak yok.** Sebep: GitHub'da paylaşılacak. Bazı güçlü VSR modelleri ve veri setleri ticari olmayan kullanım ya da NDA şartı taşıyor. Her parçanın lisansı kontrol edilip `reports/` altına yazılır.
- **Projeyi "TOD için" ya da "DRM'li yayın için" diye tanıtmak yok.** Sebep: Genel amaçlı bir video iyileştirici (dosya ve pencere) olarak konumlanır. README'de DRM atlatmadığı açıkça yazılır.
- **Ölçmeden "sığar" demek yok.** Sebep: tüm hız sayıları başka kartlardan. Her iddia bizim laptopta ölçülür.
