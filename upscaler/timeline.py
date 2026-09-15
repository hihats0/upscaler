"""Kaynak zaman cizgisi (F23).

Yakalanan karelerin zamanlari ekranin vsync izgarasina yapisir ve titrer.
SourceClock her yeni kareye:
  - kaynaktaki sira numarasini (dusen kareler atlanir),
  - esit aralikli ideal zaman damgasini
verir. pick_frames ise cikis anindaki kaynak zamani icin hangi iki karenin
arasinda oldugumuzu ve ara oranini (alpha) bulur.
"""
from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass

# Yayinlarda gorulen kare hizlari. Olculen periyot bunlardan birine yakinsa ona oturtulur.
NOMINAL_RATES = (23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0)


@dataclass(frozen=True)
class FrameStamp:
    index: int            # kaynaktaki kare sirasi (dusen kare varsa atlar)
    t: float              # ideal zaman (sn, perf_counter ile ayni saat)
    raw: float            # yakalamanin gordugu ham zaman
    discontinuity: bool   # uzun bosluk sonrasi ilk kare (akis yeniden baslar)
    repeat: bool = False  # izgarada ayni kareye dustu: video karesi degil, pencerede baska degisiklik
    fix_prev: "FrameStamp | None" = None  # onceki kare aslinda bir onceki yuvaya aitmis (gec gelmis)


def snap_period(period: float, tol: float = 0.03) -> float:
    """Periyodu en yakin yayin kare hizina oturtur, yakin degilse aynen dondurur."""
    p = nominal_period(period, tol)
    return p if p is not None else period


def nominal_period(period: float, tol: float = 0.03) -> float | None:
    """Periyot bir yayin kare hizina yakinsa onun tam periyodu, degilse None."""
    best = min(NOMINAL_RATES, key=lambda r: abs(1.0 / r - period))
    return 1.0 / best if abs(1.0 / best - period) <= tol * (1.0 / best) else None


def renumber(hist: list[tuple[int, float]], period: float) -> list[tuple[int, float]]:
    """Isinma gecmisini bulunan periyotla yeniden numaralar.

    Ilk karenin titremesi tum numaralari kaydirmasin diye faz, tum gecmisin
    ortalamasindan alinir. Sira her zaman artar; yarim periyottan kisa aralikla
    gelen (tekrar) ayni numarayi alir.
    """
    if not hist:
        return []
    base_i, base_r = hist[0]
    frac = [((r - base_r) / period) for _, r in hist]
    phase = math.atan2(sum(math.sin(2 * math.pi * f) for f in frac),
                       sum(math.cos(2 * math.pi * f) for f in frac)) / (2 * math.pi)
    out: list[tuple[int, float]] = []
    for (_, r), f in zip(hist, frac):
        i = base_i + round(f - phase)
        if out:
            pi, pr = out[-1]
            if r - pr < 0.5 * period:
                i = pi
            elif i <= pi:
                i = pi + 1
        out.append((i, r))
    return out


def estimate_period(raws: list[float], min_span: float = 1.0) -> float | None:
    """Ham zamanlardan kaynak periyodunu bulur. Dusen karelere dayaniklidir.

    Her aday periyot P icin zamanlari P'lik bir "saat kadranina" sarariz ve
    faz tutarliligini olceriz (R = |ortalama e^(i*2*pi*t/P)|). Dogru P'de
    kareler kadranda ayni yere duser (R yuksek). Dusen kare fazi bozmaz.
    Alt katlar da (25 FPS kaynak icin P=20 ms) yuksek R verir, bu yuzden
    yuksek R'liler icinden EN UZUN periyot secilir.

    Tuzak (2026-09-15): 1 sn'lik pencerede 144 Hz ekrandaki 50 FPS kaynak (3-3-...-2
    vsync duzeni + 13,9/27,8 ms titreme) 48 Hz'e R=0,28, 50 Hz'e R=0,08 verebildi; saat
    20,833 ms'e kilitlenip tekrar/bosluk uretti (TOD, onizleme acik). 3 sn'de ayni kayit
    50 Hz veriyor. Ilk kilit hizli kalsin diye pencere 1 sn; SourceClock kilitten sonra
    periyodu daha uzun gecmisle bir kez dogrular (tests/data/tod_window_144hz_48hz_trap.csv).
    """
    if len(raws) < 8 or raws[-1] - raws[0] < min_span:
        return None
    t0 = raws[0]
    scores = {}
    for rate in NOMINAL_RATES:
        p = 1.0 / rate
        c = sum(math.cos(2 * math.pi * (r - t0) / p) for r in raws)
        s = sum(math.sin(2 * math.pi * (r - t0) / p) for r in raws)
        scores[p] = math.hypot(c, s) / len(raws)
    best = max(scores.values())
    if best < 0.15:
        return None
    return max(p for p, r in scores.items() if r >= 0.6 * best)


class SourceClock:
    """Ham yakalama zamanlarindan ideal kare zamanlari uretir.

    Ilk ~1 sn periyot bilinmez (isinma): sira +1 artar, t = ham zaman.
    Periyot bulununca gecmis yeniden numaralanir ve saat kilitlenir.
    """

    REPEAT_FRACTION = 0.5  # dolu yuvaya dusup onceki kareden bu kadar periyottan once gelen = tekrar
    SLOT_FRACTION = 0.35   # dolu yuvaya dusup yuvanin ideal zamanina bu kadar yakin olan = tekrar

    def __init__(self, window: int = 50, fit_after: int = 16,
                 reset_after_periods: float = 10.0) -> None:
        self.window = window
        self.fit_after = fit_after
        self.reset_after_periods = reset_after_periods
        self.period: float | None = None
        self._hist: deque[tuple[int, float]] = deque(maxlen=window)
        self._warm: list[float] = []  # isinma sirasindaki ham zamanlar
        self._last: FrameStamp | None = None
        self._offset = 0.0  # ideal t = offset + period * index
        self.repeats = 0
        self.fixes = 0
        self._before_last_index = -(10 ** 9)  # sondan bir onceki kabul edilen karenin sirasi
        # Periyot dogrulamasi: 1 sn'lik isinma 144 Hz ekranda 50 FPS'i 48 Hz sanabiliyor.
        # Kilitten sonra verify_span sn'lik ham gecmisle bir kez yeniden tahmin edilir.
        self.verify_span = 3.0
        self.relocks = 0
        self._recent: deque[float] = deque(maxlen=400)
        self._verified = False

    def push(self, raw: float) -> FrameStamp:
        last = self._last
        if last is None:
            return self._accept(0, raw, discontinuity=False)

        gap_limit = self.reset_after_periods * self.period if self.period else 0.5
        if raw - last.raw > gap_limit:
            # Duraklama: gecmisi at, sirayi surdur, zamani yeniden capala.
            self._hist.clear()
            self._warm.clear()
            self._recent.clear()
            self._verified = False
            return self._accept(last.index + 1, raw, discontinuity=True)

        if self.period is None:
            index = last.index + 1
        else:
            index = round((raw - self._offset) / self.period)
            if index <= last.index:
                grid = self._offset + self.period * last.index
                close_in_time = raw - last.raw < self.REPEAT_FRACTION * self.period
                close_to_slot = abs(raw - grid) < self.SLOT_FRACTION * self.period
                late_prev = (last.raw - grid) > self.REPEAT_FRACTION * self.period * 0.5
                if close_to_slot and late_prev and last.index - 1 > self._before_last_index:
                    # Onceki kare yuvasina gec gelip bu yuvaya yuvarlanmis, bu kare ise yuvasina
                    # zamaninda geldi. Iki kare ayni yuvada olamaz: onceki bos kalan bir onceki
                    # yuvaya aittir. Onceki geri tasinir, bu kare yeni kare olarak kabul edilir.
                    self.fixes += 1
                    fixed = FrameStamp(last.index - 1, last.t - self.period, last.raw, last.discontinuity)
                    hi, hr = self._hist[-1]
                    if hi == last.index:
                        self._hist[-1] = (hi - 1, hr)
                    self._last = fixed
                    stamp = self._accept(fixed.index + 1, raw, discontinuity=False)
                    return FrameStamp(stamp.index, stamp.t, stamp.raw, False, fix_prev=fixed)
                if close_in_time or close_to_slot:
                    # Izgarada dolu yuvaya dustu VE gercekten ona yakin: yeni video karesi
                    # degil (oynatici kontrolu, imlec, sayfa animasyonu). Sira ilerlemez.
                    # Bu ham zaman yuvaya daha yakinsa saat onu kullanir.
                    self.repeats += 1
                    hi, hr = self._hist[-1]
                    if hi == last.index and abs(raw - grid) < abs(hr - grid):
                        self._hist[-1] = (hi, raw)
                    return FrameStamp(last.index, last.t, raw, False, repeat=True)
                # Izgara "dolu" diyor ama zaman olarak uzak: saat ofseti kaymis, gercek kare.
                index = last.index + 1
            # index > last.index: izgara yeni yuva diyor. Onceki kare gec gelip bu kare
            # hemen ardindan zamaninda geldiyse de (kisa aralik) bu GERCEK yeni karedir.
        return self._accept(index, raw, discontinuity=False)

    def _accept(self, index: int, raw: float, discontinuity: bool) -> FrameStamp:
        self._hist.append((index, raw))
        self._recent.append(raw)
        relocks = self.relocks
        self._update_period()
        if self.period is not None:
            if self.relocks != relocks:
                index = self._hist[-1][0]  # periyot degisti: sira yeni numaralamadan gelir
            else:
                index = max(index, self._hist[-1][0])  # yeniden numaralama ileri atlatmis olabilir
            # Medyan: fazladan guncellemeler ve gec kareler ofseti kaydirmasin.
            res = sorted(r - self.period * i for i, r in self._hist)
            self._offset = res[len(res) // 2]
            t = self._offset + self.period * index
        else:
            t = raw
        if self._last is not None and t <= self._last.t:
            t = self._last.t + 1e-6  # zaman asla geri gitmez
        stamp = FrameStamp(index, t, raw, discontinuity)
        self._before_last_index = self._last.index if self._last is not None else self._before_last_index
        self._last = stamp
        return stamp

    def ideal(self, raw: float) -> tuple[int, float] | None:
        """Daha once gorulmus bir ham zamani mevcut saate gore yeniden damgalar.

        Saat kilitlenince, isinma sirasinda titrek damgayla tampona girmis
        karelerin zamanini duzeltmek icin kullanilir.
        """
        if self.period is None:
            return None
        index = round((raw - self._offset) / self.period)
        return index, self._offset + self.period * index

    def _update_period(self) -> None:
        if self.period is None:
            self._warm.append(self._hist[-1][1])
            p = estimate_period(self._warm)
            if p is None:
                return
            self.period = p
            self._warm.clear()
            # Gecmisi yeni periyotla yeniden numarala (isinmada dusen kareler).
            self._hist = deque(renumber(list(self._hist), p), maxlen=self.window)
            return
        if not self._verified and self._recent[-1] - self._recent[0] >= self.verify_span:
            self._verified = True
            p = estimate_period(list(self._recent), min_span=self.verify_span)
            if p is not None and abs(p - self.period) > 0.01 * self.period:
                self.period = p
                self.relocks += 1
                self._hist = deque(renumber(list(self._hist), p), maxlen=self.window)
                return
        if len(self._hist) < self.fit_after or self._hist[-1][0] == self._hist[0][0]:
            return
        # Egim sadece izgaraya uyan karelerden (fazladan guncellemeler egimi bozmasin).
        pts = [(i, r) for i, r in self._hist
               if abs(r - (self._offset + self.period * i)) < 0.25 * self.period]
        n = len(pts)
        if n < max(self.fit_after, 0.7 * len(self._hist)):
            return
        mi = sum(i for i, _ in pts) / n
        mr = sum(r for _, r in pts) / n
        var = sum((i - mi) ** 2 for i, _ in pts)
        if var == 0:
            return
        slope = sum((i - mi) * (r - mr) for i, r in pts) / var
        p = nominal_period(slope)
        if p is not None:  # yayin disi bir egim (bozuk numaralama) saati kaydirmasin
            self.period = p


def pick_frames(times: list[float], s: float) -> tuple[int, int, float] | None:
    """Kaynak zamani s icin (i, j, alpha): kare = (1-alpha)*kare_i + alpha*kare_j.

    s ilk kareden onceyse None (henuz gosterilecek bir sey yok).
    s son kareden sonraysa son kare tutulur (tampon bosaldi).
    """
    if not times or s < times[0]:
        return None
    last = len(times) - 1
    if s >= times[last]:
        return (last, last, 0.0)
    j = bisect.bisect_right(times, s)
    i = j - 1
    return (i, j, (s - times[i]) / (times[j] - times[i]))
