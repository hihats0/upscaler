"""GPU halka tamponu.

Yakalanan kareler kaynak cozunurlugunde (BGRA uint8) VRAM'de tutulur. Her kare
SourceClock'tan ideal zaman damgasi alir. Cikis dongusu pick() ile iki kare ve
alpha ister; o slotlar islem bitene kadar kilitlenir, yakalama onlari ezmez.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .timeline import FrameStamp, SourceClock, pick_frames


@dataclass
class Entry:
    stamp: FrameStamp
    slot: int
    meta: Any = None


@dataclass
class Pick:
    a: Entry
    b: Entry
    alpha: float
    hold: bool          # tampon bosaldi, son kare tekrarlaniyor
    newest_t: float
    _ring: "GpuFrameRing" = field(repr=False, default=None)
    # Secim anindaki slot gorunumleri: kaynak boyutu degisip tampon yeniden ayrilsa da
    # bu tensorler eski bellegi canli tutar, islem yarida kalmaz.
    fa: torch.Tensor | None = field(repr=False, default=None)
    fb: torch.Tensor | None = field(repr=False, default=None)
    gen: int = 0

    def frames(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fa, self.fb

    def release(self) -> None:
        self._ring._release(self.gen, self.a.slot, self.b.slot)


class GpuFrameRing:
    def __init__(self, capacity: int, device: str = "cuda") -> None:
        self.capacity = capacity
        self.device = device
        self.clock = SourceClock()
        self.slots: torch.Tensor | None = None
        self.shape: tuple[int, int] | None = None
        self.resets = 0
        self.generation = 0  # her yeniden ayirmada artar; eski secimlerin release'i yok sayilir
        self.restamps = 0
        self.repeats = 0
        self.fixes = 0
        self.hole_fills = 0
        self._tentative: Entry | None = None
        self._orphans: set[int] = set()
        self._pinned: torch.Tensor | None = None
        self._stream = torch.cuda.Stream(device=device)
        self._entries: deque[Entry] = deque()
        self._free: list[int] = []
        self._in_use: dict[int, int] = {}
        self._lock = threading.Lock()

    # --- yakalama tarafi (WGC is parcacigi) ---------------------------------
    def push(self, raw: float, bgra: np.ndarray, meta: Any = None) -> FrameStamp:
        h, w = bgra.shape[:2]
        if self.shape != (h, w):
            self._reallocate(h, w)
        self._pinned_np[...] = bgra
        with self._lock:
            slot = self._take_slot()
        with torch.cuda.stream(self._stream):
            self.slots[slot].copy_(self._pinned, non_blocking=True)
        self._stream.synchronize()  # pinned tampon bir sonraki karede yeniden kullanilir
        with self._lock:
            was_locked = self.clock.period is not None
            relocks = self.clock.relocks
            stamp = self.clock.push(raw)
            if (not was_locked and self.clock.period is not None) or self.clock.relocks != relocks:
                self._restamp()  # ilk kilit ya da periyot dogrulamasi saati yeniden kurdu
            if stamp.fix_prev is not None and self._entries:
                # Saat, onceki karenin aslinda bir onceki yuvaya ait oldugunu anladi.
                self.fixes += 1
                self._entries[-1].stamp = stamp.fix_prev
            if stamp.repeat and self._entries:
                # Karar bir sonraki kareye ertelenir: gercek tekrar mi (pencerede video disi
                # degisiklik), yoksa 144 Hz'de biriken gecikmeden sonra 1 vsync ile "yetisen"
                # gercek kare mi? Sonraki kare bir sira atlarsa ikincisidir. Tampon 1-2 sn
                # geriden okundugu icin erteleme bedava.
                if self._tentative is not None:
                    self._retire(self._tentative.slot)
                    self.repeats += 1
                self._tentative = Entry(stamp, slot, meta)
            else:
                self._resolve_tentative(stamp)
                self._entries.append(Entry(stamp, slot, meta))
        return stamp

    def _resolve_tentative(self, new: FrameStamp) -> None:
        tent, self._tentative = self._tentative, None
        if tent is None or not self._entries:
            if tent is not None:
                self._retire(tent.slot)
            return
        last = self._entries[-1]
        period = self.clock.period
        if period and not new.discontinuity and new.index >= last.stamp.index + 2:
            # Sira bos kalmis: "tekrar" sanilan kare o siranin gercek karesi.
            self.hole_fills += 1
            idx = last.stamp.index + 1
            t = new.t - period * (new.index - idx)
            self._entries.append(Entry(FrameStamp(idx, t, tent.stamp.raw, False), tent.slot, tent.meta))
        else:
            # Gercek tekrar: ayni karenin daha yeni icerigi son girdinin yerine gecer.
            self.repeats += 1
            self._retire(last.slot)
            self._entries[-1] = Entry(FrameStamp(last.stamp.index, last.stamp.t, tent.stamp.raw,
                                                 last.stamp.discontinuity, True), tent.slot, tent.meta)

    def _retire(self, slot: int) -> None:
        """Girdisi silinen slotu bosalt; kullanimdaysa islem bitince bosalir."""
        if slot in self._in_use:
            self._orphans.add(slot)
        else:
            self._free.append(slot)

    def _restamp(self) -> None:
        """Saat kilitlendi: isinmada titrek damgayla giren kareleri izgaraya oturt.

        Izgarada ayni kareye dusen girdilerden sadece en yenisi kalir.
        """
        period = self.clock.period
        kept: list[Entry] = []
        for e in self._entries:
            index, t = self.clock.ideal(e.stamp.raw)
            if kept and index <= kept[-1].stamp.index:
                prev = kept[-1]
                if e.stamp.raw - prev.stamp.raw < 0.5 * period:
                    # gercekten tekrar: yeni icerik eskisinin yerine gecer
                    self._retire(prev.slot)
                    kept.pop()
                    index, t = prev.stamp.index, prev.stamp.t
                else:
                    # gercek yeni kare, titreme numarayi carpistirdi: bir sonraki sira
                    index = prev.stamp.index + 1
                    t = prev.stamp.t + period
            e.stamp = FrameStamp(index, t, e.stamp.raw, e.stamp.discontinuity)
            kept.append(e)
        self._entries.clear()
        self._entries.extend(kept)
        self.restamps += 1

    def _reallocate(self, h: int, w: int) -> None:
        with self._lock:
            if self.shape is not None:
                self.resets += 1
            self.generation += 1
            self.shape = (h, w)
            self.slots = None
            torch.cuda.empty_cache()
            self.slots = torch.empty((self.capacity, h, w, 4), dtype=torch.uint8, device=self.device)
            self._pinned = torch.empty((h, w, 4), dtype=torch.uint8).pin_memory()
            self._pinned_np = self._pinned.numpy()
            self._entries.clear()
            self._tentative = None
            self._orphans.clear()
            self._free = list(range(self.capacity))
            self._in_use.clear()
            self.clock = SourceClock()

    def _take_slot(self) -> int:
        if self._free:
            return self._free.pop()
        for k, e in enumerate(self._entries):  # en eski, kullanimda olmayan
            if e.slot not in self._in_use:
                del self._entries[k]
                return e.slot
        raise RuntimeError("tampon dolu ve tum slotlar kullanimda")

    # --- cikis tarafi (ana is parcacigi) -------------------------------------
    def pick(self, s: float) -> Pick | None:
        with self._lock:
            if not self._entries:
                return None
            times = [e.stamp.t for e in self._entries]
            r = pick_frames(times, s)
            if r is None:
                return None
            i, j, alpha = r
            a, b = self._entries[i], self._entries[j]
            for slot in (a.slot, b.slot):
                self._in_use[slot] = self._in_use.get(slot, 0) + 1
            hold = s > times[-1]
            return Pick(a, b, alpha, hold, times[-1], self,
                        self.slots[a.slot], self.slots[b.slot], self.generation)

    def _release(self, gen: int, *slots: int) -> None:
        with self._lock:
            if gen != self.generation:
                return  # tampon bu arada yeniden ayrildi: sayim zaten sifirlandi
            for slot in slots:
                n = self._in_use.get(slot, 0) - 1
                if n <= 0:
                    self._in_use.pop(slot, None)
                    if slot in self._orphans:
                        self._orphans.discard(slot)
                        self._free.append(slot)
                else:
                    self._in_use[slot] = n

    def __len__(self) -> int:
        return len(self._entries)
