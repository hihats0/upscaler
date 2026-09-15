"""Senkron ses (Hat 1.4 paket 3).

Chrome sesi ProcTap (WASAPI process loopback) ile yakalanir, goruntu hattinin gercek
gecikmesiyle bizim cikisimizdan calinir. Ses SADECE bellekte tutulur (birkac saniyelik halka),
diske yazilmaz.

Saat modeli (hepsi perf_counter saniyesi):
- Yakalama: n. ornegin yakalama zamani c(n) = c0 + n / fs_cap. Parcalarin gelis zamani titrer
  (p90 11 ms), gelis her zaman yakalamadan SONRA oldugu icin alt zarf (en erken gelisler)
  kullanilir; fs_cap uzun pencerede egimle izlenir (ses kartinin saati perf_counter'dan kayar).
- Goruntu: kaynak zamani s olan kare, ekrana T + sunum_gecikmesi aninda cikar, s = T - delay.
  Chrome'da ses ve goruntu ayni anda cikiyordu; yani yakalama zamani c olan ses ornegi
  c + delay + sunum_gecikmesi aninda calinmali. Toplam gecikme = AvSync.video_delay().
- Calma: PortAudio geri cagrisi her blokta DAC zamanini verir (outputBufferDacTime, ayni saate
  cevrilir). Blok icin istenen yakalama zamani = dac - toplam_gecikme -> okuma konumu. Okuma
  imleci bu hedefe kucuk hiz ayariyla (en fazla +-%0,5, dogrusal yeniden ornekleme) kilitlenir;
  50 ms'yi asan hatada sert atlama yapilir. Boylece uzun koşuda kayma birikmez.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

FS = 48000
CH = 2


class CaptureClock:
    """Ornek sayisi -> yakalama zamani (alt zarf + egim)."""

    def __init__(self, fs: float = FS, window_s: float = 20.0) -> None:
        self.fs_nominal = fs
        self.fs = fs
        self.window_s = window_s
        self._pts: deque[tuple[int, float]] = deque()  # (parca sonu ornek sayisi, gelis zamani)
        self._c0: float | None = None

    def add(self, end_sample: int, arrival: float) -> None:
        self._pts.append((end_sample, arrival))
        while self._pts and arrival - self._pts[0][1] > self.window_s:
            self._pts.popleft()
        if len(self._pts) >= 50 and self._pts[-1][1] - self._pts[0][1] > 5.0:
            # Egim: pencerenin ilk ve son ceyreginin alt zarflarindan (gelis titremesine dayanikli).
            pts = list(self._pts)
            q = max(len(pts) // 4, 5)
            n1, t1 = self._lower(pts[:q])
            n2, t2 = self._lower(pts[-q:])
            if t2 > t1 and n2 > n1:
                fs = (n2 - n1) / (t2 - t1)
                if abs(fs - self.fs_nominal) < 0.005 * self.fs_nominal:
                    self.fs += 0.1 * (fs - self.fs)
        # c0: sabit fs ile tum noktalarin alt zarfi
        self._c0 = min(t - n / self.fs for n, t in self._pts)

    def _lower(self, pts: list[tuple[int, float]]) -> tuple[float, float]:
        best = min(pts, key=lambda p: p[1] - p[0] / self.fs)
        return float(best[0]), best[1]

    def time_of(self, sample: float) -> float | None:
        return None if self._c0 is None else self._c0 + sample / self.fs

    def sample_at(self, t: float) -> float | None:
        return None if self._c0 is None else (t - self._c0) * self.fs


class AudioRing:
    """Son birkac saniyelik stereo float32 ses. Yazici yakalama is parcacigi, okuyucu calma geri cagrisi."""

    def __init__(self, seconds: float = 6.0) -> None:
        self.cap = int(seconds * FS)
        self.buf = np.zeros((self.cap, CH), np.float32)
        self.written = 0  # toplam yazilan ornek
        self.lock = threading.Lock()

    def write(self, x: np.ndarray) -> int:
        n = len(x)
        with self.lock:
            if n >= self.cap:
                x = x[-self.cap:]
                self.written += n - self.cap
                n = self.cap
            i = self.written % self.cap
            k = min(n, self.cap - i)
            self.buf[i:i + k] = x[:k]
            if k < n:
                self.buf[:n - k] = x[k:]
            self.written += n
            return self.written

    def read_interp(self, pos: np.ndarray) -> np.ndarray:
        """Mutlak ornek konumlarinda dogrusal ara deger. Tamponda olmayan konumlar sessiz."""
        with self.lock:
            lo = self.written - self.cap
            out = np.zeros((len(pos), CH), np.float32)
            p0 = np.floor(pos).astype(np.int64)
            ok = (p0 >= lo) & (p0 + 1 < self.written)
            if ok.any():
                i0 = p0[ok] % self.cap
                i1 = (p0[ok] + 1) % self.cap
                f = (pos[ok] - p0[ok]).astype(np.float32)[:, None]
                out[ok] = self.buf[i0] * (1 - f) + self.buf[i1] * f
            return out


@dataclass
class PlayerStats:
    blocks: int = 0
    hard_jumps: int = 0
    underruns: int = 0
    err_ms: float = 0.0       # son blokta okuma imleci - hedef (ms)
    ratio: float = 1.0        # uygulanan hiz orani
    silent_blocks: int = 0


class SyncedPlayer:
    """Yakalanan sesi hedef gecikmeyle calar, surekli kayma duzeltmesiyle."""

    MAX_ADJ = 0.005
    HARD_JUMP_S = 0.05

    def __init__(self, ring: AudioRing, clock: CaptureClock, device=None, blocksize: int = 480,
                 latency: str | float = "low") -> None:
        self.ring, self.clock = ring, clock
        self.device, self.blocksize, self.latency = device, blocksize, latency
        self.delay_s: float | None = None  # toplam gecikme (yakalama -> calma); None: sessiz
        self.volume = 1.0
        self.stats = PlayerStats()
        self.err_log: deque[tuple[float, float]] = deque(maxlen=20000)  # (zaman, hata ms)
        self._cursor: float | None = None
        self._stream = None
        self._pa_offset: float | None = None  # perf_counter - stream.time
        self.on_block = None  # istege bagli olcum kancasi: (dac zamani, cikis blogu)

    def start(self) -> "SyncedPlayer":
        import sounddevice as sd
        self._stream = sd.OutputStream(samplerate=FS, channels=CH, dtype="float32", device=self.device,
                                       blocksize=self.blocksize, latency=self.latency, callback=self._cb)
        self._stream.start()
        return self

    def output_latency_ms(self) -> float:
        return self._stream.latency * 1000 if self._stream else float("nan")

    def _to_perf(self, pa_time: float) -> float:
        # PortAudio zamani ile perf_counter arasindaki fark (WASAPI'de ikisi de QPC tabanli; fark sabit)
        if self._pa_offset is None:
            self._pa_offset = time.perf_counter() - self._stream.time
        return pa_time + self._pa_offset

    def _cb(self, outdata, frames, tinfo, status) -> None:
        st = self.stats
        st.blocks += 1
        if status.output_underflow:
            st.underruns += 1
        d = self.delay_s
        dac = self._to_perf(tinfo.outputBufferDacTime) if tinfo.outputBufferDacTime else time.perf_counter()
        target = self.clock.sample_at(dac - d) if d is not None else None
        if target is None:
            outdata.fill(0)
            st.silent_blocks += 1
            self._cursor = None
            return
        if self._cursor is None or abs(self._cursor - target) > self.HARD_JUMP_S * FS:
            if self._cursor is not None:
                st.hard_jumps += 1
            self._cursor = target
        err = self._cursor - target  # ornek; + ise ileri okuyoruz (ses erken), yavasla
        # Orantili duzeltme: hatayi ~1 sn'de kapat, hiz ayari sinirli (duyulmaz).
        adj = max(-self.MAX_ADJ, min(self.MAX_ADJ, -err / FS))
        ratio = 1.0 + adj
        pos = self._cursor + np.arange(frames) * ratio
        outdata[:] = self.ring.read_interp(pos) * self.volume
        if self.on_block is not None:
            self.on_block(dac, outdata)
        self._cursor += frames * ratio
        st.err_ms = err / FS * 1000
        st.ratio = ratio
        self.err_log.append((dac, st.err_ms))

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class ChromeAudio:
    """ProcTap yakalama -> AudioRing + CaptureClock."""

    def __init__(self, pid: int, ring: AudioRing, clock: CaptureClock) -> None:
        self.pid, self.ring, self.clock = pid, ring, clock
        self.samples = 0
        self.chunks = 0
        self.last_arrival: float | None = None
        self.peak = 0.0
        self._tap = None
        self.on_chunk = None  # istege bagli: (bitis_ornegi, float32 ndarray) -> None (olcum)

    def start(self) -> "ChromeAudio":
        from proctap import ProcessAudioCapture
        self.started = time.perf_counter()
        self._tap = ProcessAudioCapture(self.pid, on_data=self._on_data)
        self._tap.start()
        return self

    def _on_data(self, pcm: bytes, frames: int) -> None:
        now = time.perf_counter()
        x = np.frombuffer(pcm, dtype=np.float32)
        if x.size < CH:
            return
        x = x[: x.size // CH * CH].reshape(-1, CH)
        end = self.ring.write(x)
        self.clock.add(end, now)
        self.samples = end
        self.chunks += 1
        self.last_arrival = now
        if self.on_chunk is not None:
            self.on_chunk(end, x)

    def alive(self, timeout: float = 3.0) -> bool:
        now = time.perf_counter()
        if self.last_arrival is None:  # henuz parca gelmedi: baslangic payi
            return now - getattr(self, "started", now) < timeout
        return now - self.last_arrival < timeout

    def stop(self) -> None:
        if self._tap is not None:
            try:
                self._tap.close()
            except Exception:
                pass
            self._tap = None


def chrome_main_pid() -> int | None:
    """Ebeveyni chrome.exe olmayan chrome.exe = tarayici ana sureci (ProcTap agaci dahil eder)."""
    import psutil
    procs = [p for p in psutil.process_iter(["name", "ppid", "pid"]) if (p.info["name"] or "").lower() == "chrome.exe"]
    pids = {p.info["pid"] for p in procs}
    roots = [p.info["pid"] for p in procs if p.info["ppid"] not in pids]
    return roots[0] if roots else None


def window_pid(hwnd: int) -> int:
    import win32process
    return win32process.GetWindowThreadProcessId(hwnd)[1]


def root_pid(pid: int, name: str | None = None) -> int:
    """Ayni adli ebeveyn zincirinin en ustu (Chrome penceresi ana surecte olsa da garanti)."""
    import psutil
    p = psutil.Process(pid)
    name = name or p.name()
    while True:
        par = p.parent()
        if par is None or par.name() != name:
            return p.pid
        p = par
