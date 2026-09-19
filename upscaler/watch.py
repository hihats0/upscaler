"""Tek komutla mac izleme (Hat 1.4 paket 4-5).

    .venv/Scripts/python.exe -m upscaler watch            # TOD Chrome penceresi, Esc ile cikis
    .venv/Scripts/python.exe -m upscaler watch --info     # bilgi katmani acik baslar

Akis: TOD penceresini bul -> WGC yakalama -> GPU tampon + saat kilidi -> 60 Hz cikis dongusu
(RIFE ara kare + RT4KSR, kaynasik TensorRT) -> GPU'da kalan tam ekran sunucu. Chrome sesi ProcTap
ile yakalanir, goruntunun olculen gecikmesiyle calinir.

Dayaniklilik: pencere kapanirsa aranir, yakalama durursa yeniden baslatilir, simge durumunda ve
donmada son kare gosterilir (siyah ekran yok), boyut degisince tampon ve motor yeniden secilir,
ekran listesi degisince pencere uygun ekranda yeniden acilir, ses yakalama/calma dusurse kurulur.
Olaylar runs/watch_<zaman>/events.jsonl, 10 sn'lik istatistik stats.csv, ozet summary.json.
Hicbir kare ya da ses diske yazilmaz; kayitlar sadece sayi.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import subprocess
import threading
import time
import traceback
from array import array
from collections import Counter, deque
from dataclasses import asdict, dataclass

import numpy as np
import torch

from . import schedule, winutil
from .process import make_processor
from .ring import GpuFrameRing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class WatchConfig:
    title: str = "TOD"
    title_must: str = "Google Chrome"   # baslikta bu da gecmeli ("" = sart yok)
    delay: float = 1.5
    out_fps: float = 60.0
    monitor: str = "auto"
    vsync: str = "auto"
    seconds: float = 0.0                # 0: Esc'e kadar
    audio: bool = True
    audio_device: str = ""
    av_offset_ms: float = 0.0
    split: bool = False
    sr: str = "rt4ksr-x2-ft2"           # ince ayarli (yoksa hazir rt4ksr-x2)
    proc: str = "fused-sr"
    out_h: int = 2160                   # cikis (sunucu dokusu) boyutu
    out_w: int = 3840
    tv: bool = False                    # TV modu: 1080p, kaynak hizi (50), ara kare/SR yok
    repair: str = ""                    # TV modunda onarim agi (weights/repair/<ad>.pth), "" = ham
    ai: str = ""                        # TV modunda AI yeniden cizim agi (weights/ai/<ad>.pth), "" = yok
    ai_gain: float = 1.0                # AI farkinin carpani (agresiflik)
    ai_sat: float = 1.0                 # AI: renk doygunlugu
    ai_con: float = 1.0                 # AI: kontrast
    ai_blend: float = 0.6               # AI: durgun bolge harmani gucu (titreme bastirma), 0 = kapali
    probe_sharp: float = 0.0            # >0: saniyede bu kadar cikis/ham keskinlik olcumu (sadece sayi)
    info: bool = False
    log_root: str = os.path.join(ROOT, "runs")
    run_name: str = ""
    stats_every: float = 10.0
    stall_restart_s: float = 6.0
    av_measure: bool = False
    dump_timing: bool = False
    snap: bool = True
    bring_front: bool = True            # kaynak bulununca one getir (ustu kapali Chrome cizmez)
    score: str = ""                     # simule klip meta json'u (tools/tod_sim.py): canli puan
    score_shift: int = 0                # puan kaydirma testi (GT kare)


# --- kayit ----------------------------------------------------------------------

class RunLog:
    def __init__(self, cfg: WatchConfig) -> None:
        name = cfg.run_name or time.strftime("watch_%Y%m%d_%H%M%S")
        self.dir = os.path.join(cfg.log_root, name)
        os.makedirs(self.dir, exist_ok=True)
        self._ev = open(os.path.join(self.dir, "events.jsonl"), "a", encoding="utf-8")
        self._stats_f = None
        self._stats_w = None
        self.counts: Counter[str] = Counter()
        self.t0 = time.perf_counter()
        self.lock = threading.Lock()

    def event(self, kind: str, **data) -> None:
        with self.lock:
            self.counts[kind] += 1
            rec = {"t": round(time.perf_counter() - self.t0, 3), "unix": round(time.time(), 3),
                   "saat": time.strftime("%H:%M:%S"), "olay": kind, **data}
            self._ev.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._ev.flush()
        print(f"[olay] {kind} {data if data else ''}", flush=True)

    def stats(self, row: dict) -> None:
        if self._stats_w is None:
            self._stats_f = open(os.path.join(self.dir, "stats.csv"), "w", newline="", encoding="utf-8")
            self._stats_w = csv.DictWriter(self._stats_f, fieldnames=list(row))
            self._stats_w.writeheader()
        self._stats_w.writerow(row)
        self._stats_f.flush()

    def write_json(self, name: str, data) -> None:
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def close(self) -> None:
        self._ev.close()
        if self._stats_f:
            self._stats_f.close()


class ManagerThread(threading.Thread):
    """Kaynak ve ses yonetimi ayri is parcaciginda: pencere arama, WGC/ProcTap baslatma saniyeler
    surebiliyor, cikis dongusunu bloke etmesin (ilk denemede 7 sn'lik 444 gec tik)."""

    def __init__(self, source, audio, log, every: float = 0.25) -> None:
        super().__init__(daemon=True)
        self.source, self.audio, self.log, self.every = source, audio, log, every
        self.presenter = None
        self._stop = threading.Event()

    def run(self) -> None:
        flag = os.path.join(self.log.dir, "reopen.flag")
        while not self._stop.is_set():
            now = time.perf_counter()
            try:
                if self.presenter is not None and os.path.exists(flag):
                    # Test kancasi: ekran kablosu cikip takilmis gibi sunucuyu yeniden actir
                    os.remove(flag)
                    self.log.event("test_ekran_degisimi_benzetimi")
                    self.presenter.force_reopen = True
                self.source.tick(now)
                if self.audio is not None:
                    self.audio.tick(self.source.hwnd, now)
            except Exception as e:
                self.log.event("yonetici_hatasi", hata=repr(e)[:200])
            self._stop.wait(self.every)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=5)


class GpuMonitor(threading.Thread):
    """nvidia-smi ile sicaklik, guc, bellek, kullanim (10 sn'de bir, ayri is parcacigi)."""

    def __init__(self, every: float = 10.0) -> None:
        super().__init__(daemon=True)
        self.every = every
        self.last: dict = {}
        self.first_used_mb: float | None = None
        self._stop = threading.Event()

    def run(self) -> None:
        q = "temperature.gpu,power.draw,memory.used,utilization.gpu,clocks_throttle_reasons.active"
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
                t, p, m, u, thr = [x.strip() for x in out.split(",")[:5]]
                self.last = {"sicaklik_c": float(t), "guc_w": float(p), "vram_smi_mb": float(m),
                             "gpu_kullanim": float(u), "kisitlama": thr}
            except Exception as e:  # nvidia-smi yoksa ya da takildiysa hat devam eder
                self.last = {"hata": str(e)[:80]}
            self._stop.wait(self.every)

    def stop(self) -> None:
        self._stop.set()


# --- kaynak (pencere + yakalama) ---------------------------------------------------

class SourceManager:
    """Pencereyi bulur, WGC yakalamayi baslatir, kapanma/donma/simge durumunda toparlar."""

    MIN_H, MIN_W = 180, 320

    def __init__(self, cfg: WatchConfig, ring: GpuFrameRing, log: RunLog, probe=None) -> None:
        self.cfg, self.ring, self.log, self.probe = cfg, ring, log, probe
        self.hwnd: int | None = None
        self.refocus_hwnd: int | None = None  # one getirmeden sonra klavye odagi (Esc/S/I) sunucuya doner
        self.src = None
        self.state = "pencere araniyor"
        self.last_unique = 0.0
        self.last_delivered_n = 0
        self.last_delivered_t = 0.0
        self.unique = 0
        self._search_t = 0.0
        self._restart_t = 0.0
        self._iconic = False
        self._frozen = False
        self.timing: list[tuple] | None = [] if cfg.dump_timing else None

    def find(self) -> int | None:
        import win32gui
        hits = []

        def cb(h, _):
            if not win32gui.IsWindowVisible(h):
                return
            t = win32gui.GetWindowText(h)
            if self.cfg.title in t and (not self.cfg.title_must or self.cfg.title_must in t):
                hits.append(h)
        win32gui.EnumWindows(cb, None)
        return hits[0] if hits else None

    def _on_frame(self, raw: float, bgra: np.ndarray) -> None:
        if bgra.shape[0] < self.MIN_H or bgra.shape[1] < self.MIN_W:
            return  # simge durumu gecisinde gelen kucuk kareler tamponu yeniden ayirmasin
        meta = None
        if self.cfg.score:
            from .score import read_barcode
            meta = read_barcode(bgra)
        stamp = self.ring.push(raw, bgra, meta)
        self.unique += 1
        self.last_unique = time.perf_counter()
        if self.timing is not None:
            self.timing.append((raw, -1, stamp.index, stamp.t, int(bool(stamp.repeat))))
        if self.probe is not None:
            self.probe.on_capture_frame(raw, bgra, stamp.t)

    def _start(self, hwnd: int) -> None:
        from .capture import WgcSource
        if self.cfg.bring_front:
            # Chrome, baska pencerenin (orn. Gezgin) tamamen altinda kalinca cizmeyi birakir.
            # Sunucu penceresi 1 px kisa oldugu icin kaynak onde olunca yakalama 50 FPS kalir.
            try:
                winutil.bring_front(hwnd)
                if self.refocus_hwnd:
                    time.sleep(0.2)
                    winutil.bring_front(self.refocus_hwnd)  # sunucu en ustte kalir, odak ona
            except Exception as e:
                self.log.event("one_getirme_hatasi", hata=repr(e)[:120])
        self.hwnd = hwnd
        self.src = WgcSource(hwnd, self._on_frame, 1).start()
        now = time.perf_counter()
        self.last_delivered_t = now
        self.last_delivered_n = 0
        self.last_unique = now
        self.state = "yakalaniyor"

    def _stop(self) -> None:
        if self.src is not None:
            try:
                self.src.stop()
            except Exception as e:
                self.log.event("yakalama_durdurma_hatasi", hata=repr(e))
            self.src = None

    def tick(self, now: float) -> None:
        import win32gui
        try:
            if self.src is None:
                if now - self._search_t < 1.0:
                    return
                self._search_t = now
                hwnd = self.find()
                if hwnd is None:
                    self.state = "pencere araniyor"
                    return
                self._start(hwnd)
                self.log.event("kaynak_baslatildi", baslik=win32gui.GetWindowText(hwnd), hwnd=hwnd)
                return
            src = self.src
            if src.error is not None:
                self.log.event("yakalama_hatasi", hata=repr(src.error))
                self._stop()
                return
            if not win32gui.IsWindow(self.hwnd):
                self.log.event("pencere_kapandi")
                self._stop()
                self.hwnd = None
                return
            iconic = bool(win32gui.IsIconic(self.hwnd))
            if iconic != self._iconic:
                self._iconic = iconic
                self.log.event("pencere_simge_durumu" if iconic else "pencere_geri_acildi")
            if iconic:
                self.state = "simge durumunda (son kare)"
                return
            if src.delivered != self.last_delivered_n:
                self.last_delivered_n = src.delivered
                self.last_delivered_t = now
            frozen = now - self.last_unique > 1.0
            if frozen != self._frozen:
                self._frozen = frozen
                if frozen:
                    self.log.event("yayin_dondu", son_kare_sn_once=round(now - self.last_unique, 2))
                else:
                    self.log.event("yayin_devam")
            self.state = "donma (son kare)" if frozen else "yakalaniyor"
            if now - self.last_delivered_t > self.cfg.stall_restart_s and now - self._restart_t > self.cfg.stall_restart_s:
                # WGC hic kare vermiyor: oturum olmus olabilir (Chrome GPU sureci yeniden basladi vb.)
                self._restart_t = now
                self.log.event("yakalama_yeniden_baslatildi", sessiz_sn=round(now - self.last_delivered_t, 1))
                hwnd = self.hwnd
                self._stop()
                if win32gui.IsWindow(hwnd):
                    self._start(hwnd)
        except Exception as e:
            self.log.event("kaynak_yonetici_hatasi", hata=repr(e), iz=traceback.format_exc()[-400:])
            self._stop()

    def close(self) -> None:
        self._stop()


# --- ses -----------------------------------------------------------------------------

def _wasapi_output(name: str = "") -> int | None:
    import sounddevice as sd
    for api in sd.query_hostapis():
        if "WASAPI" not in api["name"]:
            continue
        if name:
            for i in api["devices"]:
                d = sd.query_devices(i)
                if d["max_output_channels"] > 0 and name.lower() in d["name"].lower():
                    return i
        return api["default_output_device"]
    return None


class AudioManager:
    """Kaynak pencerenin surec agacinin sesi -> gecikmeli calma. Duserse yeniden kurar."""

    def __init__(self, cfg: WatchConfig, log: RunLog, probe=None) -> None:
        from .audio import AudioRing, CaptureClock
        self.cfg, self.log, self.probe = cfg, log, probe
        self.ring = AudioRing(seconds=max(6.0, cfg.delay + 3.0))
        self.clock = CaptureClock()
        self.cap = None
        self.player = None
        self.pid: int | None = None
        self._retry_t = 0.0
        self.state = "kapali"

    def tick(self, hwnd: int | None, now: float) -> None:
        from .audio import AudioRing, CaptureClock, ChromeAudio, SyncedPlayer, root_pid, window_pid
        if now - self._retry_t < 2.0:
            return
        try:
            if self.player is None or not self.player._stream or not self.player._stream.active:
                if self.player is not None:
                    self.log.event("ses_cikisi_dustu_yeniden_kuruluyor")
                    self.player.stop()
                dev = _wasapi_output(self.cfg.audio_device)
                self.player = SyncedPlayer(self.ring, self.clock, device=dev).start()
                if self.probe is not None:
                    self.player.on_block = self.probe.on_output_block
                self.log.event("ses_cikisi_acildi", aygit=dev, gecikme_ms=round(self.player.output_latency_ms(), 1))
            if hwnd is None:
                self.state = "kaynak yok"
                return
            pid = root_pid(window_pid(hwnd))
            dead = self.cap is not None and not self.cap.alive(5.0)
            if pid != self.pid or self.cap is None or dead:
                if self.cap is not None:
                    self.log.event("ses_yakalama_yeniden", eski_pid=self.pid, yeni_pid=pid, sessiz=dead)
                    self.cap.stop()
                # Yeni surec: ornek sayaci sifirdan baslar, saat ve halka da yeni
                self.ring = AudioRing(seconds=max(6.0, self.cfg.delay + 3.0))
                self.clock = CaptureClock()
                self.player.clock, self.player.ring = self.clock, self.ring
                self.player._cursor = None
                self.cap = ChromeAudio(pid, self.ring, self.clock)
                if self.probe is not None:
                    self.cap.on_chunk = lambda end, x, c=self.cap: self.probe.on_capture_chunk(end, x, c.clock)
                self.cap.start()
                self.pid = pid
                self.log.event("ses_yakalama_basladi", pid=pid)
            self.state = "calıyor" if self.player.delay_s is not None else "bekliyor"
        except Exception as e:
            self._retry_t = now
            self.state = "hata"
            self.log.event("ses_hatasi", hata=repr(e)[:200])

    def set_delay(self, d: float | None) -> None:
        if self.player is not None:
            self.player.delay_s = d

    def close(self) -> None:
        if self.cap is not None:
            self.cap.stop()
        if self.player is not None:
            self.player.stop()


# --- A/V olcum sondasi (flas + bip test klibi) ---------------------------------------------

class AvProbe:
    """Girdide ve cikista flas/bip baslangic zamanlarini toplar (sadece zaman sayilari).

    Ham zamanlar ve saat durumlari `dump()` ile runs/<ad>/av_ham.json'a yazilir: cikistaki A/V
    kaymasinin kaynaktan mi (girdi A/V), goruntu yolundan mi, ses yolundan mi geldigi ayrilir.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.flash_in: list[float] = []
        self.flash_out: list[float] = []
        self.beep_in: list[float] = []
        self.beep_out: list[float] = []
        self.flash_in_stamp: list[float] = []   # flas karesinin zaman cizgisi damgasi
        self.flash_out_info: list[tuple] = []    # (t_end, tik T, secilen damga)
        self.clock_log: list[tuple] = []         # (gelis, c0, fs) ~1 sn'de bir
        self.lag_log: list[tuple] = []           # (T, lag_ema, ses gecikmesi)
        self._fin = self._fout = False
        self._loud_in_t = -1e9
        self._loud_out_t = -1e9
        self._clock_t = -1e9
        self._lag_t = -1e9

    def on_capture_frame(self, raw: float, bgra: np.ndarray, stamp_t: float | None = None) -> None:
        m = float(bgra[::64, ::64, 1].mean())
        if not self._fin and m > 220:
            self._fin = True
            with self.lock:
                self.flash_in.append(raw)
                self.flash_in_stamp.append(raw if stamp_t is None else stamp_t)
        elif self._fin and m < 180:
            self._fin = False

    def on_output_frame(self, y: torch.Tensor, t_end: float, tick: float = 0.0, stamp_t: float = 0.0) -> None:
        m = float(y[0, 1, ::54, ::96].float().mean())
        if not self._fout and m > 200:
            self._fout = True
            with self.lock:
                self.flash_out.append(t_end)
                self.flash_out_info.append((t_end, tick, stamp_t))
        elif self._fout and m < 170:
            self._fout = False

    def on_lag(self, tick: float, lag_ema: float, audio_delay: float | None) -> None:
        if tick - self._lag_t >= 1.0:
            self._lag_t = tick
            self.lag_log.append((tick, lag_ema, audio_delay))

    @staticmethod
    def _onset(x: np.ndarray, thr: float = 0.2) -> int | None:
        idx = np.flatnonzero(np.abs(x[:, 0]) > thr)
        return int(idx[0]) if idx.size else None

    def on_capture_chunk(self, end: int, x: np.ndarray, clock) -> None:
        now = time.perf_counter()
        if now - self._clock_t >= 1.0 and clock._c0 is not None:
            self._clock_t = now
            self.clock_log.append((now, end, clock._c0, clock.fs))
        i = self._onset(x)
        if i is None:
            return
        t = clock.time_of(end - len(x) + i)
        if t is None:
            return
        if t - self._loud_in_t > 0.5:
            with self.lock:
                self.beep_in.append(t)
        self._loud_in_t = t

    def on_output_block(self, dac: float, out: np.ndarray) -> None:
        from .audio import FS
        i = self._onset(out)
        if i is None:
            return
        t = dac + i / FS
        if t - self._loud_out_t > 0.5:
            with self.lock:
                self.beep_out.append(t)
        self._loud_out_t = t

    @staticmethod
    def pair(a: list[float], b: list[float], window: float = 0.5,
             expected: float = 0.0) -> list[tuple[float, float]]:
        """Her a icin a + expected'e en yakin b (|sapma| < window): (a, b - a).

        Olaylar periyodik (2 sn) oldugu icin gecikmeli eslestirmede expected verilmeli."""
        if not a or not b:
            return []
        bb = np.asarray(b)
        out = []
        for t in a:
            j = int(np.argmin(np.abs(bb - t - expected)))
            if abs(bb[j] - t - expected) < window:
                out.append((t, float(bb[j] - t)))
        return out

    @staticmethod
    def _slope_ms_h(pairs: list[tuple[float, float]]) -> float | None:
        if len(pairs) < 3 or pairs[-1][0] - pairs[0][0] < 60:
            return None
        ts = np.array([t for t, _ in pairs])
        ds = np.array([d for _, d in pairs]) * 1000
        return round(float(np.polyfit(ts - ts[0], ds, 1)[0] * 3600), 1)

    def summary(self, expected_delay: float = 1.5) -> dict:
        with self.lock:
            fin, fout, bin_, bout = list(self.flash_in), list(self.flash_out), list(self.beep_in), list(self.beep_out)
        d_in = self.pair(fin, bin_)
        d_out = self.pair(fout, bout)
        res = {"flas_giris": len(fin), "flas_cikis": len(fout), "bip_giris": len(bin_), "bip_cikis": len(bout),
               "eslesen_giris": len(d_in), "eslesen_cikis": len(d_out)}
        if len(d_in) >= 3 and len(d_out) >= 3:
            din = np.array([d for _, d in d_in]) * 1000
            dout = np.array([d for _, d in d_out]) * 1000
            res["kaynak_av_ms_medyan"] = round(float(np.median(din)), 1)
            res["cikis_av_ms_medyan_p5_p95"] = [round(float(np.percentile(dout, q)), 1) for q in (50, 5, 95)]
            res["hattin_ekledigi_av_ms_medyan"] = round(float(np.median(dout) - np.median(din)), 1)
            sl = self._slope_ms_h(d_out)
            if sl is not None:
                res["av_kayma_ms_saatte"] = sl
                res["kaynak_av_kayma_ms_saatte"] = self._slope_ms_h(d_in)
                n = len(dout) // 6 or 1
                res["av_ms_ilk_son_bolum_medyan"] = [round(float(np.median(dout[:n])), 1),
                                                     round(float(np.median(dout[-n:])), 1)]
            vd = self.pair(fin, fout, window=0.5, expected=expected_delay)
            ad = self.pair(bin_, bout, window=0.5, expected=expected_delay)
            # Olay basina hattin ekledigi A/V: (ses cikis - ses giris) - (goruntu cikis - goruntu giris).
            # Kaynagin kendi A/V farkindan (ffplay dongu basamaklari) bagimsizdir; asil olcut budur.
            vmap = dict(vd)
            a_t = np.array([t for t, _ in ad])
            added = []
            for f, b in d_in:
                if f not in vmap or not len(a_t):
                    continue
                j = int(np.argmin(np.abs(a_t - (f + b))))
                if abs(a_t[j] - (f + b)) < 1e-6:
                    added.append((f, ad[j][1] - vmap[f]))
            if len(added) >= 3:
                arr = np.array([x for _, x in added]) * 1000
                res["hattin_ekledigi_av_ms_p5_p50_p95"] = [round(float(np.percentile(arr, q)), 1) for q in (5, 50, 95)]
                res["hattin_ekledigi_kayma_ms_saatte"] = self._slope_ms_h(added)
                n = len(arr) // 6 or 1
                res["hattin_ekledigi_ilk_son_bolum_ms"] = [round(float(np.median(arr[:n])), 1),
                                                           round(float(np.median(arr[-n:])), 1)]
            if vd and ad:
                res["goruntu_gecikmesi_ms_medyan"] = round(float(np.median([d for _, d in vd])) * 1000, 1)
                res["ses_gecikmesi_ms_medyan"] = round(float(np.median([d for _, d in ad])) * 1000, 1)
                res["goruntu_gecikmesi_kayma_ms_saatte"] = self._slope_ms_h(vd)
                res["ses_gecikmesi_kayma_ms_saatte"] = self._slope_ms_h(ad)
        return res

    def dump(self) -> dict:
        with self.lock:
            return {"flash_in": self.flash_in, "flash_in_stamp": self.flash_in_stamp, "beep_in": self.beep_in,
                    "flash_out": self.flash_out, "flash_out_info": self.flash_out_info, "beep_out": self.beep_out,
                    "clock_log": self.clock_log, "lag_log": self.lag_log}


# --- ana dongu ------------------------------------------------------------------------

def _pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def _rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 2**20


class SharpProbe(threading.Thread):
    """Canli keskinlik olcumu (goal 2026-09-20): cikis karesi ve ayni anin ham karesi, saniyede
    `rate` kez CPU'ya kopyalanir, ayri is parcaciginda upscaler/sharpness.measure ile olculur.
    Kareler bellekte kalir ve atilir; diske SADECE sayi yazilir (TOD kurali)."""

    def __init__(self, rate: float) -> None:
        super().__init__(daemon=True)
        self.period = 1.0 / rate
        self.next_t = 0.0
        self.q: queue.Queue = queue.Queue(maxsize=2)
        self.rows: list[dict] = []
        self.dropped = 0

    def offer(self, y: torch.Tensor, raw: torch.Tensor | None, now: float) -> None:
        if now < self.next_t:
            return
        self.next_t = now + self.period
        a = y[0].permute(1, 2, 0).cpu().numpy()
        b = raw.permute(1, 2, 0).cpu().numpy() if raw is not None else None
        try:
            self.q.put_nowait((a, b))
        except queue.Full:
            self.dropped += 1

    def run(self) -> None:
        from .sharpness import luma, measure
        while True:
            item = self.q.get()
            if item is None:
                return
            a, b = item
            r = {f"cikis|{k}": v for k, v in measure(luma(a, bgr=True)).items()}
            if b is not None:
                r.update({f"ham|{k}": v for k, v in measure(luma(b, bgr=True)).items()})
            self.rows.append(r)

    def summary(self) -> dict:
        from .sharpness import summarize
        self.q.put(None)
        self.join(timeout=5)
        out = {"olcum": len(self.rows), "atlanan": self.dropped}
        for side in ("cikis", "ham"):
            rows = [{k.split("|", 1)[1]: v for k, v in r.items() if k.startswith(side + "|")} for r in self.rows]
            rows = [r for r in rows if r]
            if rows:
                out[side] = summarize(rows)
        return out


class Watcher:
    def __init__(self, cfg: WatchConfig) -> None:
        self.cfg = cfg
        self.log = RunLog(cfg)

    def _make_proc(self):
        from .models.sr import resolve_sr
        sr = resolve_sr(self.cfg.sr)
        if sr != self.cfg.sr:
            self.log.event("sr_yedek", istenen=self.cfg.sr, kullanilan=sr)
        proc = make_processor(self.cfg.proc, self.cfg.out_h, self.cfg.out_w, sr=sr, split=self.cfg.split,
                              repair=self.cfg.repair, ai=self.cfg.ai, ai_blend=self.cfg.ai_blend,
                              ai_gain=self.cfg.ai_gain, ai_sat=self.cfg.ai_sat, ai_con=self.cfg.ai_con)
        canv = getattr(proc, "canvases", None) or [(1080, 1920)]
        for h, w in canv:
            z = torch.zeros((h, w, 4), dtype=torch.uint8, device="cuda")
            for a in (0.0, 0.5, 0.0):
                proc(z, z, a)
        torch.cuda.synchronize()
        return proc

    def run(self) -> dict:
        import psutil
        cfg, log = self.cfg, self.log
        t_boot = psutil.Process().create_time()
        winutil.dpi_aware()
        winutil.fine_timer()
        log.event("baslangic", ayarlar=asdict(cfg))
        from .present import GlPresenter, pacing_summary
        presenter = GlPresenter(src_w=cfg.out_w, src_h=cfg.out_h, monitor=cfg.monitor, out_fps=cfg.out_fps,
                                vsync=cfg.vsync)
        presenter.show_info = True
        presenter.set_info(["upscaler: hazirlaniyor..."])
        presenter.redraw(["upscaler: hazirlaniyor..."])
        presenter.show_info = cfg.info
        log.event("sunucu", gl=presenter.renderer, ekran=asdict(presenter.monitor), mod=presenter.mode,
                  interval=presenter.interval, pencere=getattr(presenter, "fb_size", None))
        # Yakalama ve ses islemci hazirlanirken baslar: tampon dolar, saat kilitlenir, gecikme
        # suresi o arada gecer; ilk kare hazirlik biter bitmez cikar.
        ring = GpuFrameRing(math.ceil((cfg.delay + 1.0) * 60))
        probe = AvProbe() if cfg.av_measure else None
        sharp = SharpProbe(cfg.probe_sharp) if cfg.probe_sharp > 0 else None
        if sharp is not None:
            sharp.start()
        scorer = None
        if cfg.score:
            from .score import GtBank, LiveScorer
            bank = GtBank(cfg.score, shift=cfg.score_shift).load()
            scorer = LiveScorer(bank)
            log.event("gt_yuklendi", kare=len(bank.frames), sn=round(bank.load_s, 1),
                      vram_gb=round(torch.cuda.memory_allocated() / 2**30, 2))
        source = SourceManager(cfg, ring, log, probe)
        source.refocus_hwnd = getattr(presenter, "hwnd", None)
        audio = AudioManager(cfg, log, probe) if cfg.audio else None
        mgr = ManagerThread(source, audio, log)
        mgr.presenter = presenter
        mgr.start()
        gpu = GpuMonitor(cfg.stats_every)
        gpu.start()
        proc = self._make_proc()
        log.event("islemci_hazir", ad=proc.name, tuvaller=getattr(proc, "canvases", None),
                  hazirlik_sn=round(time.time() - t_boot, 2))

        out_period = 1.0 / cfg.out_fps
        last_lock_T, free_logged = None, False
        t0 = time.perf_counter()
        k = 0
        first_frame_logged = False
        lag_ema: float | None = None
        tot = Counter()
        proc_ms: deque[float] = deque(maxlen=3000)
        e2e_ms: deque[float] = deque(maxlen=3000)
        pres_ms: deque[float] = deque(maxlen=3000)
        # Uzun koşu (100 dk = 360 bin kare): Python float listesi ~32 B/ornek, array 8 B.
        all_proc, all_e2e, all_pres = array("d"), array("d"), array("d")
        end_times, tick_times = array("d"), array("d")
        reopens = presenter.reopens
        ring_gen = 0
        win = Counter()  # istatistik penceresi
        win_t = t0
        info_t = 0.0
        mgr_t = 0.0
        errors: deque[float] = deque(maxlen=50)
        vram_samples: list[tuple[float, float, float]] = []
        tick_log: list[tuple] | None = [] if cfg.dump_timing else None
        pending_late = 0
        route = ""
        info_lines = ["upscaler"]
        stop_reason = "sure doldu"
        try:
            while True:
                now = time.perf_counter()
                if cfg.seconds and now - t0 > cfg.seconds:
                    break
                try:
                    presenter.poll()
                except Exception as e:
                    log.event("sunucu_hatasi_yeniden_aciliyor", hata=repr(e)[:200])
                    try:
                        presenter.reopen()
                    except Exception as e2:  # ekran yok (kablo cikti, laptop kapagi): kisa bekle, tekrar dene
                        log.event("sunucu_acilamadi", hata=repr(e2)[:200])
                        time.sleep(0.5)
                        continue
                if presenter.quit:
                    stop_reason = "Esc"
                    break
                if presenter.toggle_split:
                    presenter.toggle_split = False
                    if hasattr(proc, "split"):
                        proc.split = not proc.split
                        log.event("split", acik=proc.split)
                if presenter.reopens != reopens:
                    reopens = presenter.reopens
                    source.refocus_hwnd = presenter.hwnd
                    log.event("sunucu_yeniden_acildi", ekran=asdict(presenter.monitor), mod=presenter.mode)

                # --- tik zamani ---
                if presenter.mode == "lock":
                    T = presenter.vclock.next_vsync(time.perf_counter())
                    # Swap bloklamiyorsa (pencere odak disi/ortulu, DWM baska ekranin hizinda) dongu
                    # serbest kalip 142 FPS basiyordu (2026-09-19 TOD). Tik araligini cikis periyoduna bagla.
                    if last_lock_T is not None and T - last_lock_T < 0.75 * out_period:
                        if not free_logged:
                            log.event("vsync_bloklamiyor", aralik_ms=round((T - last_lock_T) * 1000, 2))
                            free_logged = True
                        T = last_lock_T + out_period
                        wait = T - time.perf_counter() - 0.002
                        if wait > 0:
                            time.sleep(wait)
                    last_lock_T = T
                    if scorer is not None:
                        scorer.idle(T - time.perf_counter() - 0.004, T - t0)
                else:
                    T = t0 + k * out_period
                    now = time.perf_counter()
                    if now < T:
                        if scorer is not None:
                            scorer.idle(T - now, T - t0)
                            now = time.perf_counter()
                        if T - now > 0.002:
                            time.sleep(T - now - 0.0015)
                        while time.perf_counter() < T:
                            pass
                    now = time.perf_counter()
                    if now - T > out_period:
                        missed = int((now - T) / out_period)
                        if first_frame_logged:
                            tot["gec_tik"] += missed
                            win["gec_tik"] += missed
                            pending_late += missed
                        else:
                            tot["baslangic_tik"] += missed  # hazirlik sirasinda gecen tikler
                        k += missed
                        continue

                t_a = time.perf_counter()
                phase = None
                if scorer is not None:
                    c = ring.meta_offset(T - cfg.delay)
                    if c is not None:
                        # icerik konumu (sira + c) * gt/sim oraninin tam sayi olacagi faz (50->60: -c mod 5)
                        phase = (-c) % 5
                p, s = schedule.select(ring, T - cfg.delay, cfg.out_fps, cfg.snap, phase)
                if p is None:
                    tot["bekleme_tik"] += 1
                    win["bekleme_tik"] += 1
                    try:
                        presenter.redraw(info_lines)
                    except Exception as e:
                        log.event("cizim_hatasi", hata=repr(e)[:200])
                    k += 1
                    continue
                try:
                    fa, fb = p.frames()
                    y = proc(fa, fb, p.alpha)
                    torch.cuda.synchronize()
                    t_b = time.perf_counter()
                    r = presenter.present(y, info_lines)
                    if probe is not None:
                        probe.on_output_frame(y, r["t_end"], T, p.a.stamp.t)
                    if sharp is not None and first_frame_logged:
                        sharp.offer(y, getattr(proc, "last_raw", None), r["t_end"])
                    if scorer is not None:
                        scorer.offer(y, p.a.meta, p.b.meta, p.alpha, T - t0,
                                     T + out_period - time.perf_counter())
                except Exception as e:
                    errors.append(time.perf_counter())
                    tot["islem_hatasi"] += 1
                    log.event("islem_hatasi", hata=repr(e)[:300], iz=traceback.format_exc()[-600:])
                    if len(errors) >= 10 and errors[-1] - errors[-10] < 10:
                        log.event("islemci_yeniden_kuruluyor")
                        errors.clear()
                        del proc
                        torch.cuda.empty_cache()
                        proc = self._make_proc()
                    k += 1
                    continue
                finally:
                    p.release()
                    if p.gen != ring_gen:
                        # Tampon yeniden ayrildi; eski slotlar bu secim bitene kadar canliydi. Simdi
                        # serbest: onbellekte kalirsa her boyut degisiminde VRAM ~1 GB buyuyordu.
                        ring_gen = p.gen
                        fa = fb = p.fa = p.fb = None
                        torch.cuda.empty_cache()
                t_end = r["t_end"]
                pm = (t_b - t_a) * 1000
                em = (t_end - t_a) * 1000
                prm = r["upload_ms"] + r["draw_ms"] + r["swap_ms"]
                proc_ms.append(pm); e2e_ms.append(em); pres_ms.append(prm)
                all_proc.append(pm); all_e2e.append(em); all_pres.append(prm)
                end_times.append(t_end); tick_times.append(T)
                lag = t_end - T
                lag_ema = lag if lag_ema is None else lag_ema + 0.02 * (lag - lag_ema)
                if audio is not None:
                    audio.set_delay(cfg.delay + lag_ema + cfg.av_offset_ms / 1000)
                if probe is not None:
                    probe.on_lag(T, lag_ema, cfg.delay + lag_ema + cfg.av_offset_ms / 1000)
                tot["cikis"] += 1
                win["cikis"] += 1
                if p.hold:
                    tot["tutma"] += 1
                    win["tutma"] += 1
                if schedule.is_real(p.alpha):
                    tot["gercek_kare_tik"] += 1
                if tick_log is not None:
                    tick_log.append((k, T - t0, (t_a - T) * 1000, s, p.alpha, p.a.stamp.index, p.b.stamp.index,
                                     p.a.stamp.t, p.b.stamp.t, int(p.hold), pm, pending_late, len(ring)))
                pending_late = 0
                if not first_frame_logged:
                    first_frame_logged = True
                    log.event("ilk_kare", baslatmadan_sn=round(time.time() - t_boot, 2))
                k += 1
                route = getattr(proc, "route", proc.name)

                # --- bilgi katmani ---
                if presenter.show_info and t_end - info_t > 0.5:
                    info_t = t_end
                    dt = max(t_end - win_t, 1e-3)
                    g = gpu.last
                    info_lines = [
                        f"cikis {win['cikis'] / dt:5.1f} fps | giris {source.unique / max(t_end - t0, 1e-3):4.1f} fps (ort) | "
                        f"gec tik {win['gec_tik']} | bekleme {win['bekleme_tik']}",
                        f"gecikme {cfg.delay * 1000 + (lag_ema or 0) * 1000:.0f} ms | islem p95 {_pct(list(proc_ms)[-300:], 95):.1f} ms | "
                        f"sunum p95 {_pct(list(pres_ms)[-300:], 95):.2f} ms | uctan uca p95 {_pct(list(e2e_ms)[-300:], 95):.1f} ms",
                        f"VRAM {torch.cuda.memory_reserved() / 2**30:.2f} GB (smi {g.get('vram_smi_mb', 0) / 1024:.2f}) | "
                        f"{g.get('sicaklik_c', '?')} C {g.get('guc_w', '?')} W",
                        f"kaynak: {source.state} | {route} | split {'acik' if getattr(proc, 'split', False) else 'kapali'}",
                        f"ses: {audio.state if audio else 'kapali'}"
                        + (f" | hata {audio.player.stats.err_ms:+.1f} ms, atlama {audio.player.stats.hard_jumps}"
                           if audio and audio.player else ""),
                        f"ekran {presenter.monitor.w}x{presenter.monitor.h} {presenter.monitor.hz} Hz, mod {presenter.mode} | "
                        f"olaylar {sum(log.counts.values())}",
                    ]
                    presenter.set_info(info_lines)

                # --- istatistik satiri ---
                if t_end - win_t >= cfg.stats_every:
                    dt = t_end - win_t
                    g = gpu.last
                    alloc = torch.cuda.memory_allocated() / 2**20
                    res = torch.cuda.memory_reserved() / 2**20
                    vram_samples.append((t_end - t0, res, g.get("vram_smi_mb", float("nan"))))
                    ast = audio.player.stats if audio and audio.player else None
                    recent = slice(-int(dt * cfg.out_fps) or None, None)
                    pl = pacing_summary(end_times[recent], tick_times[recent], out_period)
                    log.stats({
                        "t_sn": round(t_end - t0, 1), "cikis_fps": round(win["cikis"] / dt, 2),
                        "giris_benzersiz": source.unique, "gec_tik": win["gec_tik"], "bekleme_tik": win["bekleme_tik"],
                        "tutma": win["tutma"],
                        "islem_p50": round(_pct(list(proc_ms)[recent], 50), 2), "islem_p95": round(_pct(list(proc_ms)[recent], 95), 2),
                        "uctan_uca_p95": round(_pct(list(e2e_ms)[recent], 95), 2), "sunum_p95": round(_pct(list(pres_ms)[recent], 95), 3),
                        "gec_sunum": pl.get("gec_sunum", ""), "sunum_gecikme_ema_ms": round((lag_ema or 0) * 1000, 2),
                        "vram_alloc_mb": round(alloc), "vram_reserved_mb": round(res), "vram_smi_mb": g.get("vram_smi_mb", ""),
                        "rss_mb": round(_rss_mb()), "sicaklik_c": g.get("sicaklik_c", ""), "guc_w": g.get("guc_w", ""),
                        "gpu_kullanim": g.get("gpu_kullanim", ""), "kisitlama": g.get("kisitlama", ""),
                        "ses_hata_ms": round(ast.err_ms, 2) if ast else "", "ses_atlama": ast.hard_jumps if ast else "",
                        "ses_eksik": ast.underruns if ast else "", "saat_ms": round((ring.clock.period or 0) * 1000, 3),
                        "yeniden_damgalama": ring.restamps, "tampon_sifirlama": ring.resets,
                        "kaynak": source.state, "yol": route,
                    })
                    win = Counter()
                    win_t = t_end
        finally:
            t_close = time.time()
            log.event("kapanis_basladi", sebep=stop_reason)
            mgr.stop()
            source.close()
            if audio is not None:
                audio.close()
            gpu.stop()
            try:
                presenter.close()
            except Exception:
                pass

        elapsed = time.perf_counter() - t0
        active = (end_times[-1] - end_times[0]) if len(end_times) > 1 else 0
        summary = {
            "sure_sn": round(elapsed, 1),
            "durma_sebebi": stop_reason,
            "cikis_kare": tot["cikis"],
            "cikis_fps_aktif": round((len(end_times) - 1) / active, 3) if active else None,
            "gec_tik": tot["gec_tik"],
            "gec_tik_orani": round(tot["gec_tik"] / max(tot["cikis"] + tot["gec_tik"], 1), 5),
            "bekleme_tik": tot["bekleme_tik"],
            "tutma_tik": tot["tutma"],
            "gercek_kare_tik_orani": round(tot["gercek_kare_tik"] / max(tot["cikis"], 1), 3),
            "islem_hatasi": tot["islem_hatasi"],
            "islem_ms_p50_p95_p99": [round(_pct(all_proc, q), 2) for q in (50, 95, 99)],
            "uctan_uca_ms_p50_p95_p99": [round(_pct(all_e2e, q), 2) for q in (50, 95, 99)],
            "sunum_ms_p50_p95_p99": [round(_pct(all_pres, q), 3) for q in (50, 95, 99)],
            **pacing_summary(end_times, tick_times, out_period),
            "sunum_gecikmesi_ema_ms": round((lag_ema or 0) * 1000, 2),
            "giris_benzersiz_kare": source.unique,
            "saat_periyot_ms": round((ring.clock.period or 0) * 1000, 3),
            "tampon_sifirlama": ring.resets, "yeniden_damgalama": ring.restamps,
            "vram_tepe_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "vram_ornekleri_bas_son_mb": [vram_samples[0], vram_samples[-1]] if vram_samples else None,
            "olaylar": dict(log.counts),
            "ekran": asdict(presenter.monitor), "sunum_modu": presenter.mode, "gl": presenter.renderer,
            "yol": route,
        }
        if audio is not None and audio.player is not None:
            st = audio.player.stats
            errs = [abs(e) for _, e in audio.player.err_log]
            summary["ses"] = {"blok": st.blocks, "sert_atlama": st.hard_jumps, "eksik": st.underruns,
                              "hata_ms_p50_p99_abs": [round(_pct(errs, 50), 2), round(_pct(errs, 99), 2)],
                              "yakalama_fs": round(audio.clock.fs, 2)}
        if probe is not None:
            summary["av"] = probe.summary(expected_delay=cfg.delay)
            log.write_json("av_ham.json", probe.dump())
        if sharp is not None:
            summary["keskinlik"] = sharp.summary()
        if scorer is not None:
            scorer.finish()
            summary["puan"] = scorer.summary()
            log.write_json("puan_kareler.json", scorer.dump())
        if cfg.dump_timing:
            path = os.path.join(log.dir, "timing.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("raw_s,barcode,index,t,repeat\n")
                f.writelines(f"{r:.7f},{c},{i},{t:.7f},{rp}\n" for r, c, i, t, rp in source.timing)
            with open(os.path.join(log.dir, "timing_ticks.csv"), "w", encoding="utf-8") as f:
                f.write("k,T_s,start_late_ms,s,alpha,a_index,b_index,a_t,b_t,hold,proc_ms,late_before,ring_len\n")
                for row in tick_log:
                    kk, Ts, sl, ss, al, ai, bi, at, bt, hd, pm, lb, rl = row
                    f.write(f"{kk},{Ts:.6f},{sl:.3f},{ss:.7f},{al:.5f},{ai},{bi},{at:.7f},{bt:.7f},{hd},{pm:.3f},{lb},{rl}\n")
        summary["kapanis_sn"] = round(time.time() - t_close, 2)
        log.write_json("summary.json", summary)
        log.event("bitti", kapanis_sn=summary["kapanis_sn"])
        log.close()
        return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m upscaler watch", description="TOD canli maci 4K 60 FPS izle")
    d = WatchConfig()
    ap.add_argument("--title", default=d.title, help="pencere basliginda gecen metin")
    ap.add_argument("--title-must", default=d.title_must, help="baslikta ayrica gecmesi gereken metin ('' = yok)")
    ap.add_argument("--tv", action="store_true",
                    help="TV modu: 1080p cikis, 50 FPS, ara kare ve SR yok, harici ekran, vsync kilidi (50 Hz)")
    ap.add_argument("--repair", default="", help="TV modunda sikistirma onarim agi (ör. rep_v0); bos = ham")
    ap.add_argument("--ai", default="", help="TV modunda AI yeniden cizim agi (weights/ai/<ad>.pth); bos = yok")
    ap.add_argument("--ai-guc", type=float, default=d.ai_gain, help="AI farkinin carpani (1 = egitildigi gibi, 2 = cok agresif)")
    ap.add_argument("--ai-renk", type=float, default=d.ai_sat, help="AI: renk doygunlugu (1 = dokunma, 1,2 = canli)")
    ap.add_argument("--ai-kontrast", type=float, default=d.ai_con, help="AI: kontrast (1 = dokunma)")
    ap.add_argument("--ai-blend", type=float, default=d.ai_blend, help="AI durgun bolge harmani (0 = kapali)")
    ap.add_argument("--probe-sharp", type=float, default=0.0, help="saniyede N kez cikis ve ham keskinligi (sadece sayi)")
    ap.add_argument("--delay", type=float, default=d.delay)
    ap.add_argument("--monitor", default=None, help="auto ya da ekran adinin parcasi")
    ap.add_argument("--vsync", default=d.vsync, choices=["auto", "lock", "timer"])
    ap.add_argument("--out-fps", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=0.0, help="bu kadar sonra kapan (0: Esc)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--audio-device", default="", help="WASAPI cikis aygiti adinin parcasi")
    ap.add_argument("--av-offset-ms", type=float, default=0.0, help="ses/goruntu ince ayari (+: ses gec)")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--info", action="store_true", help="bilgi katmani acik baslasin (I)")
    ap.add_argument("--proc", default=None)
    ap.add_argument("--sr", default=d.sr, help="SR modeli (rt4ksr-x2 ya da ince ayarli rt4ksr-x2-<ad>)")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--av-measure", action="store_true", help="flas+bip test klibiyle A/V olcumu")
    ap.add_argument("--score", default="", help="simule klip meta json'u: canli PSNR/SSIM (tools/live_score.py)")
    ap.add_argument("--score-shift", type=int, default=0, help="puan kaydirma testi (GT kare)")
    ap.add_argument("--dump-timing", action="store_true", help="kare ve tik zamanlarini CSV'ye yaz (sadece sayi)")
    a = ap.parse_args(argv)
    if a.tv:
        # TV: kaynak 1080p50, ekran 1080p 50 Hz. Cikis kaynak hizinda, her tik gercek kare.
        mon, fps, oh, ow = a.monitor or "external", a.out_fps or 50.0, 1080, 1920
        proc = a.proc or ("ai" if a.ai else "repair" if a.repair else "pass")
    else:
        mon, fps, proc, oh, ow = a.monitor or d.monitor, a.out_fps or d.out_fps, a.proc or d.proc, d.out_h, d.out_w
    cfg = WatchConfig(title=a.title, title_must=a.title_must, delay=a.delay, monitor=mon, vsync=a.vsync,
                      out_fps=fps, seconds=a.seconds, tv=a.tv, repair=a.repair, ai=a.ai, ai_blend=a.ai_blend, ai_gain=a.ai_guc, ai_sat=a.ai_renk, ai_con=a.ai_kontrast, probe_sharp=a.probe_sharp, out_h=oh, out_w=ow, audio=not a.no_audio, audio_device=a.audio_device,
                      av_offset_ms=a.av_offset_ms, split=a.split, info=a.info, proc=proc, sr=a.sr, run_name=a.run_name,
                      av_measure=a.av_measure, dump_timing=a.dump_timing, score=a.score,
                      score_shift=a.score_shift)
    w = Watcher(cfg)
    summary = w.run()
    print("--- ozet (" + w.log.dir + ")")
    for key, v in summary.items():
        print(f"{key}: {v}")


if __name__ == "__main__":
    main()
