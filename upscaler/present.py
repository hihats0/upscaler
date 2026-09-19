"""GPU'da kalan sunucu (Hat 1.4 paket 2).

4K kare (CUDA tensoru) -> CUDA-OpenGL interop ile GL dokusuna GPU icinde kopya -> tam ekran
pencereye cizim -> SwapBuffers. CPU'ya kare kopyasi yok.

Onemli noktalar (olculdu / dogrulandi, 2026-09-15):
- Laptop Optimus: ekran Intel iGPU'da. GL baglami NVIDIA'da olmali, yoksa CUDA dokuyu kaydedemez.
  SHIM_MCCOMPAT=0x800000001 ortam degiskeni surec BASLARKEN ayarli olmali (watch.bat ayarlar,
  ensure_nvidia_gl() gerekirse sureci bu degiskenle yeniden baslatir).
- Chrome, ustu tamamen kapanan pencerede cizimi birakir (WGC 5 sn'de 1 kare). Tam ekran
  pencereye WS_EX_TOOLWINDOW verilir: Chrome'un ortulme hesabi arac pencerelerini saymaz.
- Ekran 60 Hz ise (4K monitor) swap interval 1 ile vsync kilidi; 144 Hz laptop ekraninda 60'a
  kilit yok, dongu kendi saatiyle surer ve DWM gosterir ("timer" modu).

Kisayollar: Esc cikis, S split ac/kapa, I bilgi katmani.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

NVIDIA_GL_ENV = ("SHIM_MCCOMPAT", "0x800000001")
GL_TEXTURE_2D = 0x0DE1
CUDA_REGISTER_WRITE_DISCARD = 2
CUDA_MEMCPY_D2D = 3
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000


# --- saf yardimcilar (birim testli) -------------------------------------------

@dataclass(frozen=True)
class MonitorInfo:
    name: str
    x: int
    y: int
    w: int
    h: int
    hz: int
    primary: bool = False


def pick_monitor(mons: list[MonitorInfo], prefer: str = "auto") -> int | None:
    """Cikis ekrani secimi: ada gore (prefer), yoksa en az 3840x2160 ve >=59 Hz olan en buyuk, yoksa birincil.

    prefer="external": birincil olmayan ekran (TV modu; HDMI TV ile laptop paneli ayni adi
    "Generic PnP Monitor" tasiyabiliyor, ad ise yaramaz). Birden fazlaysa 50 Hz'e en yakin. Yoksa birincil.
    """
    if not mons:
        return None
    if prefer == "external":
        ext = [i for i, m in enumerate(mons) if not m.primary]
        if ext:
            return min(ext, key=lambda i: abs(mons[i].hz - 50))
        prim = [i for i, m in enumerate(mons) if m.primary]
        return prim[0] if prim else 0
    if prefer not in ("auto", ""):
        for i, m in enumerate(mons):
            if prefer.lower() in m.name.lower():
                return i
    big = [i for i, m in enumerate(mons) if m.w >= 3840 and m.h >= 2160 and m.hz >= 59]
    if big:
        return max(big, key=lambda i: (mons[i].w * mons[i].h, -abs(mons[i].hz - 60)))
    prim = [i for i, m in enumerate(mons) if m.primary]
    return prim[0] if prim else 0


def swap_plan(refresh_hz: float, out_fps: float, mode: str = "auto") -> tuple[str, int]:
    """(mod, swap interval). lock: ekran yenilemesi cikis hizinin tam kati, dongu vsync ile gider."""
    n = max(1, round(refresh_hz / out_fps)) if refresh_hz else 1
    exact = bool(refresh_hz) and abs(refresh_hz / n - out_fps) < 0.6
    if mode == "lock" or (mode == "auto" and exact):
        return ("lock", n) if exact else ("timer", 0)
    return ("timer", 0)


def fit_rect(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int, int, int]:
    """Doku pencereye en-boy korunarak: (x, y, w, h) piksel."""
    s = min(dst_w / src_w, dst_h / src_h)
    w, h = round(src_w * s), round(src_h * s)
    return (dst_w - w) // 2, (dst_h - h) // 2, w, h


class VsyncClock:
    """Swap donus zamanlarindan ekran periyodu ve bir sonraki vsync tahmini (lock modu)."""

    def __init__(self, period: float, window: int = 600) -> None:
        self.nominal = period
        self.period = period
        self.last: float | None = None
        self._n = 0  # toplam vsync sayaci (kacan vsync'ler dahil)
        self._pts: deque[tuple[int, float]] = deque(maxlen=window)

    def update(self, t_end: float) -> None:
        if self.last is not None:
            d = t_end - self.last
            n = round(d / self.period)
            if n < 1 or abs(d - n * self.period) > 0.25 * self.period:
                self._pts.clear()  # takilma ya da pencere olayi: regresyonu yeniden baslat
                n = max(n, 1)
            self._n += n
        self._pts.append((self._n, t_end))
        self.last = t_end
        if len(self._pts) >= 60:
            # En kucuk kareler egimi: tek tek aralik titremesi (DWM, swap donusu) ortalamada kaybolur.
            ns = np.fromiter((p[0] for p in self._pts), float)
            ts = np.fromiter((p[1] for p in self._pts), float)
            ns -= ns.mean()
            slope = float((ns * (ts - ts.mean())).sum() / (ns * ns).sum())
            if abs(slope - self.nominal) < 0.02 * self.nominal:
                self.period = slope

    def next_vsync(self, now: float) -> float:
        if self.last is None:
            return now + self.period
        k = max(1, int((now - self.last) / self.period) + 1)
        return self.last + k * self.period


# --- CUDA-GL interop ------------------------------------------------------------

_CUDART = None


def _cudart():
    global _CUDART
    if _CUDART is None:
        path = os.path.join(os.path.dirname(torch.__file__), "lib", "cudart64_12.dll")
        _CUDART = ctypes.CDLL(path)
        c = _CUDART
        c.cudaGraphicsGLRegisterImage.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
        c.cudaGraphicsMapResources.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        c.cudaGraphicsUnmapResources.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        # Dikkat: ilk arguman cikis (cudaArray_t*), sonra kaynak (cuda_runtime_api.h). Ters sira 400 verir.
        c.cudaGraphicsSubResourceGetMappedArray.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
        c.cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]
        c.cudaMemcpy2DToArray.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
                                         ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int]
    return _CUDART


def _chk(code: int, what: str) -> None:
    if code != 0:
        raise RuntimeError(f"CUDA-GL {what} hatasi: {code} (GL baglami NVIDIA'da degil olabilir)")


class CudaGlTexture:
    """GL_RGBA8 dokuya CUDA tensorunden GPU icinde kopya."""

    def __init__(self, tex_id: int, w: int, h: int) -> None:
        self.rt = _cudart()
        self.w, self.h = w, h
        self.res = ctypes.c_void_p()
        torch.cuda.init()
        _chk(self.rt.cudaGraphicsGLRegisterImage(ctypes.byref(self.res), tex_id, GL_TEXTURE_2D,
                                                CUDA_REGISTER_WRITE_DISCARD), "kayit")

    def upload(self, hwc4: torch.Tensor) -> None:
        assert hwc4.is_contiguous() and hwc4.shape == (self.h, self.w, 4) and hwc4.dtype == torch.uint8
        torch.cuda.current_stream().synchronize()
        _chk(self.rt.cudaGraphicsMapResources(1, ctypes.byref(self.res), None), "map")
        try:
            arr = ctypes.c_void_p()
            _chk(self.rt.cudaGraphicsSubResourceGetMappedArray(ctypes.byref(arr), self.res, 0, 0), "array")
            # width "columns in bytes": RGBA8 dizide 4 * piksel (piksel verilince satirin 1/4'u kopyalandi)
            _chk(self.rt.cudaMemcpy2DToArray(arr, 0, 0, hwc4.data_ptr(), self.w * 4, self.w * 4, self.h,
                                            CUDA_MEMCPY_D2D), "kopya")
        finally:
            self.rt.cudaGraphicsUnmapResources(1, ctypes.byref(self.res), None)

    def close(self) -> None:
        if self.res:
            self.rt.cudaGraphicsUnregisterResource(self.res)
            self.res = ctypes.c_void_p()


def ensure_nvidia_gl() -> None:
    """Optimus: GL baglaminin NVIDIA'da acilmasi icin ortam degiskeni surec basinda gerekir.
    Yoksa ayni komutu degiskenle alt surec olarak calistirir ve onun cikis koduyla cikar."""
    key, val = NVIDIA_GL_ENV
    if os.environ.get(key) == val:
        return
    import subprocess
    env = dict(os.environ, **{key: val})
    # orig_argv "-m paket.modul" bicimini korur (sys.argv dosya yoluna cevirir, goreli import bozulur)
    sys.exit(subprocess.call([sys.executable] + sys.orig_argv[1:], env=env))


# --- sunucu ---------------------------------------------------------------------

_VS = """#version 120
void main() { gl_TexCoord[0] = gl_MultiTexCoord0; gl_Position = gl_Vertex; }"""
_FS = """#version 120
uniform sampler2D tex;
void main() { gl_FragColor = vec4(texture2D(tex, gl_TexCoord[0].st).bgr, 1.0); }"""


@dataclass
class PresentStats:
    # Sinirli: 100 dk'lik koşuda sinirsiz float listeleri dakikada ~0,5 MB buyuyordu (2026-09-16).
    upload_ms: deque[float] = field(default_factory=lambda: deque(maxlen=20000))
    draw_ms: deque[float] = field(default_factory=lambda: deque(maxlen=20000))
    swap_ms: deque[float] = field(default_factory=lambda: deque(maxlen=20000))
    end_times: deque[float] = field(default_factory=lambda: deque(maxlen=20000))


class GlPresenter:
    """Tam ekran, kenarliksiz, en ustte arac penceresi. present() 4K uint8 BGR kareyi gosterir."""

    def __init__(self, src_w: int = 3840, src_h: int = 2160, monitor: str = "auto", out_fps: float = 60.0,
                 vsync: str = "auto", title: str = "upscaler", windowed: tuple[int, int] | None = None) -> None:
        import glfw
        from OpenGL import GL
        self.glfw, self.GL = glfw, GL
        self.src_w, self.src_h = src_w, src_h
        self.prefer, self.out_fps, self.vsync_pref, self.title = monitor, out_fps, vsync, title
        self.windowed = windowed
        self.quit = False
        self.toggle_split = False
        self.show_info = False
        self.monitors_changed = False
        self.force_reopen = False
        self.reopens = 0
        self.hwnd = None
        self.stats = PresentStats()
        self.win = None
        self._tex = self._cuda = self._prog = None
        self._info_tex = None
        self._info_size = (0, 0)
        self._hwc = torch.empty((src_h, src_w, 4), dtype=torch.uint8, device="cuda")
        self._hwc[..., 3] = 255
        if not glfw.init():
            raise RuntimeError("glfw baslatilamadi")
        glfw.set_monitor_callback(self._on_monitor)
        self._open()

    # --- pencere ---
    def monitors(self) -> list[MonitorInfo]:
        glfw = self.glfw
        prim = glfw.get_primary_monitor()
        out = []
        for m in glfw.get_monitors() or []:
            mode = glfw.get_video_mode(m)
            x, y = glfw.get_monitor_pos(m)
            name = glfw.get_monitor_name(m)
            name = name.decode(errors="replace") if isinstance(name, bytes) else str(name)
            out.append(MonitorInfo(name, x, y, mode.size.width, mode.size.height, mode.refresh_rate,
                                   ctypes.cast(m, ctypes.c_void_p).value == ctypes.cast(prim, ctypes.c_void_p).value))
        return out

    def _open(self) -> None:
        glfw, GL = self.glfw, self.GL
        mons = self.monitors()
        i = pick_monitor(mons, self.prefer)
        if i is None:
            raise RuntimeError("ekran yok")
        self.monitor = mons[i]
        m = self.monitor
        glfw.default_window_hints()
        glfw.window_hint(glfw.DECORATED, False)
        glfw.window_hint(glfw.FLOATING, self.windowed is None)
        glfw.window_hint(glfw.AUTO_ICONIFY, False)
        glfw.window_hint(glfw.FOCUS_ON_SHOW, True)
        glfw.window_hint(glfw.VISIBLE, False)
        # Tam ekran pencere 1 px kisa: ekrani birebir kaplayan ve cizen pencere altinda Chrome'un
        # yakalanan kare hizi 50'den 35'e dustu (DWM tam ekran yolu); 1 px kisa pencerede 50,0
        # (olculdu 2026-09-15, arac penceresi ve normal pencere ayni).
        # Birincil olmayan ekranda (TV) Chrome yok: pencere tam boy. 1 px kisa pencere orada DWM
        # kompozisyonuna dusup birincil panelin 144 Hz'inde sunuyordu (TV 50 Hz'e kilitlenmedi,
        # takilma; 2026-09-19 16:24 sonrasi tum TV koşulari 144 FPS). Tam boyda TV'nin vsync'i.
        short = 0 if not m.primary else 1
        w, h = self.windowed or (m.w, m.h - short)
        self.win = glfw.create_window(w, h, self.title, None, None)
        if not self.win:
            raise RuntimeError("pencere acilamadi")
        glfw.set_window_pos(self.win, m.x + (0 if self.windowed is None else 40), m.y + (0 if self.windowed is None else 40))
        if self.windowed is None:
            self._make_tool_window()
        glfw.show_window(self.win)
        # Farkli olcekli ekrana tasininca (panel %125 -> TV %100) Windows pencereyi DPI oraninda
        # kucultuyor (TV'de %80 alan kapladi, 2026-09-19). Boyutu ve konumu yeniden zorla, dogrula.
        for _ in range(3):
            if self.windowed is not None or tuple(glfw.get_framebuffer_size(self.win)) == (w, h):
                break
            glfw.set_window_size(self.win, w, h)
            glfw.set_window_pos(self.win, m.x, m.y)
            glfw.poll_events()
        self.fb_size = tuple(glfw.get_framebuffer_size(self.win))
        glfw.focus_window(self.win)
        glfw.make_context_current(self.win)
        glfw.set_key_callback(self.win, self._on_key)
        self.renderer = GL.glGetString(GL.GL_RENDERER).decode(errors="replace")
        self.mode, self.interval = swap_plan(m.hz, self.out_fps, self.vsync_pref)
        glfw.swap_interval(self.interval)
        self.vclock = VsyncClock(max(self.interval, 1) / m.hz if m.hz else 1 / self.out_fps)

        self._tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, self.src_w, self.src_h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._cuda = CudaGlTexture(int(self._tex), self.src_w, self.src_h)
        from OpenGL.GL import shaders
        self._prog = shaders.compileProgram(shaders.compileShader(_VS, GL.GL_VERTEX_SHADER),
                                            shaders.compileShader(_FS, GL.GL_FRAGMENT_SHADER))
        self._info_tex = GL.glGenTextures(1)
        self._info_size = (0, 0)
        GL.glClearColor(0, 0, 0, 1)

    def _make_tool_window(self) -> None:
        hwnd = self.glfw.get_win32_window(self.win)
        u32 = ctypes.windll.user32
        u32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
        u32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
        ex = u32.GetWindowLongPtrW(ctypes.c_void_p(hwnd), GWL_EXSTYLE)
        u32.SetWindowLongPtrW(ctypes.c_void_p(hwnd), GWL_EXSTYLE, (ex | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW)
        self.hwnd = hwnd

    def _close_window(self) -> None:
        if self.win is None:
            return
        self.glfw.make_context_current(self.win)
        if self._cuda is not None:
            self._cuda.close()
            self._cuda = None
        self.glfw.destroy_window(self.win)
        self.win = None

    def reopen(self) -> None:
        """Ekran listesi degisti (kablo cikti/takildi): pencereyi uygun ekranda yeniden ac."""
        self._close_window()
        self.monitors_changed = False
        self.force_reopen = False
        self._open()
        self.reopens += 1

    def _on_monitor(self, mon, event) -> None:
        self.monitors_changed = True

    def _on_key(self, win, key, scancode, action, mods) -> None:
        glfw = self.glfw
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            self.quit = True
        elif key == glfw.KEY_S:
            self.toggle_split = True
        elif key == glfw.KEY_I:
            self.show_info = not self.show_info

    # --- kare ---
    def poll(self) -> None:
        self.glfw.poll_events()
        if self.glfw.window_should_close(self.win):
            self.quit = True
        if self.force_reopen:
            self.reopen()
            return
        if self.monitors_changed:
            want = self.monitors()
            names = [(m.name, m.x, m.y, m.w, m.h, m.hz) for m in want]
            cur = (self.monitor.name, self.monitor.x, self.monitor.y, self.monitor.w, self.monitor.h, self.monitor.hz)
            i = pick_monitor(want, self.prefer)
            if cur not in names or (i is not None and names[i] != cur):
                self.reopen()
            else:
                self.monitors_changed = False

    def upload(self, bgr_nchw: torch.Tensor) -> float:
        """(1,3,H,W) uint8 BGR -> doku. Suresi ms."""
        t0 = time.perf_counter()
        self._hwc[..., :3].copy_(bgr_nchw[0].permute(1, 2, 0))
        self._cuda.upload(self._hwc)
        return (time.perf_counter() - t0) * 1000

    def draw_and_swap(self, info_lines: list[str] | None = None) -> tuple[float, float, float]:
        """Ciz + swap. (cizim ms, swap ms, swap bitis zamani)."""
        glfw, GL = self.glfw, self.GL
        t0 = time.perf_counter()
        fw, fh = glfw.get_framebuffer_size(self.win)
        GL.glViewport(0, 0, fw, fh)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        x, y, w, h = fit_rect(self.src_w, self.src_h, fw, fh)
        GL.glViewport(x, y, w, h)
        GL.glUseProgram(self._prog)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
        self._quad(-1, -1, 1, 1)
        GL.glUseProgram(0)
        if self.show_info and info_lines:
            GL.glViewport(0, 0, fw, fh)
            self._draw_info(info_lines, fw, fh)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        t1 = time.perf_counter()
        glfw.swap_buffers(self.win)
        if self.mode == "lock":
            GL.glFinish()
        t2 = time.perf_counter()
        self.vclock.update(t2)
        return (t1 - t0) * 1000, (t2 - t1) * 1000, t2

    def present(self, bgr_nchw: torch.Tensor, info_lines: list[str] | None = None) -> dict:
        up = self.upload(bgr_nchw)
        dr, sw, t_end = self.draw_and_swap(info_lines)
        s = self.stats
        s.upload_ms.append(up)
        s.draw_ms.append(dr)
        s.swap_ms.append(sw)
        s.end_times.append(t_end)
        return {"upload_ms": up, "draw_ms": dr, "swap_ms": sw, "t_end": t_end}

    def redraw(self, info_lines: list[str] | None = None) -> float:
        """Yeni kare yokken (donma, bekleme) son dokuyu tekrar ciz: siyah ekran olmaz."""
        return self.draw_and_swap(info_lines)[2]

    def _quad(self, x0, y0, x1, y1) -> None:
        GL = self.GL
        GL.glBegin(GL.GL_TRIANGLE_STRIP)
        GL.glTexCoord2f(0, 1); GL.glVertex2f(x0, y0)
        GL.glTexCoord2f(1, 1); GL.glVertex2f(x1, y0)
        GL.glTexCoord2f(0, 0); GL.glVertex2f(x0, y1)
        GL.glTexCoord2f(1, 0); GL.glVertex2f(x1, y1)
        GL.glEnd()

    def set_info(self, lines: list[str]) -> None:
        """Bilgi katmani metnini dokuya yazar (saniyede birkac kez; kucuk CPU dokusu, kare degil)."""
        import pygame
        GL = self.GL
        if not pygame.font.get_init():
            pygame.font.init()
            self._font = pygame.font.SysFont("consolas", max(16, self.monitor.h // 54))
        font = self._font
        surfs = [font.render(t, True, (255, 255, 255)) for t in lines]
        lh = font.get_linesize()
        w = max(s.get_width() for s in surfs) + 24
        h = lh * len(surfs) + 16
        img = pygame.Surface((w, h), pygame.SRCALPHA)
        img.fill((0, 0, 0, 170))
        for k, s in enumerate(surfs):
            img.blit(s, (12, 8 + k * lh))
        data = pygame.image.tobytes(img, "RGBA", True)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._info_tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._info_size = (w, h)

    def _draw_info(self, lines, fw, fh) -> None:
        GL = self.GL
        w, h = self._info_size
        if not w:
            return
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._info_tex)
        x0, y1 = -1 + 2 * 20 / fw, 1 - 2 * 20 / fh
        x1, y0 = x0 + 2 * w / fw, y1 - 2 * h / fh
        GL.glBegin(GL.GL_TRIANGLE_STRIP)
        GL.glTexCoord2f(0, 0); GL.glVertex2f(x0, y0)
        GL.glTexCoord2f(1, 0); GL.glVertex2f(x1, y0)
        GL.glTexCoord2f(0, 1); GL.glVertex2f(x0, y1)
        GL.glTexCoord2f(1, 1); GL.glVertex2f(x1, y1)
        GL.glEnd()
        GL.glDisable(GL.GL_BLEND)

    def read_texture_rgba(self) -> np.ndarray:
        """Dogrulama icin dokuyu CPU'ya okur (sadece test, canli hatta kullanilmaz)."""
        GL = self.GL
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
        data = GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return np.frombuffer(data, np.uint8).reshape(self.src_h, self.src_w, 4)

    def close(self) -> None:
        self._close_window()
        self.glfw.terminate()


def pacing_summary(end_times: list[float], ticks: list[float], period: float) -> dict:
    """Swap bitis zamanlari ve hedef tik zamanlari: aralik dagilimi, gec sunum orani.

    Gec sunum: swap, hedef tikten yarim cikis periyodundan fazla sonra bitti (o karenin
    ekrandaki yuvasi kacti, onceki kare tekrar gosterildi)."""
    if len(end_times) < 3:
        return {}
    e = np.asarray(end_times)
    d = np.diff(e) * 1000
    lag = (e - np.asarray(ticks[:len(e)])) * 1000
    late = int(np.sum(lag > 0.5 * period * 1000 + np.median(lag)))
    return {
        "sunum_araligi_ms_p1_p50_p99": [round(float(np.percentile(d, q)), 2) for q in (1, 50, 99)],
        "sunum_gecikmesi_ms_p50_p99": [round(float(np.percentile(lag, q)), 2) for q in (50, 99)],
        "gec_sunum": late,
        "gec_sunum_orani": round(late / len(e), 5),
    }


def _bench() -> None:
    """Sentetik 4K karelerle sunum olcumu (TOD yok): python -m upscaler.present --seconds 10"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--monitor", default="auto")
    ap.add_argument("--vsync", default="auto", choices=["auto", "lock", "timer"])
    ap.add_argument("--out-fps", type=float, default=60)
    ap.add_argument("--verify", action="store_true", help="dokuyu geri okuyup icerigi dogrula")
    args = ap.parse_args()
    ensure_nvidia_gl()
    from . import winutil
    winutil.dpi_aware()
    winutil.fine_timer()
    p = GlPresenter(monitor=args.monitor, out_fps=args.out_fps, vsync=args.vsync)
    print(f"GL: {p.renderer} | ekran {p.monitor} | mod {p.mode} interval {p.interval}", flush=True)
    H, W = 2160, 3840
    yy = torch.arange(H, device="cuda").view(H, 1)
    xx = torch.arange(W, device="cuda").view(1, W)
    frame = torch.zeros((1, 3, H, W), dtype=torch.uint8, device="cuda")
    if args.verify:
        frame[0, 0] = (xx % 256).to(torch.uint8)          # B
        frame[0, 1] = (yy % 256).to(torch.uint8)          # G
        frame[0, 2] = ((xx + yy) % 251).to(torch.uint8)   # R
        p.upload(frame)
        tex = p.read_texture_rgba()
        want = frame[0].permute(1, 2, 0).cpu().numpy()
        ok = np.array_equal(tex[..., :3], want) and (tex[..., 3] == 255).all()
        bad = np.argwhere((tex[..., :3] != want).any(-1))
        print(f"doku dogrulamasi: {'ESIT' if ok else 'FARKLI'} (farkli piksel {len(bad)}, ilk {bad[:1].tolist()}, "
              f"alfa 255 degil {int((tex[..., 3] != 255).sum())})")
    period = 1.0 / args.out_fps
    p.show_info = True
    t0 = time.perf_counter()
    k = 0
    ticks, e2e = [], []
    while not p.quit and time.perf_counter() - t0 < args.seconds:
        p.poll()
        if p.mode == "lock":
            T = p.vclock.next_vsync(time.perf_counter())
        else:
            T = t0 + k * period
            now = time.perf_counter()
            if now < T:
                if T - now > 0.002:
                    time.sleep(T - now - 0.0015)
                while time.perf_counter() < T:
                    pass
        ts = time.perf_counter()
        ph = (k * 8) % W
        frame[0, 1].copy_(((xx - ph).abs() < 40).to(torch.uint8).expand(H, W) * 255)  # kayan dikey cubuk
        torch.cuda.synchronize()
        if k % 30 == 0:
            p.set_info([f"sunum testi k={k}", f"GL {p.renderer[:40]}", f"mod {p.mode}"])
        r = p.present(frame, ["x"])
        ticks.append(T)
        e2e.append((r["t_end"] - ts) * 1000)
        k += 1
    st = p.stats
    summ = pacing_summary(st.end_times, ticks, period)
    print({"kare": k,
           "yukleme_ms_p50_p95": [round(float(np.percentile(st.upload_ms, q)), 2) for q in (50, 95)],
           "cizim_ms_p50_p95": [round(float(np.percentile(st.draw_ms, q)), 2) for q in (50, 95)],
           "swap_ms_p50_p95": [round(float(np.percentile(st.swap_ms, q)), 2) for q in (50, 95)],
           "sunum_toplam_ms_p50_p95": [round(float(np.percentile(e2e, q)), 2) for q in (50, 95)],
           **summ})
    p.close()


if __name__ == "__main__":
    _bench()
