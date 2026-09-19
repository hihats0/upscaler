"""Kare isleme.

BaselineProcessor, Hat 1.1'in taban islemcisidir: model YOK. Iki kare arasi
dogrusal harman (ara kare) + bicubic buyutme. Amaci hattin (zaman cizgisi,
tampon, hiz, cikis) dogru calistigini olcmek.

RifeProcessor (Hat 1.2): ara kare RIFE 4.25 ile kaynak cozunurlukte, buyutme bicubic.
RifeFlowProcessor: RIFE akisi yarim cozunurlukte, kaydirma kaynak cozunurlukte, bicubic.
SrProcessor (Hat 1.2): ara kare (rife-flow ya da harman) kaynak cozunurlukte, buyutme
gercek SR modeli (RT4KSR x2, TensorRT). split=True: sol yari SR, sag yari bicubic.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def fit_size(src_h: int, src_w: int, out_h: int, out_w: int) -> tuple[int, int, int, int]:
    """En-boy oranini koruyarak sigdirma: (yukseklik, genislik, ust bosluk, sol bosluk)."""
    scale = min(out_w / src_w, out_h / src_h)
    h = min(out_h, round(src_h * scale))
    w = min(out_w, round(src_w * scale))
    return h, w, (out_h - h) // 2, (out_w - w) // 2


def plan_input(h: int, w: int, canvases: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Kaynak boyutu icin kullanilacak motor tuvali (yukseklik, genislik).

    Tam eslesme varsa o. Yoksa kaynagi en-boy koruyarak en buyuk icerik alaniyla tasiyan tuval
    (esitlikte kucuk tuval). Hic tuval yoksa None (ayri parcalar).
    Not: TOD ABR 720p'ye dusse de pencere boyutu degismez (Chrome videoyu pencereye buyutur);
    boyut degisimi pratikte tam ekran <-> pencere gecisinde olur.
    """
    if not canvases:
        return None
    if (h, w) in canvases:
        return (h, w)

    def score(c):
        fh, fw, _, _ = fit_size(h, w, c[0], c[1])
        return (fh * fw, -c[0] * c[1])
    return max(canvases, key=score)


def fit_bgra(bgra: torch.Tensor, ch: int, cw: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """(H, W, 4) uint8 BGRA -> (ch, cw, 4) uint8, en-boy korunur, bosluk siyah, alfa 255 (GPU)."""
    h, w, top, left = fit_size(bgra.shape[0], bgra.shape[1], ch, cw)
    if out is None:
        out = torch.zeros((ch, cw, 4), dtype=torch.uint8, device=bgra.device)
        out[..., 3] = 255
    if (h, w) == tuple(bgra.shape[:2]):
        out[top:top + h, left:left + w].copy_(bgra)
        out[top:top + h, left:left + w, 3] = 255
        return out
    x = bgra[..., :3].permute(2, 0, 1).unsqueeze(0).to(torch.float16)
    y = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
    out[top:top + h, left:left + w, :3].copy_(y[0].permute(1, 2, 0).add_(0.5).clamp_(0, 255))
    return out


def available_canvases(sr_name: str, out_h: int = 2160, out_w: int = 3840) -> list[tuple[int, int]]:
    """weights/trt altinda still + rgbsr motoru birlikte bulunan kaynak boyutlari."""
    import re
    from .models.rife import ROOT
    base = os.path.join(ROOT, "weights", "trt")
    if not os.path.isdir(base):
        return []
    pat = re.compile(rf"pipe_still\+{re.escape(sr_name)}_(\d+)x(\d+)_to{out_w}x{out_h}_fp16\.engine$")
    found = []
    for f in os.listdir(base):
        m = pat.match(f)
        if m and os.path.exists(os.path.join(base, f.replace("pipe_still+", "pipe_rgbsr+"))):
            found.append((int(m.group(2)), int(m.group(1))))
    return sorted(found)


class BaselineProcessor:
    name = "baseline"

    def __init__(self, out_h: int = 2160, out_w: int = 3840, dtype: torch.dtype = torch.float16) -> None:
        self.out_h, self.out_w, self.dtype = out_h, out_w, dtype
        self._canvas: torch.Tensor | None = None

    @staticmethod
    def to_float(bgra: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """(H, W, 4) uint8 BGRA -> (1, 3, H, W) BGR 0..255."""
        return bgra[..., :3].permute(2, 0, 1).unsqueeze(0).to(dtype)

    @torch.inference_mode()
    def __call__(self, a_bgra: torch.Tensor, b_bgra: torch.Tensor, alpha: float) -> torch.Tensor:
        """(H, W, 4) uint8 iki kare -> (1, 3, out_h, out_w) uint8 BGR."""
        x = self.to_float(a_bgra, self.dtype)
        if alpha > 0.0:
            x = torch.lerp(x, self.to_float(b_bgra, self.dtype), alpha)
        return self.upscale(x)

    def upscale(self, x: torch.Tensor) -> torch.Tensor:
        """(1, 3, H, W) BGR 0..255 float -> (1, 3, out_h, out_w) uint8, en-boy korunur."""
        h, w, top, left = fit_size(x.shape[2], x.shape[3], self.out_h, self.out_w)
        up = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False)
        up = up.clamp_(0, 255).round_().to(torch.uint8)
        if (h, w) == (self.out_h, self.out_w):
            return up
        if self._canvas is None or self._canvas.device != up.device:
            self._canvas = torch.zeros((1, 3, self.out_h, self.out_w), dtype=torch.uint8, device=up.device)
        self._canvas[:, :, top:top + h, left:left + w] = up
        return self._canvas


class RifeProcessor(BaselineProcessor):
    """Ara kare RIFE ile (kaynak cozunurlukte), buyutme bicubic."""

    EPS = 1e-3  # alpha bu kadar 0/1'e yakinsa gercek kare kullanilir

    def __init__(self, version: str = "4.25", out_h: int = 2160, out_w: int = 3840,
                 dtype: torch.dtype = torch.float16) -> None:
        super().__init__(out_h, out_w, dtype)
        from .models.rife import Rife
        self.rife = Rife(version, dtype=dtype)
        self.name = f"rife-{version}"

    @torch.inference_mode()
    def __call__(self, a_bgra: torch.Tensor, b_bgra: torch.Tensor, alpha: float) -> torch.Tensor:
        if alpha <= self.EPS:
            return self.upscale(self.to_float(a_bgra, self.dtype))
        if alpha >= 1.0 - self.EPS:
            return self.upscale(self.to_float(b_bgra, self.dtype))
        # BGR -> RGB, 0..1 (RIFE RGB bekler), sonra geri
        a = self.to_float(a_bgra, self.dtype).flip(1) / 255.0
        b = self.to_float(b_bgra, self.dtype).flip(1) / 255.0
        mid = self.rife.interpolate(a, b, alpha)
        return self.upscale(mid.flip(1) * 255.0)


class RifeFlowProcessor(BaselineProcessor):
    """F1/F2: RIFE akis+maskeyi kaynagin yarim cozunurlugunde hesaplar, gercek
    kaynak kareleri bu akisla kaydirip harmanlar, sonra bicubic 4K."""

    EPS = 1e-3

    def __init__(self, version: str = "4.25", flow_scale: float = 0.5, out_h: int = 2160, out_w: int = 3840,
                 dtype: torch.dtype = torch.float16, trt: bool = False) -> None:
        super().__init__(out_h, out_w, dtype)
        from .models.rife_flow import RifeFlow
        self.rf = RifeFlow(version, dtype=dtype)
        self.version = version
        self.flow_scale = flow_scale
        self.trt = trt
        self._engines: dict[tuple[int, int], object] = {}
        self.name = f"rife-flow-{version}@{flow_scale}" + ("-trt" if trt else "")

    def _flow(self, sa: torch.Tensor, sb: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
        """TensorRT motoru bu boyut icin varsa onu, yoksa PyTorch'u kullanir."""
        if not self.trt:
            return self.rf.flow(sa, sb, alpha)
        h, w = sa.shape[2:]
        m = self.rf.rife.multiple
        ph, pw = (h + m - 1) // m * m, (w + m - 1) // m * m
        key = (ph, pw)
        if key not in self._engines:
            from .models.rife import ROOT
            path = os.path.join(ROOT, "weights", "trt", f"rifeflow_{self.version}_{pw}x{ph}_fp16.engine")
            if os.path.exists(path):
                from .models.trt_engine import TrtEngine
                self._engines[key] = TrtEngine(path)
            else:
                print(f"[rife-flow-trt] motor yok ({pw}x{ph}), PyTorch kullaniliyor: tools/build_trt_flow.py --h {h} --w {w}")
                self._engines[key] = None
        eng = self._engines[key]
        if eng is None:
            return self.rf.flow(sa, sb, alpha)
        pad = (0, pw - w, 0, ph - h)
        x0 = F.pad(sa, pad) if any(pad) else sa
        x1 = F.pad(sb, pad) if any(pad) else sb
        ts = torch.full((1, 1, ph, pw), float(alpha), dtype=self.dtype, device=sa.device)
        out = eng(x0=x0.to(self.dtype), x1=x1.to(self.dtype), timestep=ts)
        return out["flow"][:, :, :h, :w].float(), torch.sigmoid(out["mask"][:, :, :h, :w].float())

    @torch.inference_mode()
    def mid_rgb(self, a_bgra: torch.Tensor, b_bgra: torch.Tensor, alpha: float) -> torch.Tensor:
        """Ara kare, kaynak cozunurlugunde RGB 0..1 (self.dtype)."""
        a = self.to_float(a_bgra, self.dtype).flip(1) / 255.0
        b = self.to_float(b_bgra, self.dtype).flip(1) / 255.0
        h, w = round(a.shape[2] * self.flow_scale), round(a.shape[3] * self.flow_scale)
        sa = F.interpolate(a, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        sb = F.interpolate(b, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        flow, mask = self._flow(sa, sb, alpha)
        return self.rf.synthesize(a, b, flow, mask)

    @torch.inference_mode()
    def __call__(self, a_bgra: torch.Tensor, b_bgra: torch.Tensor, alpha: float) -> torch.Tensor:
        if alpha <= self.EPS:
            return self.upscale(self.to_float(a_bgra, self.dtype))
        if alpha >= 1.0 - self.EPS:
            return self.upscale(self.to_float(b_bgra, self.dtype))
        return self.upscale(self.mid_rgb(a_bgra, b_bgra, alpha).flip(1) * 255.0)


def overlay_bicubic_right(out: torch.Tensor, mid: torch.Tensor, fit: tuple[int, int, int, int]) -> None:
    """Kiyas modu: cikisin sag yarisina ayni karenin bicubic buyutmesini yazar, ortaya 4 px cizgi.

    out: (1,3,out_h,out_w) uint8 BGR (yerinde degisir), mid: (1,3,H,W) RGB 0..1, fit: fit_size sonucu.
    Sadece sag yarinin kaynak sutunlari buyutulur (tam 4K bicubic ~3 ms yiyordu). Tam 2x olcekte
    kesilen dilimin sonucu tam goruntununkiyle birebir ayni: bicubic 4 komsu kullanir, 4 kaynak
    sutun pay kesim kenarinin etkisini disarida birakir.
    """
    h, w, top, left = fit
    half = out.shape[3] // 2
    x0 = max(half - left, 0)
    if x0 >= w:
        return
    src_w = mid.shape[3]
    if w == 2 * src_w and h == 2 * mid.shape[2]:
        c0 = max(x0 // 2 - 4, 0)
        src, off, bw = mid[:, :, :, c0:], x0 - 2 * c0, 2 * (src_w - c0)
    else:
        src, off, bw = mid, x0, w
    bic = F.interpolate(src.flip(1) * 255.0, size=(h, bw), mode="bicubic", align_corners=False)
    out[:, :, top:top + h, left + x0:left + w].copy_(bic[:, :, :, off:].add_(0.5).clamp_(0, 255))
    out[:, :, top:top + h, max(half - 2, 0):half + 2] = 255


class SrUpscaler:
    """SR modeli: RGB 0..1 [1,3,H,W] -> RGB 0..1 [1,3,sH,sW].

    Her girdi boyutu icin weights/trt/sr_<ad>_<W>x<H>_fp16.engine varsa TensorRT, yoksa PyTorch.
    Dikkat: TensorRT cikti tamponu sonraki cagrida uzerine yazilir.
    """

    def __init__(self, name: str = "rt4ksr-x2", dtype: torch.dtype = torch.float16, trt: bool = True) -> None:
        self.name, self.dtype, self.trt = name, dtype, trt
        self.net: torch.nn.Module | None = None
        self._engines: dict[tuple[int, int], object] = {}

    def _engine(self, h: int, w: int):
        key = (h, w)
        if key not in self._engines:
            from .models.rife import ROOT
            path = os.path.join(ROOT, "weights", "trt", f"sr_{self.name}_{w}x{h}_fp16.engine")
            if self.trt and os.path.exists(path):
                from .models.trt_engine import TrtEngine
                self._engines[key] = TrtEngine(path)
            else:
                if self.trt:
                    print(f"[sr] motor yok ({w}x{h}), PyTorch kullaniliyor: "
                          f"tools/build_trt_sr.py --model {self.name} --h {h} --w {w}")
                self._engines[key] = None
        return self._engines[key]

    @torch.inference_mode()
    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        h, w = rgb.shape[2:]
        ph, pw = h + h % 2, w + w % 2  # PixelUnshuffle cift boyut ister
        x = rgb.to(self.dtype)
        if (ph, pw) != (h, w):
            x = F.pad(x, (0, pw - w, 0, ph - h), mode="replicate")
        eng = self._engine(ph, pw)
        if eng is not None:
            y = eng(x=x)["y"]
        else:
            if self.net is None:
                from .models.sr import load_sr
                self.net = load_sr(self.name).to(self.dtype)
            y = self.net(x)
        s = y.shape[2] // ph
        return y[:, :, :h * s, :w * s] if (ph, pw) != (h, w) else y


class SrProcessor(BaselineProcessor):
    """Ara kare kaynak cozunurlugunde (rife-flow ya da harman), buyutme SR modeliyle.

    split=True: kiyas modu, sol yari SR, sag yari ayni ara karenin bicubic buyutmesi.
    """

    EPS = 1e-3

    def __init__(self, sr_name: str = "rt4ksr-x2", interp: str = "rife-flow", split: bool = False,
                 out_h: int = 2160, out_w: int = 3840, dtype: torch.dtype = torch.float16, trt: bool = True) -> None:
        super().__init__(out_h, out_w, dtype)
        if interp not in ("rife-flow", "lerp"):
            raise ValueError(f"bilinmeyen ara kare yontemi: {interp}")
        self.sr = SrUpscaler(sr_name, dtype, trt)
        self.flow = RifeFlowProcessor("4.25", 0.5, out_h, out_w, dtype, trt=trt) if interp == "rife-flow" else None
        self.split = split
        self._out: torch.Tensor | None = None
        self._fit: tuple[int, int, int, int] | None = None
        self.name = f"{interp}{'-trt' if trt else ''}+{sr_name}{'+split' if split else ''}"

    def rgb(self, bgra: torch.Tensor) -> torch.Tensor:
        return self.to_float(bgra, self.dtype).flip(1) / 255.0

    @torch.inference_mode()
    def __call__(self, a_bgra: torch.Tensor, b_bgra: torch.Tensor, alpha: float) -> torch.Tensor:
        if alpha <= self.EPS:
            mid = self.rgb(a_bgra)
        elif alpha >= 1.0 - self.EPS:
            mid = self.rgb(b_bgra)
        elif self.flow is not None:
            mid = self.flow.mid_rgb(a_bgra, b_bgra, alpha)
        else:
            mid = torch.lerp(self.rgb(a_bgra), self.rgb(b_bgra), alpha)
        return self.finish(mid)

    def finish(self, mid: torch.Tensor) -> torch.Tensor:
        """RGB 0..1 kaynak cozunurlugu -> (1, 3, out_h, out_w) uint8 BGR (tampon yeniden kullanilir)."""
        fit = fit_size(mid.shape[2], mid.shape[3], self.out_h, self.out_w)
        h, w, top, left = fit
        if self._out is None or self._out.device != mid.device:
            self._out = torch.zeros((1, 3, self.out_h, self.out_w), dtype=torch.uint8, device=mid.device)
        if fit != self._fit:  # kaynak boyutu degisti: eski kenar bosluklarini temizle
            self._out.zero_()
            self._fit = fit
        out = self._out
        sr = self.sr(mid)
        if sr.shape[2:] != (h, w):
            sr = F.interpolate(sr.float(), size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        y = sr.flip(1)  # RGB -> BGR (kopya)
        out[:, :, top:top + h, left:left + w].copy_(y.mul_(255.0).add_(0.5).clamp_(0, 255))
        if self.split:
            overlay_bicubic_right(out, mid, fit)
        return out


class FusedSrProcessor(BaselineProcessor):
    """Karma yol (models/fused.py): ara karede akis TensorRT + warp PyTorch, SR + cikti donusumu
    (RGB->BGR, uint8, siyah bant) tek TensorRT motorunda. Gercek karede BGRA -> 4K tek motorda.

    Motor yoksa ayri parcali SrProcessor'a duser. Donen tensor motorun cikti tamponudur,
    sonraki cagrida uzerine yazilir.
    """

    EPS = 1e-3

    def __init__(self, sr_name: str = "rt4ksr-x2", out_h: int = 2160, out_w: int = 3840,
                 dtype: torch.dtype = torch.float16, split: bool = False,
                 canvases: list[tuple[int, int]] | None = None) -> None:
        super().__init__(out_h, out_w, dtype)
        self.sr_name = sr_name
        self.split = split
        self.flow = RifeFlowProcessor("4.25", 0.5, out_h, out_w, dtype, trt=True)
        self._engines: dict[tuple[int, int], tuple | None] = {}
        self._fallback: SrProcessor | None = None
        # Kaynak boyutu -> motor tuvali. Motoru olmayan boyut (pencere, 720p) en yakin tuvale
        # sigdirilir; boylece her boyut hizli yoldan gecer. Hic tuval yoksa ayri parcalar.
        self.canvases = available_canvases(sr_name, out_h, out_w) if canvases is None else list(canvases)
        self._plans: dict[tuple[int, int], tuple[int, int] | None] = {}
        self._fit_bufs: dict[tuple, torch.Tensor] = {}
        self.route = "henuz kare yok"
        self.name = f"rife-flow-trt+{sr_name}+fused-out" + ("+split" if split else "")

    def _get(self, h: int, w: int):
        key = (h, w)
        if key not in self._engines:
            from .models.fused import engine_paths
            paths = engine_paths(self.sr_name, h, w, self.out_h, self.out_w)
            if os.path.exists(paths["still"]) and os.path.exists(paths["rgbsr"]):
                from .models.trt_engine import TrtEngine
                self._engines[key] = (TrtEngine(paths["still"]), TrtEngine(paths["rgbsr"]))
            else:
                print(f"[fused] motor yok ({w}x{h}), ayri parcalar kullaniliyor: "
                      f"tools/build_trt_pipeline.py --h {h} --w {w}")
                self._engines[key] = None
        return self._engines[key]

    def _plan(self, h: int, w: int) -> tuple[int, int] | None:
        key = (h, w)
        if key not in self._plans:
            canvas = plan_input(h, w, self.canvases)
            if canvas is not None and self._get(*canvas) is None:
                canvas = None
            self._plans[key] = canvas
            self._fit_bufs.clear()
        canvas = self._plans[key]
        if canvas is None:
            self.route = f"ayri parcalar {w}x{h}"
        elif canvas == key:
            self.route = f"motor {w}x{h}"
        else:
            self.route = f"sigdirma {w}x{h}->{canvas[1]}x{canvas[0]}"
        return canvas

    def _fit(self, bgra: torch.Tensor, canvas: tuple[int, int], slot: int) -> torch.Tensor:
        key = (tuple(bgra.shape[:2]), canvas, slot)
        buf = self._fit_bufs.get(key)
        out = fit_bgra(bgra, canvas[0], canvas[1], buf)
        self._fit_bufs[key] = out
        return out

    @torch.inference_mode()
    def __call__(self, a_bgra: torch.Tensor, b_bgra: torch.Tensor, alpha: float) -> torch.Tensor:
        h, w = a_bgra.shape[:2]
        canvas = self._plan(h, w)
        if canvas is None:
            if self._fallback is None:
                self._fallback = SrProcessor(self.sr_name, "rife-flow", self.split, self.out_h, self.out_w, self.dtype)
            self._fallback.split = self.split
            return self._fallback(a_bgra, b_bgra, alpha)
        still, rgbsr = self._get(*canvas)
        real = alpha <= self.EPS or alpha >= 1.0 - self.EPS
        if canvas != (h, w):
            if real:
                src = self._fit(a_bgra if alpha <= self.EPS else b_bgra, canvas, 0)
                a_bgra = b_bgra = src
            else:
                a_bgra, b_bgra = self._fit(a_bgra, canvas, 0), self._fit(b_bgra, canvas, 1)
        mid = None
        if real:
            src = a_bgra if alpha <= self.EPS else b_bgra
            y = still(a=src)["y"]
        else:
            mid = self.flow.mid_rgb(a_bgra, b_bgra, alpha)
            y = rgbsr(x=mid.to(torch.float16))["y"]
        if self.split:
            if mid is None:
                mid = self.to_float(src, self.dtype).flip(1) / 255.0
            overlay_bicubic_right(y, mid, fit_size(mid.shape[2], mid.shape[3], self.out_h, self.out_w))
        return y


class PassProcessor(BaselineProcessor):
    """TV modu (1080p ekran, cikis = kaynak hizi): ara kare yok, SR yok, kare oldugu gibi.

    Cikis hizi kaynak hiziyla ayni oldugunda schedule her tikte gercek kareye oturur (alpha 0/1);
    oturmazsa (saat kaymasi) en yakin kare secilir, harman yapilmaz. Kaynak cikis boyutunda
    degilse en-boy korunarak sigdirilir. Donen tensor tamponun gorunumudur (kopya yok); sunucu
    onu kendi dokusuna kopyalar, secim ondan sonra birakilir.
    """

    name = "pass"

    def __init__(self, out_h: int = 1080, out_w: int = 1920) -> None:
        super().__init__(out_h, out_w)
        self.split = False
        self._fit_buf: torch.Tensor | None = None
        self.route = "henuz kare yok"

    def pick(self, a_bgra: torch.Tensor, b_bgra: torch.Tensor, alpha: float) -> torch.Tensor:
        """Secilen kare, (out_h, out_w, 4) uint8 BGRA."""
        src = a_bgra if alpha < 0.5 else b_bgra
        h, w = src.shape[:2]
        if (h, w) == (self.out_h, self.out_w):
            self.route = f"dogrudan {w}x{h}"
            return src
        if self._fit_buf is not None and self._fit_buf.shape[:2] != (self.out_h, self.out_w):
            self._fit_buf = None
        self._fit_buf = fit_bgra(src, self.out_h, self.out_w, self._fit_buf)
        self.route = f"sigdirma {w}x{h}->{self.out_w}x{self.out_h}"
        return self._fit_buf

    @torch.inference_mode()
    def __call__(self, a_bgra: torch.Tensor, b_bgra: torch.Tensor, alpha: float) -> torch.Tensor:
        return self.pick(a_bgra, b_bgra, alpha)[..., :3].permute(2, 0, 1).unsqueeze(0)


class RepairProcessor(PassProcessor):
    """TV modu + sikistirma onarimi (F27, models/repair.py): secilen kare 1080p onarim agindan gecer.

    TensorRT motoru (weights/trt/repair_<ad>_1920x1080_fp16.engine) yoksa PyTorch FP16 (yavas).
    split=True: sol yari onarimli, sag yari ham kare, ortada 4 px beyaz cizgi (S tusu).
    """

    def __init__(self, repair: str, out_h: int = 1080, out_w: int = 1920, split: bool = False) -> None:
        super().__init__(out_h, out_w)
        from .models.repair import RepairBgr255, engine_path, load_repair
        self.split = split
        self.repair = repair
        path = engine_path(repair, out_h, out_w)
        self.eng = self.net = None
        if os.path.exists(path):
            from .models.trt_engine import TrtEngine
            self.eng = TrtEngine(path)
        else:
            print(f"[repair] motor yok ({path}), PyTorch kullaniliyor: tools/build_trt_repair.py --name {repair}")
            self.net = RepairBgr255(load_repair(repair)).half()
        self._out = torch.empty((1, 3, out_h, out_w), dtype=torch.uint8, device="cuda")
        self.name = f"pass+repair-{repair}" + ("+split" if split else "")

    @torch.inference_mode()
    def __call__(self, a_bgra: torch.Tensor, b_bgra: torch.Tensor, alpha: float) -> torch.Tensor:
        src = self.pick(a_bgra, b_bgra, alpha)
        raw = src[..., :3].permute(2, 0, 1).unsqueeze(0)
        x = raw.half()
        y = self.eng(x=x)["y"] if self.eng is not None else self.net(x)
        out = self._out
        out.copy_(y)  # motor ciktisi zaten yuvarlanmis ve 0..255
        if self.split:
            half = self.out_w // 2
            out[:, :, :, half:].copy_(raw[:, :, :, half:])
            out[:, :, :, half - 2:half + 2] = 255
        return out


class AiProcessor(PassProcessor):
    """TV modu + AI yeniden cizim (goal 2026-09-20, models/ai.py AiNet): 1080p -> 1080p.

    Cok kareli ag (frames=3) t-1, t, t+1 karelerini ister: islemci son 3 yeni kareyi kendi
    tamponunda tutar ve bir kare geriden gelir (cikis = f[n-1], girdi f[n-2], f[n-1], f[n]).
    Bu 1 kare (20 ms) ek goruntu gecikmesi demek. Yeni kare, secilen tampon diliminin adresinden
    anlasilir (ayni dilim = ayni kare tekrar).
    gain: AI farkinin carpani (agresiflik); blend: durgun bolge harmani (titreme bastirma).
    3 kareli agda ikisi motorun icinde (weights/trt/ai_<ad>_bNNN[_gNNN]_..., build_trt_ai.py --blend
    --gain); motor yoksa PyTorch'ta (yavas). split=True: sol yari AI, sag yari ham (S tusu).
    """

    def __init__(self, ai: str, out_h: int = 1080, out_w: int = 1920, split: bool = False,
                 blend: float = 0.6, gain: float = 1.0, sat: float = 1.0, con: float = 1.0) -> None:
        super().__init__(out_h, out_w)
        from .models.ai import AiBgr255, StaticBlend, blend_engine_path, engine_path, load_ai
        from .models.trt_engine import TrtEngine
        self.split = split
        self.ai = ai
        net = load_ai(ai)
        self.frames = net.frames
        self.gain = gain
        self.eng = self.net = None
        self.eng_blend = False
        self.blend = None
        bpath = blend_engine_path(ai, blend, gain, out_h, out_w, sat, con)
        path = engine_path(ai, out_h, out_w)
        if self.frames == 3 and os.path.exists(bpath):
            self.eng, self.eng_blend = TrtEngine(bpath), True
        elif os.path.exists(path):
            self.eng = TrtEngine(path)
            print(f"[ai] harman/guc/renk motoru yok ({bpath}), harman ve guc PyTorch'ta, renk YOK")
        else:
            print(f"[ai] motor yok ({path}), PyTorch kullaniliyor: tools/build_trt_ai.py --name {ai}")
            self.net = AiBgr255(net.cuda().eval(), self.frames).half()
        if blend > 0 and not self.eng_blend:
            self.blend = StaticBlend(blend, 2.0 / 255, scale=255.0)
        self._hist = torch.zeros((self.frames, 3, out_h, out_w), dtype=torch.float16, device="cuda")
        self._po = torch.zeros((1, 3, out_h, out_w), dtype=torch.float16, device="cuda")
        self._n = 0
        self._last_ptr = None
        self._out = torch.empty((1, 3, out_h, out_w), dtype=torch.uint8, device="cuda")
        self.last_raw: torch.Tensor | None = None  # cikisla ayni anin ham karesi (keskinlik probu)
        self.name = (f"pass+ai-{ai}" + (f"+guc{gain}" if gain != 1.0 else "") + (f"+renk{sat}/{con}" if (sat, con) != (1.0, 1.0) else "") + (f"+blend{blend}" if blend > 0 else "")
                     + ("(motor)" if self.eng_blend else "") + ("+split" if split else ""))

    @torch.inference_mode()
    def __call__(self, a_bgra: torch.Tensor, b_bgra: torch.Tensor, alpha: float) -> torch.Tensor:
        src = self.pick(a_bgra, b_bgra, alpha)
        ptr = src.data_ptr()
        if ptr != self._last_ptr or self._n == 0:
            self._last_ptr = ptr
            raw = src[..., :3].permute(2, 0, 1)
            h = self._hist
            for i in range(self.frames - 1):  # kaydir (yerinde, yeni bellek yok)
                h[i].copy_(h[i + 1])
            h[-1].copy_(raw)
            mid = h[self.frames // 2]
            if self._n == 0:  # ilk kare: gecmisi ve onceki cikisi ayni kareyle doldur
                h[:] = h[-1]
                self._po.copy_(mid[None])
            self._n += 1
            x = h.view(1, 3 * self.frames, self.out_h, self.out_w)
            if self.eng_blend:
                y = self.eng(x=x, po=self._po)["y"]
                self._po.copy_(y)
            else:
                y = self.eng(x=x)["y"] if self.eng is not None else self.net(x)
                if self.gain != 1.0:
                    y = torch.round(torch.clamp(mid[None] + self.gain * (y - mid[None]), 0, 255))
                if self.blend is not None:
                    y = torch.round(self.blend(mid[None].float(), y.float()))
            self._out.copy_(y)
            self.last_raw = mid
            if self.split:
                half = self.out_w // 2
                self._out[0, :, :, half:].copy_(mid[:, :, half:])
                self._out[:, :, :, half - 2:half + 2] = 255
        return self._out


class AiAheadProcessor(AiProcessor):
    """AI on isleme (2026-09-19 mac gecesi): yayin 1,5 sn gecikmeli gosterildigi icin her kare
    tampona girince ayri bir GPU akisinda AI'dan gecer, sonuc onbellekte hazir bekler. Sunum aninda
    sadece hazir kare gosterilir (~1-2 ms). AI suresi TV tikine bagli degil: isinip yavaslayan GPU'da
    ara sira 20 ms'i asan kare vsync kacirmaz (senkron hatta 10 sn'de 1-4 kacirma, isi kisitlamasi).

    Sirali isler (3 kare penceresi ve harman onceki cikisa bagli). Sadece motor-ici harman/guc/renk
    motoruyla (eng_blend) calisir; yoksa AiProcessor gibi davranir. Watch: attach(ring), her tikte
    present_pick(p) (gosterilecek), sunumdan sonra work_ahead() (sonraki kareleri kuyruga at).
    """

    LEAD = 8      # gosterilen karenin en fazla kac kare onunu isle
    PER_CALL = 3  # bir cagrida en fazla kac kare (geride kalinca yetisme)

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.ring = None
        self.async_ahead = self.eng_blend
        self.ai_stream = torch.cuda.Stream()
        n = self.LEAD + 8
        self._pool = [torch.empty((1, 3, self.out_h, self.out_w), dtype=torch.uint8, device="cuda") for _ in range(n)]
        self._cache: dict[int, tuple[torch.Tensor, torch.cuda.Event]] = {}
        self._pending: list[tuple[torch.cuda.Event, int, int]] = []  # (kopya bitti, gen, slot)
        self._last_in: int | None = None
        self._gen: int | None = None
        self._disp: int | None = None
        self._run = 0
        self.hits = self.misses = self.fits = 0
        if self.async_ahead:
            self.name += "+onisleme"

    def attach(self, ring) -> None:
        self.ring = ring

    def _reset(self) -> None:
        for buf, _ in self._cache.values():
            self._pool.append(buf)
        self._cache.clear()
        self._last_in = None
        self._run = 0

    def _free_done(self) -> None:
        keep = []
        for ev, gen, slot in self._pending:
            if ev.query():
                self.ring._release(gen, slot)
            else:
                keep.append((ev, gen, slot))
        self._pending = keep

    def _evict(self, below: int) -> None:
        for i in [i for i in self._cache if i < below]:
            buf, ev = self._cache.pop(i)
            ev.synchronize()
            self._pool.append(buf)

    @torch.inference_mode()
    def work_ahead(self) -> None:
        if not self.async_ahead or self.ring is None or self._disp is None:
            return
        self._free_done()
        items = self.ring.take_after(self._last_in, self._disp + self.LEAD, self.PER_CALL)
        if not items:
            return
        with torch.cuda.stream(self.ai_stream):
            for e, view, gen in items:
                if tuple(view.shape[:2]) != (self.out_h, self.out_w):
                    # Kaynak 1080p degil (Chrome tam ekrandan cikti, 1920x1079): on isleme yok,
                    # present_pick senkron yola (sigdirma) duser.
                    self.ring._release(gen, e.slot)
                    self._last_in = e.stamp.index
                    self._run = 0
                    continue
                if gen != self._gen:
                    self._gen = gen
                    self._reset()
                idx = e.stamp.index
                if self._last_in is not None and idx != self._last_in + 1:
                    self._run = 0  # sira atladi (sure kaymasi, yeniden kilit): pencereyi bastan kur
                h = self._hist
                for i in range(self.frames - 1):
                    h[i].copy_(h[i + 1])
                h[-1].copy_(view[..., :3].permute(2, 0, 1))
                ev_copy = torch.cuda.Event()
                ev_copy.record()
                self._pending.append((ev_copy, gen, e.slot))
                if self._run == 0:
                    h[:] = h[-1]
                    self._po.copy_(h[-1][None])
                self._run += 1
                out_idx = idx if self._run == 1 else idx - self.frames // 2
                y = self.eng(x=h.view(1, 3 * self.frames, self.out_h, self.out_w), po=self._po)["y"]
                self._po.copy_(y)
                if out_idx in self._cache:
                    buf, _ = self._cache.pop(out_idx)
                else:
                    if not self._pool:
                        self._evict(min(self._cache) + 1)
                    buf = self._pool.pop()
                buf.copy_(y)
                ev = torch.cuda.Event()
                ev.record()
                self._cache[out_idx] = (buf, ev)
                self._last_in = idx

    @torch.inference_mode()
    def present_pick(self, p) -> torch.Tensor:
        """Gosterilecek kare: onbellekte hazirsa AI cikisi, degilse ham (sayilir)."""
        e = p.a if p.alpha < 0.5 else p.b
        src = p.fa if p.alpha < 0.5 else p.fb
        if tuple(src.shape[:2]) != (self.out_h, self.out_w):
            self._disp = e.stamp.index
            self.fits += 1
            return AiProcessor.__call__(self, p.fa, p.fb, p.alpha)  # sigdirma + senkron AI
        idx = e.stamp.index
        if p.gen != self._gen and self._gen is not None:
            self._reset()
            self._gen = p.gen
        if self._disp is None:
            self._gen = p.gen
            self._last_in = idx - 1  # ilk karede onceki kareleri isleme
        self._disp = idx
        self._evict(idx - 1)
        raw = src[..., :3].permute(2, 0, 1).unsqueeze(0)
        hit = self._cache.get(idx)
        if hit is None:
            self.misses += 1
            if self.split:
                self._out.copy_(raw)
                return self._out
            return raw
        self.hits += 1
        self.last_raw = raw[0]  # keskinlik probu: ayni anin ham karesi
        buf, ev = hit
        torch.cuda.current_stream().wait_event(ev)
        if not self.split:
            return buf
        half = self.out_w // 2
        self._out.copy_(buf)
        self._out[:, :, :, half:].copy_(raw[:, :, :, half:])
        self._out[:, :, :, half - 2:half + 2] = 255
        return self._out


PROCESSORS = ["baseline", "rife", "rife-lite", "rife-flow", "rife-lite-flow", "rife-flow-trt",
              "sr", "rife-flow-sr", "rife-flow-trt-sr", "fused-sr", "pass", "repair", "ai"]


def make_processor(name: str, out_h: int = 2160, out_w: int = 3840, sr: str = "rt4ksr-x2",
                   split: bool = False, repair: str = "", ai: str = "", ai_blend: float = 0.6, ai_gain: float = 1.0,
                   ai_sat: float = 1.0, ai_con: float = 1.0, ai_ahead: bool = True) -> BaselineProcessor:
    if name == "baseline":
        return BaselineProcessor(out_h, out_w)
    if name == "rife":
        return RifeProcessor("4.25", out_h, out_w)
    if name == "rife-lite":
        return RifeProcessor("4.25.lite", out_h, out_w)
    if name == "rife-flow":
        return RifeFlowProcessor("4.25", 0.5, out_h, out_w)
    if name == "rife-lite-flow":
        return RifeFlowProcessor("4.25.lite", 0.5, out_h, out_w)
    if name == "rife-flow-trt":
        return RifeFlowProcessor("4.25", 0.5, out_h, out_w, trt=True)
    if name == "sr":
        return SrProcessor(sr, "lerp", split, out_h, out_w, trt=True)
    if name == "rife-flow-sr":
        return SrProcessor(sr, "rife-flow", split, out_h, out_w, trt=False)
    if name == "rife-flow-trt-sr":
        return SrProcessor(sr, "rife-flow", split, out_h, out_w, trt=True)
    if name == "fused-sr":
        return FusedSrProcessor(sr, out_h, out_w, split=split)
    if name == "pass":
        return PassProcessor(out_h, out_w)
    if name == "repair":
        return RepairProcessor(repair, out_h, out_w, split=split)
    if name == "ai":
        cls = AiAheadProcessor if ai_ahead else AiProcessor
        return cls(ai, out_h, out_w, split=split, blend=ai_blend, gain=ai_gain, sat=ai_sat, con=ai_con)
    raise ValueError(f"bilinmeyen islemci: {name}")
