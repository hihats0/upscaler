"""Kare isleme.

BaselineProcessor, Hat 1.1'in taban islemcisidir: model YOK. Iki kare arasi
dogrusal harman (ara kare) + bicubic buyutme. Amaci hattin (zaman cizgisi,
tampon, hiz, cikis) dogru calistigini olcmek.

RifeProcessor (Hat 1.2): ara kare RIFE 4.25 ile kaynak cozunurlukte, buyutme
yine bicubic (SR modeli sonra gelir).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def fit_size(src_h: int, src_w: int, out_h: int, out_w: int) -> tuple[int, int, int, int]:
    """En-boy oranini koruyarak sigdirma: (yukseklik, genislik, ust bosluk, sol bosluk)."""
    scale = min(out_w / src_w, out_h / src_h)
    h = min(out_h, round(src_h * scale))
    w = min(out_w, round(src_w * scale))
    return h, w, (out_h - h) // 2, (out_w - w) // 2


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
        import os
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
    def __call__(self, a_bgra: torch.Tensor, b_bgra: torch.Tensor, alpha: float) -> torch.Tensor:
        if alpha <= self.EPS:
            return self.upscale(self.to_float(a_bgra, self.dtype))
        if alpha >= 1.0 - self.EPS:
            return self.upscale(self.to_float(b_bgra, self.dtype))
        a = self.to_float(a_bgra, self.dtype).flip(1) / 255.0
        b = self.to_float(b_bgra, self.dtype).flip(1) / 255.0
        h, w = round(a.shape[2] * self.flow_scale), round(a.shape[3] * self.flow_scale)
        sa = F.interpolate(a, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        sb = F.interpolate(b, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        flow, mask = self._flow(sa, sb, alpha)
        mid = self.rf.synthesize(a, b, flow, mask)
        return self.upscale(mid.flip(1) * 255.0)


def make_processor(name: str, out_h: int = 2160, out_w: int = 3840) -> BaselineProcessor:
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
    raise ValueError(f"bilinmeyen islemci: {name}")
