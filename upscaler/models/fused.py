"""Canli hattin ara kare + SR yolunu tek TensorRT motorunda birlestirir (Hat 1.2 hiz).

Olcum (2026-09-15, 1920x1020 -> 4K, ayri parcalar, ara kare): BGRA->RGB 0,5 + kucultme 0,5 +
akis TRT 5,0 + warp/harman 2,8 + SR TRT 3,2 + cikti uint8 2,1 = 12-14 ms (TOD canli p95 15,8).
PyTorch yapistirma islemleri ve aradaki ara tamponlar tek motorda kaynasir.

InterpSR: (a, b: BGRA uint8 [H,W,4], t: [1,1,1,1]) -> [1,3,out_h,out_w] uint8 BGR (letterbox)
StillSR:  (a) -> ayni cikti; gercek kare (alpha 0/1) icin.
RgbSR:    (x: [1,3,H,W] fp16 RGB 0..1) -> ayni cikti; karma yolda ara kare icin.

Olcum (1920x1020): InterpSR tek motor 12,60 ms, ayri parcalar 12,12 ms (kazanc yok, kullanilmiyor,
tools/build_trt_pipeline.py --full ile derlenir). StillSR 2,82 ms, ayri parcalar 4,87 ms.
Karma yol (FusedSrProcessor): akis TRT + warp PyTorch'ta, SR + cikti donusumu RgbSR motorunda.

Kucultme farki: antialias bilinear 0,5 yerine onun ic bolgede birebir esdegeri [1,3,3,1]/8
ayrik filtre (kenarda replicate; PyTorch kenarda agirliklari yeniden normalize eder).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..process import fit_size
from .rife import ROOT


def engine_paths(sr_name: str, h: int, w: int, out_h: int = 2160, out_w: int = 3840) -> dict[str, str]:
    """Motor yollari: interp (tam kaynasik), still (gercek kare), rgbsr (karma yol ara kare)."""
    base = os.path.join(ROOT, "weights", "trt")
    tag = f"{w}x{h}_to{out_w}x{out_h}_fp16"
    return {"interp": os.path.join(base, f"pipe_rifeflow4.25+{sr_name}_{tag}.engine"),
            "still": os.path.join(base, f"pipe_still+{sr_name}_{tag}.engine"),
            "rgbsr": os.path.join(base, f"pipe_rgbsr+{sr_name}_{tag}.engine")}


def bgra_to_rgb(bgra: torch.Tensor) -> torch.Tensor:
    """[H,W,4] uint8 BGRA -> [1,3,H,W] fp16 RGB 0..1. Once cast: TensorRT uint8'i sadece G/C ve Cast'te destekler."""
    x = bgra.to(torch.float16)[:, :, :3].flip(-1)
    return x.permute(2, 0, 1).unsqueeze(0) / 255.0


class HalfDown(nn.Module):
    """2x kucultme, antialias bilinear esdegeri ([1,3,3,1]/8 x [1,3,3,1]/8)."""

    def __init__(self) -> None:
        super().__init__()
        k = torch.tensor([1.0, 3.0, 3.0, 1.0]) / 8.0
        self.register_buffer("k", (k[:, None] * k[None, :]).expand(3, 1, 4, 4).clone().half())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), self.k.to(x.dtype), stride=2, groups=3)


class ToOutput(nn.Module):
    """[1,3,sH,sW] fp16 RGB 0..1 -> [1,3,out_h,out_w] uint8 BGR, en-boy korunur (siyah bant)."""

    def __init__(self, sh: int, sw: int, H: int, W: int, out_h: int, out_w: int) -> None:
        super().__init__()
        h, w, top, left = fit_size(H, W, out_h, out_w)
        if (h, w) != (sh, sw):
            raise ValueError(f"SR ciktisi {sw}x{sh} cikis alanina ({w}x{h}) tam oturmuyor")
        self.pad = (left, out_w - w - left, top, out_h - h - top)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        y = (rgb.flip(1) * 255.0 + 0.5).clamp(0.0, 255.0)
        if any(self.pad):
            y = F.pad(y, self.pad)
        return y.to(torch.uint8)


class InterpSR(nn.Module):
    def __init__(self, rf, sr: nn.Module, H: int, W: int, out_h: int = 2160, out_w: int = 3840,
                 scale: int = 2) -> None:
        super().__init__()
        if H % 2 or W % 2:
            raise ValueError("kaynak boyutu cift olmali")
        self.rf, self.net, self.sr = rf, rf.net, sr
        self.h, self.w = H // 2, W // 2
        m = rf.rife.multiple
        self.ph, self.pw = (self.h + m - 1) // m * m, (self.w + m - 1) // m * m
        div, grid = rf.rife._grid(self.ph, self.pw)
        self.register_buffer("div", div.clone())
        self.register_buffer("grid", grid.clone())
        self.down = HalfDown()
        self.out = ToOutput(H * scale, W * scale, H, W, out_h, out_w)

    def forward(self, a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ra, rb = bgra_to_rgb(a), bgra_to_rgb(b)
        pad = (0, self.pw - self.w, 0, self.ph - self.h)
        x0 = F.pad(self.down(ra), pad)
        x1 = F.pad(self.down(rb), pad)
        ts = t.to(torch.float16).expand(1, 1, self.ph, self.pw)
        flow, mask = self.rf.flow_padded(x0, x1, ts, self.div, self.grid)
        flow = flow[:, :, :self.h, :self.w].float()
        mask = torch.sigmoid(mask[:, :, :self.h, :self.w].float())
        mid = self.rf.synthesize(ra, rb, flow, mask)
        return self.out(self.sr(mid.to(torch.float16)))


class StillSR(nn.Module):
    def __init__(self, sr: nn.Module, H: int, W: int, out_h: int = 2160, out_w: int = 3840, scale: int = 2) -> None:
        super().__init__()
        self.sr = sr
        self.out = ToOutput(H * scale, W * scale, H, W, out_h, out_w)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.out(self.sr(bgra_to_rgb(a)))


class RgbSR(nn.Module):
    def __init__(self, sr: nn.Module, H: int, W: int, out_h: int = 2160, out_w: int = 3840, scale: int = 2) -> None:
        super().__init__()
        self.sr = sr
        self.out = ToOutput(H * scale, W * scale, H, W, out_h, out_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.sr(x))
