"""Super cozunurluk (SR) modelleri: yukleme ve kayit.

Girdi/cikti: [1, 3, H, W] RGB 0..1. Agirliklar weights/ altinda (gitignore).

Olcum (2026-09-15, 4070 Laptop, 1080p -> 4K):
- efrlfn-x2: eager fp16 306 ms, TensorRT 105 ms. Ag tam 1080p'de 52 kanal calisiyor,
  bellek bant genisligine takiliyor. Canli 4K60'a SIGMAZ (makale 360x480 girdiyle olcmus).
- rt4ksr-x2: ag 540p'de (PixelUnshuffle) 24 kanal. Olcum: reports/2026-09-15-hat12-sr-modeli.md
"""
from __future__ import annotations

import os

import torch

from .rife import ROOT

# ad -> (tur, olcek, agirlik yolu)
_MODELS = {
    "efrlfn-x2": ("efrlfn", 2, "weights/efrlfn/EfRLFN-2x.pt"),
    "rt4ksr-x2": ("rt4ksr", 2, "weights/rt4ksr/rt4ksr_x2.pth"),
}


def sr_names() -> list[str]:
    return sorted(_MODELS)


def sr_scale(name: str) -> int:
    return _MODELS[name][1]


def load_sr(name: str, device: str = "cuda", rep: bool = True) -> torch.nn.Module:
    """rep=False sadece rt4ksr icin: egitim bicimi (reparametrizasyon dogrulamasi)."""
    if name not in _MODELS:
        raise ValueError(f"bilinmeyen SR modeli: {name} (secenekler: {sr_names()})")
    kind, scale, rel = _MODELS[name]
    path = os.path.join(ROOT, rel)
    if kind == "efrlfn":
        from third_party.efrlfn.arch import EfRLFN
        net = EfRLFN(upscale=scale)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    elif kind == "rt4ksr":
        from third_party.rt4ksr.arch import RT4KSR, clean_checkpoint, rep_state_dict
        ck = torch.load(path, map_location="cpu", weights_only=True)
        sd = clean_checkpoint(ck["state_dict"])
        net = RT4KSR(upscale=scale, rep=rep)
        net.load_state_dict(rep_state_dict(sd) if rep else sd, strict=True)
    else:
        raise ValueError(kind)
    return net.to(device).eval()
