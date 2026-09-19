"""Sikistirma onarim agi v0 (F27): 1080p -> 1080p, tek kare, kendi kodumuz (sifirdan).

Girdi TOD benzeri H.264 (~4,8 Mbps) karesi, hedef ayni karenin sikistirilmamis hali. Ag sadece
duzeltmeyi (artik) ogrenir: cikis = girdi + duzeltme. Son katman sifirla baslar, yani egitimin
basinda ag birebir girdiyi verir (kotu baslangic "bozma" yapmaz).

Hiz icin kare 2x2 bloklara katlanir (pixel_unshuffle: 3 kanal 1080x1920 -> 12 kanal 540x960),
govde yarim cozunurlukte calisir (4 kat az piksel), sonra geri acilir. H.264 hasari (blok,
halka, renk lekesi) 4-16 px olcekte; yarim cozunurlukte 3x3 evrisim yine 6x6 px gorur.

Agirlik dosyasi: weights/repair/<ad>.pth = {"state_dict", "ch", "blocks"}.
TensorRT motoru: weights/trt/repair_<ad>_<W>x<H>_fp16.engine, girdi/cikti BGR 0..255 fp16
(donusumler motorun icinde; tools/build_trt_repair.py).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS = os.path.join(ROOT, "weights", "repair")


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c2(F.relu(self.c1(x), inplace=True))


class RepairNet(nn.Module):
    def __init__(self, ch: int = 32, blocks: int = 4) -> None:
        super().__init__()
        self.ch, self.blocks = ch, blocks
        self.head = nn.Conv2d(12, ch, 3, 1, 1)
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.tail = nn.Conv2d(ch, 12, 3, 1, 1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RGB 0..1 [N,3,H,W] (H, W cift) -> RGB [N,3,H,W] (kirpilmamis)."""
        f = self.head(F.pixel_unshuffle(x, 2))
        f = f + self.body(f)
        return x + F.pixel_shuffle(self.tail(F.relu(f)), 2)


class RepairBgr255(nn.Module):
    """TensorRT sarmalayici: BGR 0..255 fp16 -> onarim -> BGR 0..255 (yuvarlanmis, sinirli).
    Renk sirasi ve olcek donusumu motorun icinde kalsin diye (canli hatta ek cekirdek yok)."""

    def __init__(self, net: RepairNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x.flip(1) * (1.0 / 255.0)
        y = self.net(rgb).flip(1) * 255.0
        return torch.round(torch.clamp(y, 0.0, 255.0))


def weight_path(name: str) -> str:
    return os.path.join(WEIGHTS, f"{name}.pth")


def load_repair(name: str) -> RepairNet:
    ck = torch.load(weight_path(name), map_location="cpu", weights_only=True)
    net = RepairNet(ck["ch"], ck["blocks"])
    net.load_state_dict(ck["state_dict"])
    return net.eval().cuda()


def save_repair(net: RepairNet, name: str) -> str:
    os.makedirs(WEIGHTS, exist_ok=True)
    path = weight_path(name)
    sd = {k: v.detach().float().cpu() for k, v in net.state_dict().items()}
    torch.save({"state_dict": sd, "ch": net.ch, "blocks": net.blocks}, path)
    return path


def engine_path(name: str, h: int = 1080, w: int = 1920) -> str:
    return os.path.join(ROOT, "weights", "trt", f"repair_{name}_{w}x{h}_fp16.engine")
