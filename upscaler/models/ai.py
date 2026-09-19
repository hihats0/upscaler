"""AI yeniden cizim adaylari (goal 2026-09-20 "S24 Ultra gibi"): 1080p RGB -> 1080p RGB.

Hazir modeller (lisans: reports/2026-09-20-ai-yeniden-cizim.md):
- Real-ESRGAN (BSD-3): realesr-general-x4v3 / -wdn- (SRVGGNetCompact 64x32), realesr-animevideov3
  (64x16), RealESRGAN_x2plus (RRDB 23 blok). Agirliklar weights/hazir/.
- SwinIR-M real x2 GAN (Apache-2.0): sadece kiyas (canli icin cok yavas).

Yollar: "yeniden cizim" = kareyi kucult (540p ya da 270p), GAN ile 1080p'ye buyut. "Asiri ornekleme" =
1080p'yi 4K'ya buyut, geri 1080p'ye kucult (detayi korur, pahali).

Kendi ince ayarli modellerimiz: weights/ai/<ad>.pth = {"arch", "kw", "state_dict"}.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HAZIR = os.path.join(ROOT, "weights", "hazir")
AI_W = os.path.join(ROOT, "weights", "ai")


def _load_sd(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("params_ema", "params", "state_dict"):
        if isinstance(ck, dict) and k in ck:
            return ck[k]
    return ck


def srvgg(name: str, feat: int = 64, conv: int = 32, scale: int = 4) -> nn.Module:
    from third_party.realesrgan.arch import SRVGGNetCompact
    m = SRVGGNetCompact(3, 3, feat, conv, scale, "prelu")
    m.load_state_dict(_load_sd(os.path.join(HAZIR, name)))
    return m


def rrdb_x2() -> nn.Module:
    from third_party.realesrgan.arch import RRDBNet
    m = RRDBNet(3, 3, scale=2, num_feat=64, num_block=23, num_grow_ch=32)
    m.load_state_dict(_load_sd(os.path.join(HAZIR, "RealESRGAN_x2plus.pth")))
    return m


def swinir_x2() -> nn.Module:
    from third_party.swinir.network_swinir import SwinIR
    m = SwinIR(upscale=2, in_chans=3, img_size=64, window_size=8, img_range=1.0, depths=[6] * 6,
               embed_dim=180, num_heads=[6] * 6, mlp_ratio=2, upsampler="nearest+conv", resi_connection="1conv")
    m.load_state_dict(_load_sd(os.path.join(HAZIR, "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth")))
    return m


class Redraw(nn.Module):
    """1080p -> (kucult 1/down) -> model (x scale) -> gerekirse alan ortalamasiyla 1080p'ye."""

    def __init__(self, net: nn.Module, down: int, scale: int, pad: int = 1) -> None:
        super().__init__()
        self.net, self.down, self.scale, self.pad = net, down, scale, pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        s = F.avg_pool2d(x, self.down) if self.down > 1 else x
        sh, sw = s.shape[-2:]
        ph, pw = (-sh) % self.pad, (-sw) % self.pad
        if ph or pw:
            s = F.pad(s, (0, pw, 0, ph), mode="reflect")
        y = self.net(s)[..., :sh * self.scale, :sw * self.scale]
        f = self.scale // self.down
        if f > 1:
            y = F.avg_pool2d(y, f)
        return y


def build(name: str) -> nn.Module:
    """Aday adi -> RGB 0..1 [N,3,1080,1920] -> RGB [N,3,1080,1920] modul (eval, cuda)."""
    if name == "esr-gen-270":
        m = Redraw(srvgg("realesr-general-x4v3.pth"), 4, 4)
    elif name == "esr-wdn-270":
        m = Redraw(srvgg("realesr-general-wdn-x4v3.pth"), 4, 4)
    elif name == "esr-anime-270":
        m = Redraw(srvgg("realesr-animevideov3.pth", conv=16), 4, 4)
    elif name == "esr-gen-540d":
        m = Redraw(srvgg("realesr-general-x4v3.pth"), 2, 4)
    elif name == "esr-anime-540d":
        m = Redraw(srvgg("realesr-animevideov3.pth", conv=16), 2, 4)
    elif name == "esrx2-540":
        m = Redraw(rrdb_x2(), 2, 2)
    elif name == "esrx2-1080d":
        m = Redraw(rrdb_x2(), 1, 2)
    elif name == "swinir-540":
        m = Redraw(swinir_x2(), 2, 2, pad=8)
    elif name == "bicubic-540":  # referans: ayni kucultme, sadece bicubic (yeniden cizimin bilgi kaybi)
        m = _Bicubic540()
    else:
        m = load_ai(name)
    return m.eval().cuda()


class _Bicubic540(nn.Module):
    def forward(self, x):
        return F.interpolate(F.avg_pool2d(x, 2), scale_factor=2, mode="bicubic", align_corners=False)


# ---- kendi modellerimiz ----

class AiNet(nn.Module):
    """Canli AI agi: 1080p girdi 2x2 katlanir (12 kanal 540p, bilgi kaybi yok), govde 540p'de
    SRVGG tarzi (conv + PReLU), cikis 2x2 acilir ve girdiye eklenir. frames>1: komsu kareler
    (t-1, t, t+1) kanal olarak eklenir; cikis orta kareye eklenir.
    Hazir baslatma: Real-ESRGAN general-x4v3'un govde katmanlari (64 kanal) kopyalanabilir."""

    def __init__(self, feat: int = 64, conv: int = 8, frames: int = 1) -> None:
        super().__init__()
        self.feat, self.conv, self.frames = feat, conv, frames
        self.head = nn.Conv2d(12 * frames, feat, 3, 1, 1)
        self.act0 = nn.PReLU(feat)
        body = []
        for _ in range(conv):
            body += [nn.Conv2d(feat, feat, 3, 1, 1), nn.PReLU(feat)]
        self.body = nn.Sequential(*body)
        self.tail = nn.Conv2d(feat, 12, 3, 1, 1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, 3*frames, H, W] RGB 0..1 (kareler kanal ekseninde, orta kare merkez)."""
        c = (self.frames // 2) * 3
        mid = x[:, c:c + 3]
        f = self.act0(self.head(F.pixel_unshuffle(x, 2)))
        f = f + self.body(f)
        return mid + F.pixel_shuffle(self.tail(f), 2)


class AiBgr255(nn.Module):
    """BGR 0..255 (kareler kanal ekseninde) -> aday -> BGR 0..255. Donusumler motorun icinde."""

    def __init__(self, net: nn.Module, frames: int) -> None:
        super().__init__()
        self.net, self.frames = net, frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        rgb = x.view(n, self.frames, 3, h, w).flip(2).reshape(n, 3 * self.frames, h, w) * (1.0 / 255.0)
        y = self.net(rgb).flip(1) * 255.0
        return torch.round(torch.clamp(y, 0.0, 255.0))


def weight_path(name: str) -> str:
    return os.path.join(AI_W, f"{name}.pth")


def load_ai(name: str) -> nn.Module:
    ck = torch.load(weight_path(name), map_location="cpu", weights_only=True)
    net = AiNet(**ck["kw"])
    net.load_state_dict(ck["state_dict"])
    return net


def save_ai(net: AiNet, name: str, extra: dict | None = None) -> str:
    os.makedirs(AI_W, exist_ok=True)
    sd = {k: v.detach().float().cpu() for k, v in net.state_dict().items()}
    torch.save({"arch": "AiNet", "kw": {"feat": net.feat, "conv": net.conv, "frames": net.frames},
                "state_dict": sd, **(extra or {})}, weight_path(name))
    return weight_path(name)


def engine_path(name: str, h: int = 1080, w: int = 1920) -> str:
    return os.path.join(ROOT, "weights", "trt", f"ai_{name}_{w}x{h}_fp16.engine")
