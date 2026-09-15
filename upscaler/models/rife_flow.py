"""RIFE'i sadece hareket haritasi (akis) + maske icin kullanmak (F1/F2).

Olcum (2026-09-15): RIFE 4.25 1080p'de tam calisma ~45 ms (22 kare/sn). Zamanin
cogu tam cozunurlukte kopya/tip donusumu, birlestirme ve konvolusyon.

Burada RIFE dusuk cozunurlukte (orn. 960x540) calisir ve SADECE akis + maske
dondurur. Akis buyutulup yuksek cozunurluklu gercek kareler onunla kaydirilir
(warp) ve maskeyle harmanlanir. Tam cozunurlukte ag govdesi hic calismaz.

IFNet ileri gecisi third_party/vsrife/IFNet_HDv3_v4_25.py'den (MIT) uyarlanmistir;
tek fark son harmanlamayi yapmayip akis ve maskeyi dondurmesidir.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from third_party.vsrife.warplayer import warp

from .rife import Rife


class RifeFlow:
    def __init__(self, version: str = "4.25", device: str = "cuda", dtype: torch.dtype = torch.float16) -> None:
        self.rife = Rife(version, device=device, dtype=dtype, scale=1.0)
        self.net = self.rife.net
        self.device, self.dtype = device, dtype
        self._scales: dict[tuple[float, float], torch.Tensor] = {}

    @torch.inference_mode()
    def flow(self, img0: torch.Tensor, img1: torch.Tensor, t: float) -> tuple[torch.Tensor, torch.Tensor]:
        """img0, img1: [1,3,h,w] RGB 0..1 (self.dtype), t: 0..1.

        Donus: akis [1,4,h,w] float32 (piksel, h/w cozunurlugunde; 0-1: t->0, 2-3: t->1),
               maske [1,1,h,w] float32 0..1 (img0 agirligi).
        """
        net = self.net
        h, w = img0.shape[2:]
        m = self.rife.multiple
        ph, pw = (h + m - 1) // m * m, (w + m - 1) // m * m
        pad = (0, pw - w, 0, ph - h)
        x0 = F.pad(img0, pad) if any(pad) else img0
        x1 = F.pad(img1, pad) if any(pad) else img1
        div, grid = self.rife._grid(ph, pw)
        timestep = torch.full((1, 1, ph, pw), float(t), dtype=self.dtype, device=self.device)
        flow, mask = self.flow_padded(x0, x1, timestep, div, grid)
        return flow[:, :, :h, :w].float(), torch.sigmoid(mask[:, :, :h, :w].float())

    def flow_padded(self, x0, x1, timestep, div, grid):
        """Dolgulu girdiyle ag govdesi: (akis [1,4,ph,pw], maske logit [1,1,ph,pw]). ONNX/TensorRT de bunu kullanir."""
        net = self.net
        f0, f1 = net.encode(x0), net.encode(x1)
        blocks = (net.block0, net.block1, net.block2, net.block3, net.block4)
        flow = mask = feat = None
        wi0, wi1 = x0, x1
        for i, blk in enumerate(blocks):
            s = net.scale_list[i]
            if flow is None:
                flow, mask, feat = blk(torch.cat((x0, x1, f0, f1, timestep), 1), None, scale=s)
            else:
                wf0 = warp(f0, flow[:, :2], div, grid)
                wf1 = warp(f1, flow[:, 2:4], div, grid)
                fd, mask, feat = blk(torch.cat((wi0, wi1, wf0, wf1, timestep, mask, feat), 1), flow, scale=s)
                flow = flow + fd
            if i < len(blocks) - 1:  # son blokta goruntu kaydirmasina gerek yok
                wi0 = warp(x0, flow[:, :2], div, grid)
                wi1 = warp(x1, flow[:, 2:4], div, grid)
        return flow, mask

    @torch.inference_mode()
    def synthesize(self, hr0: torch.Tensor, hr1: torch.Tensor, flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Dusuk cozunurluklu akis/maske ile yuksek cozunurluklu kareleri kaydirip harmanlar.

        hr0, hr1: [1,C,H,W] (her deger araligi olur). Donus: [1,C,H,W], hr0 ile ayni tip.
        """
        H, W = hr0.shape[2:]
        h, w = flow.shape[2:]
        key = (W / w, H / h)
        if key not in self._scales:
            fx, fy = key
            self._scales[key] = torch.tensor([fx, fy, fx, fy], device=flow.device).view(1, 4, 1, 1)
        fl = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False) * self._scales[key]
        mk = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False).to(hr0.dtype)
        div, grid = self.rife._grid(H, W)
        w0 = warp(hr0, fl[:, :2], div, grid)
        w1 = warp(hr1, fl[:, 2:4], div, grid)
        return w0 * mk + w1 * (1 - mk)
