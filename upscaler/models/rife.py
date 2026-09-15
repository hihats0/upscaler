"""RIFE 4.25 ara kare modeli (Hat 1.2 rakip/yapi tasi).

Mimari: third_party/vsrife (HolyWu/vs-rife, MIT). Agirlik: weights/rife/flownet_v4.25*.pkl
(vs-rife GitHub surumu, upstream hzwer/Practical-RIFE, MIT).

Girdi: [1, 3, H, W] RGB, 0..1. Cikti ayni boyut. Boyut 64'un katina dolgu yapilir.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS = os.path.join(ROOT, "weights", "rife")


def _ifnet_class(version: str):
    if version == "4.25":
        from third_party.vsrife.IFNet_HDv3_v4_25 import IFNet
    elif version == "4.25.lite":
        from third_party.vsrife.IFNet_HDv3_v4_25_lite import IFNet
    else:
        raise ValueError(f"desteklenmeyen RIFE surumu: {version}")
    return IFNet


class Rife:
    def __init__(self, version: str = "4.25", device: str = "cuda", dtype: torch.dtype = torch.float16,
                 scale: float = 1.0) -> None:
        """scale=0.5: akis yarim cozunurlukte hesaplanir (1080p+ icin tipik, ~2-3 kat hizli)."""
        self.version, self.device, self.dtype, self.scale = version, device, dtype, scale
        net = _ifnet_class(version)(scale=scale)
        # En kaba blok girdiyi max(scale_list) kadar kucultup 4 kat daha indirir.
        self.multiple = max(64, int(max(net.scale_list) * 4))
        sd = torch.load(os.path.join(WEIGHTS, f"flownet_v{version}.pkl"), map_location="cpu", weights_only=True)
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        own = net.state_dict()
        missing = [k for k in own if k not in sd]
        if missing:
            raise RuntimeError(f"agirlikta eksik anahtarlar: {missing[:5]}")
        net.load_state_dict({k: v for k, v in sd.items() if k in own})  # caltime vb. kullanilmayanlar atlanir
        self.net = net.to(device=device, dtype=dtype).eval()
        self._grids: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _grid(self, ph: int, pw: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (ph, pw)
        if key not in self._grids:
            div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], dtype=torch.float32, device=self.device)
            hor = torch.linspace(-1.0, 1.0, pw, device=self.device).view(1, 1, 1, pw).expand(1, 1, ph, pw)
            ver = torch.linspace(-1.0, 1.0, ph, device=self.device).view(1, 1, ph, 1).expand(1, 1, ph, pw)
            self._grids[key] = (div, torch.cat([hor, ver], 1).float())
        return self._grids[key]

    def cuda_graph(self, h: int, w: int) -> "GraphedRife":
        """Sabit boyut (h, w) icin CUDA graph ile kaydedilmis cagri."""
        return GraphedRife(self, h, w)

    @torch.inference_mode()
    def interpolate(self, img0: torch.Tensor, img1: torch.Tensor, t: float) -> torch.Tensor:
        """img0, img1: [1,3,H,W] RGB 0..1 (self.dtype). t: 0..1. Donus: [1,3,H,W]."""
        h, w = img0.shape[2:]
        m = self.multiple
        ph, pw = (h + m - 1) // m * m, (w + m - 1) // m * m
        pad = (0, pw - w, 0, ph - h)
        x0 = F.pad(img0, pad) if pad != (0, 0, 0, 0) else img0
        x1 = F.pad(img1, pad) if pad != (0, 0, 0, 0) else img1
        div, grid = self._grid(ph, pw)
        f0 = self.net.encode(x0)
        f1 = self.net.encode(x1)
        timestep = torch.full((1, 1, ph, pw), float(t), dtype=self.dtype, device=self.device)
        out = self.net(x0, x1, timestep, div, grid, f0, f1)
        return out[:, :, :h, :w]


class GraphedRife:
    """Rife.interpolate'in sabit boyutlu CUDA graph kaydi.

    Girdiler statik tamponlara kopyalanir, kayitli graph oynatilir. Donen tensor
    statik cikis tamponudur: bir sonraki cagrida uzerine yazilir (gerekirse clone).
    """

    def __init__(self, rife: Rife, h: int, w: int) -> None:
        self.rife, self.h, self.w = rife, h, w
        m = rife.multiple
        self.ph, self.pw = (h + m - 1) // m * m, (w + m - 1) // m * m
        dev, dt = rife.device, rife.dtype
        self.x0 = torch.zeros((1, 3, self.ph, self.pw), device=dev, dtype=dt)
        self.x1 = torch.zeros_like(self.x0)
        self.ts = torch.zeros((1, 1, self.ph, self.pw), device=dev, dtype=dt)
        div, grid = rife._grid(self.ph, self.pw)
        net = rife.net
        with torch.inference_mode():
            side = torch.cuda.Stream()
            with torch.cuda.stream(side):
                for _ in range(3):  # isinma (cudnn algoritma secimi) graph disinda
                    net(self.x0, self.x1, self.ts, div, grid, net.encode(self.x0), net.encode(self.x1))
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = net(self.x0, self.x1, self.ts, div, grid, net.encode(self.x0), net.encode(self.x1))

    @torch.inference_mode()
    def __call__(self, img0: torch.Tensor, img1: torch.Tensor, t: float) -> torch.Tensor:
        h, w = self.h, self.w
        self.x0[:, :, :h, :w].copy_(img0)
        self.x1[:, :, :h, :w].copy_(img1)
        if self.ph != h or self.pw != w:  # dolgu bolgesi sifir (eager F.pad ile ayni)
            self.x0[:, :, h:, :].zero_(); self.x0[:, :, :, w:].zero_()
            self.x1[:, :, h:, :].zero_(); self.x1[:, :, :, w:].zero_()
        self.ts.fill_(float(t))
        self.graph.replay()
        return self.out[:, :, :h, :w]
