"""RT4KSR (Zamfir vd., CVPRW 2023, "Towards Real-Time 4K Image Super-Resolution").

Kaynak: https://github.com/eduardzamfir/RT4KSR code/model/{arch,modules}.py (Apache-2.0,
LICENSE bu klasorde). Degisiklikler (Apache-2.0 madde 4b geregi belirtilir):
- Iki dosya tek dosyada birlestirildi, kullanilmayan yollar (hfb, gamma, ECA) cikarildi
  (orijinal ayarda forget=False, eca_gamma=0: cikti degismez).
- LayerNorm2d autograd.Function yerine duz islemler (ONNX/TensorRT icin), float32'de hesaplanir
  (FP16'da eps=1e-6 sifira yuvarlanir, duz bolgelerde 0/0 olur).
- Bias ile dolgu `pad_tensor` yerine esdeger F.pad(x - b) + b.
- Reparametrizasyon (test.py `reparameterize`) cihazdan bagimsiz `rep_state_dict` fonksiyonu.

Girdi RGB 0..1 [1,3,H,W] (H, W cift), cikti RGB [1,3,H*s,W*s] (kirpilmaz).
Hesap 2x PixelUnshuffle ile yarim cozunurlukte yapilir: 1080p girdide ag 540p'de calisir.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        mu = x32.mean(1, keepdim=True)
        d = x32 - mu
        y = d / (d.pow(2).mean(1, keepdim=True) + self.eps).sqrt()
        y = self.weight.float().view(1, -1, 1, 1) * y + self.bias.float().view(1, -1, 1, 1)
        return y.to(x.dtype)


class ResBlock(nn.Module):
    """Egitim bicimi: 1x1 genislet -> 3x3 (bias dolgulu) + kisa yol -> 1x1 daralt + kisa yol."""

    def __init__(self, n_feats, ratio=2):
        super().__init__()
        m = int(ratio * n_feats)
        self.expand_conv = nn.Conv2d(n_feats, m, 1, 1, 0)
        self.fea_conv = nn.Conv2d(m, m, 3, 1, 0)
        self.reduce_conv = nn.Conv2d(m, n_feats, 1, 1, 0)

    def forward(self, x):
        out = self.expand_conv(x)
        identity = out
        b0 = self.expand_conv.bias.view(1, -1, 1, 1)
        out = F.pad(out - b0, (1, 1, 1, 1)) + b0
        out = self.fea_conv(out) + identity
        return self.reduce_conv(out) + x


class RepResBlock(nn.Module):
    """Cikarim bicimi: ResBlock'un tek 3x3 konvolusyona birlestirilmis hali."""

    def __init__(self, n_feats):
        super().__init__()
        self.rep_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

    def forward(self, x):
        return self.rep_conv(x)


class NAFBlock(nn.Module):
    """SimplifiedRepNAFBlock (layernorm=True, residual=False, eca yok)."""

    def __init__(self, n_feats, rep):
        super().__init__()
        self.conv1 = RepResBlock(n_feats) if rep else ResBlock(n_feats, 2)
        self.norm = LayerNorm2d(n_feats)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv1(self.norm(x)))


class RT4KSR(nn.Module):
    def __init__(self, num_feats=24, num_blocks=4, upscale=2, rep=True):
        super().__init__()
        self.down = nn.PixelUnshuffle(2)
        self.head = nn.Sequential(nn.Conv2d(3 * 4, num_feats, 3, padding=1))
        self.body = nn.Sequential(*[NAFBlock(num_feats, rep) for _ in range(num_blocks)])
        self.tail = nn.Sequential(LayerNorm2d(num_feats),
                                  RepResBlock(num_feats) if rep else ResBlock(num_feats, 2))
        self.upsample = nn.Sequential(nn.Conv2d(num_feats, 3 * (2 * upscale) ** 2, 3, padding=1),
                                      nn.PixelShuffle(2 * upscale))

    def forward(self, x):
        return self.upsample(self.tail(self.body(self.head(self.down(x)))))


def clean_checkpoint(state_dict):
    """DataParallel oneki ve kullanilmayan (hfb, gamma) anahtarlari atar."""
    sd = {k.removeprefix("module."): v for k, v in state_dict.items()}
    return {k: v for k, v in sd.items() if not (k == "gamma" or k.startswith("hfb."))}


@torch.no_grad()
def rep_state_dict(sd):
    """Egitim bicimi agirliklari (expand/fea/reduce) -> rep_conv (tek 3x3). float64'te birlestirir."""
    out = {}
    for k, v in sd.items():
        if k.endswith("expand_conv.weight"):
            p = k[: -len("expand_conv.weight")]
            k0, b0 = sd[p + "expand_conv.weight"].double(), sd[p + "expand_conv.bias"].double()
            k1, b1 = sd[p + "fea_conv.weight"].double().clone(), sd[p + "fea_conv.bias"].double()
            k2, b2 = sd[p + "reduce_conv.weight"].double(), sd[p + "reduce_conv.bias"].double()
            mid, n = k0.shape[:2]
            idx = torch.arange(mid)
            k1[idx, idx, 1, 1] += 1.0  # 3x3 etrafindaki kisa yol
            k01 = F.conv2d(k1, k0.permute(1, 0, 2, 3))
            b01 = F.conv2d(b0.view(1, -1, 1, 1) * torch.ones(1, mid, 3, 3, dtype=torch.float64), k1, b1)
            k012 = F.conv2d(k01.permute(1, 0, 2, 3), k2).permute(1, 0, 2, 3).contiguous()
            b012 = F.conv2d(b01, k2, b2).view(-1)
            idx = torch.arange(n)
            k012[idx, idx, 1, 1] += 1.0  # blogun genel kisa yolu
            out[p + "rep_conv.weight"] = k012.float()
            out[p + "rep_conv.bias"] = b012.float()
        elif any(s in k for s in ("expand_conv.", "fea_conv.", "reduce_conv.")):
            continue
        else:
            out[k] = v
    return out
