"""EfRLFN mimarisi (MSU Graphics & Media Lab, MIT, LICENSE bu klasorde).

Kaynak: https://github.com/EvgeneyBogatyrev/EfRLFN code/model.py + code/blocks.py
Degisiklik: iki dosya tek dosyada birlestirildi, kullanilmayan yardimcilar
(activation, sequential) sadelestirildi. Katman adlari ayni, ağırlıklar strict yuklenir.
"""
import torch
import torch.nn as nn


def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    padding = (kernel_size - 1) // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size)
    return nn.Sequential(conv, nn.PixelShuffle(upscale_factor))


class ECABlock(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).permute(0, 2, 1)
        y = self.conv(y).permute(0, 2, 1).unsqueeze(-1)
        return x * self.sigmoid(y)


class ERLFB(nn.Module):
    """Efficient Residual Local Feature Block."""

    def __init__(self, in_channels, mid_channels=None, out_channels=None):
        super().__init__()
        mid_channels = mid_channels or in_channels
        out_channels = out_channels or in_channels
        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)
        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.eca = ECABlock()

    def forward(self, x):
        out = torch.tanh(self.c1_r(x))
        out = torch.tanh(self.c2_r(out))
        out = torch.tanh(self.c3_r(out))
        return self.eca(self.c5(out + x))


class EfRLFN(nn.Module):
    """Efficient Residual Local Feature Network. Girdi RGB 0..1, cikti RGB 0..1 (x upscale)."""

    def __init__(self, in_channels=3, out_channels=3, feature_channels=52, upscale=4):
        super().__init__()
        self.conv_1 = conv_layer(in_channels, feature_channels, 3)
        self.block_1 = ERLFB(feature_channels)
        self.block_2 = ERLFB(feature_channels)
        self.block_3 = ERLFB(feature_channels)
        self.block_4 = ERLFB(feature_channels)
        self.block_5 = ERLFB(feature_channels)
        self.block_6 = ERLFB(feature_channels)
        self.conv_2 = conv_layer(feature_channels, feature_channels, 3)
        self.upsampler = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale)

    def forward(self, x):
        f = self.conv_1(x)
        out = self.block_6(self.block_5(self.block_4(self.block_3(self.block_2(self.block_1(f))))))
        return self.upsampler(self.conv_2(out + f))
