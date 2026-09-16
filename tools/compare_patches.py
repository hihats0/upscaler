"""Goz testi (F26): dogrulama yamalarini yan yana PNG yapar (CC kliplerimiz, TOD degil).

Sutunlar: girdi (en yakin komsu x2) | bicubic | modeller... | GT 4K. Her satir bir yama.

Kullanim:
    .venv/Scripts/python.exe tools/compare_patches.py --val val3 --models ft2_best q_amp2_best --out runs/karsilastirma.png
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from third_party.rt4ksr.arch import RT4KSR, clean_checkpoint  # noqa: E402
from train_sr import load_shard, shards  # noqa: E402


def load(name: str) -> torch.nn.Module:
    net = RT4KSR(upscale=2, rep=False).cuda().eval()
    ck = torch.load(os.path.join(ROOT, "weights", "rt4ksr", f"{name}.pth"), map_location="cpu", weights_only=True)
    net.load_state_dict(clean_checkpoint(ck["state_dict"]))
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="val3")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--crop", type=int, default=160, help="4K olcekte gosterilen kare boyu")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lr_all, hr_all = load_shard(shards(args.val)[0])
    # en dokulu yamalar: farki gormek icin
    tex = hr_all[:, ::4, ::4].astype(np.float32).std((1, 2, 3))
    rng = np.random.default_rng(args.seed)
    top = np.argsort(-tex)[: args.n * 6]
    idx = rng.choice(top, args.n, replace=False)
    lr = torch.from_numpy(lr_all[idx]).cuda().permute(0, 3, 1, 2).float() / 255
    cols = {"girdi": F.interpolate(lr, scale_factor=2, mode="nearest"),
            "bicubic": F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)}
    with torch.no_grad():
        for m in args.models:
            cols[m.replace("_best", "")] = load(m)(lr)
    cols["GT 4K"] = torch.from_numpy(hr_all[idx]).cuda().permute(0, 3, 1, 2).float() / 255
    c = args.crop
    o = (lr.shape[-1] * 2 - c) // 2
    rows = []
    for i in range(args.n):
        cells = []
        for t in cols.values():
            img = (t[i, :, o:o + c, o:o + c].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cells.append(cv2.resize(img, (c * 2, c * 2), interpolation=cv2.INTER_NEAREST))
        rows.append(np.concatenate([np.pad(x, ((2, 2), (2, 2), (0, 0)), constant_values=255) for x in cells], 1))
    grid = np.concatenate(rows, 0)
    head = np.full((36, grid.shape[1], 3), 255, np.uint8)
    w = grid.shape[1] // len(cols)
    for k, name in enumerate(cols):
        cv2.putText(head, name, (k * w + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    out = np.concatenate([head, grid], 0)
    cv2.imwrite(os.path.join(ROOT, args.out), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(args.out, out.shape)


if __name__ == "__main__":
    main()
