#!/usr/bin/env python3
"""テーマの見え方を数値ではなく画像で確認するための合成プレビュー。

実際のページのスクリーンショットではない。ページ背景（深海底のグラデーション
＋下からの照り返し）を近似し、`.aw-whale` の CSS（幅・top の式・マスク・不透明度）
を再現して、アートワークをそのまま置いた画像を作る。構図とコントラストの確認用。

使い方:
  ~/AI/ComfyUI/.venv/bin/python simulate.py --art art/canvas.png --scale 1.2 \
      --bottom 500 --mask 6,22,46 --viewport 1440x900 --out sim/theme.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent


def background(w: int, h: int) -> Image.Image:
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    base = np.array([4, 8, 15], dtype=np.float32)
    top = np.array([6, 14, 24], dtype=np.float32)
    t = np.clip(yy * 1.25, 0, 1)
    img = base[None, None, :] + (top - base)[None, None, :] * t[..., None]

    # .aw-upwell: radial-gradient(58% 100% at 50% 100%, rgba(58,168,232,0.18) ...)
    cx, cy = 0.5, 1.0
    dx = (xx - cx) / 0.58
    dy = (yy - cy) / 1.0
    r = np.sqrt(dx * dx + dy * dy)
    glow = np.clip(1.0 - r, 0.0, 1.0) ** 1.6
    up = np.array([58, 168, 232], dtype=np.float32)
    img = img + glow[..., None] * up * 0.16

    # .aw-shafts: repeating-linear-gradient(102deg, ... rgba(130,220,255,0.055) ...)
    ang = np.deg2rad(102.0)
    proj = (xx * np.cos(ang) + yy * np.sin(ang)) * w / max(w, h)
    band = np.clip(np.sin(proj * 2 * np.pi * 4.2), 0, 1) ** 2
    img = img + band[..., None] * np.array([130, 220, 255], dtype=np.float32) * 0.03

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), 'RGB')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--art', default=str(HERE / 'art/canvas.png'))
    ap.add_argument('--scale', type=float, default=1.2)
    ap.add_argument('--bottom', type=float, default=500.0)
    ap.add_argument('--mask', default='6,22,46', help='transparent,0.5,black stops in %%')
    ap.add_argument('--opacity', type=float, default=0.72)
    ap.add_argument('--viewport', default='1440x900')
    ap.add_argument('--out', default=str(HERE / 'sim/theme.png'))
    ap.add_argument('--no-text', action='store_true', help='会話テキストの行を描かない')
    args = ap.parse_args()

    vw, vh = (int(v) for v in args.viewport.split('x'))
    art = Image.open(args.art).convert('RGBA')

    W = min(max(vw, 360), 1900)
    H = W * 560 / 1200
    k = args.bottom / 1200.0
    top = 0.95 * vh - k * W
    left = vw / 2 - W / 2

    art = art.resize((int(round(W)), int(round(H))), Image.LANCZOS)
    a = np.asarray(art).astype(np.float32) / 255.0

    stops = [float(v) for v in args.mask.split(',')]
    yp = np.linspace(0.0, 100.0, a.shape[0])
    m = np.interp(yp, [0.0, stops[0], stops[1], stops[2], 100.0], [0.0, 0.0, 0.5, 1.0, 1.0])
    a[..., 3] *= m[:, None] * args.opacity

    canvas = background(vw, vh).convert('RGBA')
    layer = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), 'RGBA')
    canvas.alpha_composite(layer, (int(round(left)), int(round(top))))

    # 目安: 会話テキストの行を模した薄い矩形（可読性の判断用）
    if not args.no_text:
        d = ImageDraw.Draw(canvas, 'RGBA')
        for i in range(6):
            y = int(vh * 0.18) + i * 46
            d.rounded_rectangle([vw * 0.22, y, vw * 0.22 + vw * (0.35 + 0.04 * (i % 3)), y + 12], 6,
                                fill=(230, 246, 255, 26))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert('RGB').save(out)
    print(f'{out} {canvas.size} W={W:.0f} H={H:.0f} top={top:.0f} artBottomOnScreen={top + (args.bottom / 560) * H:.0f}')


if __name__ == '__main__':
    main()
