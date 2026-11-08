#!/usr/bin/env python3
"""ComfyUI の生成画像をテーマ用の光のアートワークに変換する。

1. 黒背景を輝度で抜いてアルファを作る（体はほぼ不透明、縁は光として残す）
2. 発光（ブルーム）と寒色方向の色付けを足す
3. 1200x560 の viewBox に合わせた配置で RGBA キャンバスを作る
4. WebP（アルファ付き）に書き出してサイズを確認する

使い方:
  ~/AI/ComfyUI/.venv/bin/python build.py <src.png> [--scale 1.2] [--bottom 500]
                                          [--length 1076] [--webp out.webp]

`--bottom` は viewBox 内での絵の下端 y、`--length` は絵の横幅（viewBox 単位）。
CSS の top: calc(95% - min(k*100vw, ...)) の k は bottom/1200 になる。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
VIEW_W, VIEW_H = 1200, 560


def smoothstep(x: np.ndarray, a: float, b: float) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def key_alpha(rgb: np.ndarray, lo: float = 0.010, hi: float = 0.052) -> np.ndarray:
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return smoothstep(luma, lo, hi)


def bloom(rgb: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    bright = np.clip((luma - 0.42) / 0.58, 0.0, 1.0) ** 1.5
    src = Image.fromarray((np.clip(bright, 0, 1) * 255).astype(np.uint8), 'L')
    wide = np.asarray(src.filter(ImageFilter.GaussianBlur(26))).astype(np.float32) / 255.0
    tight = np.asarray(src.filter(ImageFilter.GaussianBlur(5))).astype(np.float32) / 255.0
    glow = np.clip(wide * 0.55 + tight * 0.45, 0.0, 1.0)
    tint = np.array([0.55, 0.86, 1.0], dtype=np.float32)
    out = rgb + glow[..., None] * tint * 0.42
    return np.clip(out, 0.0, 1.0), np.clip(alpha + glow * 0.85, 0.0, 1.0)


def edge_glow(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    # 半透明の縁は「暗い汚れ」になりやすいので、光の色へ寄せる。
    # 不透明な体（alpha 1）はそのまま、輪郭の外側はシアンの光になる。
    t = smoothstep(alpha, 0.10, 0.62)[..., None]
    tint = np.array([0.26, 0.58, 0.86], dtype=np.float32)
    return np.clip(rgb * t + tint * (1.0 - t), 0.0, 1.0)


def grade(rgb: np.ndarray, gamma: float = 1.0, lift: float = 0.02) -> np.ndarray:
    # 寒色方向へ寄せ、ごく浅く持ち上げて「水の中で光っている」感じにする。
    gains = np.array([0.86, 0.98, 1.06], dtype=np.float32)
    out = np.clip(rgb * gains, 0.0, 1.0)
    out = out + (1.0 - out) * lift * np.array([0.35, 0.7, 1.0], dtype=np.float32)
    return np.clip(out, 0.0, 1.0) ** gamma


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--scale', type=float, default=1.2)
    ap.add_argument('--bottom', type=float, default=500.0)
    ap.add_argument('--length', type=float, default=1076.0)
    ap.add_argument('--center', type=float, default=616.0)
    ap.add_argument('--crop-dx', type=float, default=0.0)
    ap.add_argument('--crop-frac', type=float, default=1.0, help='右端から残す割合（尾を詰める）')
    ap.add_argument('--out', default=str(HERE / 'art/canvas.png'))
    ap.add_argument('--webp', default='')
    ap.add_argument('--webp-quality', type=int, default=72)
    ap.add_argument('--gamma', type=float, default=1.0, help='体を暗くする（1より大きいと暗い）')
    ap.add_argument('--lift', type=float, default=0.02, help='暗部の持ち上げ')
    ap.add_argument('--tail-fade', type=float, default=0.0, help='左端（尾側）を透明へ落とす幅（割合）')
    args = ap.parse_args()

    im = Image.open(args.src).convert('RGB')
    rgb = np.asarray(im).astype(np.float32) / 255.0

    if args.crop_frac < 1.0:
        keep = int(rgb.shape[1] * args.crop_frac)
        rgb = rgb[:, :keep]
    alpha = key_alpha(rgb)
    rgb, alpha = bloom(rgb, alpha)
    rgb = grade(rgb, args.gamma, args.lift)
    rgb = edge_glow(rgb, alpha)

    if args.tail_fade > 0:
        w0 = rgb.shape[1]
        xr = np.linspace(0.0, 1.0, w0, dtype=np.float32)
        fade = np.clip(xr / max(args.tail_fade, 1e-6), 0.0, 1.0) ** 1.2
        alpha = alpha * (0.12 + 0.88 * fade)[None, :]

    # 内容の外接矩形
    ys, xs = np.where(alpha > 0.02)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    rgb = rgb[y0:y1 + 1, x0:x1 + 1]
    alpha = alpha[y0:y1 + 1, x0:x1 + 1]

    pad = 12
    rgb = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)))
    alpha = np.pad(alpha, ((pad, pad), (pad, pad)))

    h, w = alpha.shape
    target = int(round(args.length * args.scale))
    scale = target / float(w)
    new_w, new_h = target, int(round(h * scale))
    rgba = np.dstack([rgb, alpha])
    img = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), 'RGBA')
    img = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new('RGBA', (int(round(VIEW_W * args.scale)), int(round(VIEW_H * args.scale))), (0, 0, 0, 0))
    cx = args.center * args.scale + args.crop_dx * args.scale
    by = args.bottom * args.scale
    canvas.alpha_composite(img, (int(round(cx - new_w / 2)), int(round(by - new_h))))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)

    # 実測値（viewBox 単位）
    a = np.asarray(canvas)[..., 3] / 255.0
    ys2, xs2 = np.where(a > 0.02)
    bx0, bx1 = xs2.min() / args.scale, xs2.max() / args.scale
    by0, by1 = ys2.min() / args.scale, ys2.max() / args.scale
    print(f'canvas {canvas.size} -> viewBox bbox x {bx0:.1f}..{bx1:.1f} y {by0:.1f}..{by1:.1f}')
    print(f'length {bx1 - bx0:.1f}  height {by1 - by0:.1f}  bottom {by1:.1f}')
    print(f'CSS k = bottom/1200 = {by1 / 1200:.4f}  -> min({by1 / 12:.1f}vw, {by1 * 1900 / 1200:.0f}px)')
    print(f'element mask: top {(by0 / VIEW_H) * 100:.1f}%  bottom {(by1 / VIEW_H) * 100:.1f}%')

    if args.webp:
        wp = Path(args.webp)
        canvas.save(wp, 'WEBP', quality=args.webp_quality, method=6, alpha_quality=100, exact=True)
        print(f'webp {wp} {wp.stat().st_size} B  base64 {math.ceil(wp.stat().st_size / 3) * 4} B')


if __name__ == '__main__':
    main()
