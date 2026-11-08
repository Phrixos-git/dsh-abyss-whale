#!/usr/bin/env python3
"""ComfyUI で描き直した鯨に、周りの光（箱・光の筋・ボケ）を足して仕上げる。

generate_neon.py の出力は黒地に発光する鯨。これを輝度で抜いて透明にし、
参考画像にある「ワイヤーフレームの箱」「流れる光の筋」「水中のボケ」を
neon.py と同じ描き方でまわりに足し、テーマ用の 1440x672 RGBA にまとめる。

使い方:
  ~/AI/ComfyUI/.venv/bin/python neon_post.py --src out/neon85_s11.png \
      --out art/neon-render.png --webp art/neon-render-q88.webp --webp-quality 88
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter, label

import neon
from neon import (C_BOKEH, C_CUBE, C_HOT, C_MID, C_NET, C_WAVE, SS, Layer,
                  cube_segments, jitter_grid, over, raster_polylines,
                  raster_polys, raster_segments, sample, smoothstep, stamp,
                  wave_polylines)

HERE = Path(__file__).resolve().parent
CANVAS_W, CANVAS_H = 1440, 672


def body_mask(L: np.ndarray) -> np.ndarray:
    """鯨のシルエット（輝度のしきい値 → 穴埋め → 最大連結成分 → 拡張 → 平滑化）。"""
    m = L > 0.050
    m = binary_fill_holes(m)
    lab, _ = label(m)
    if lab.max() > 0:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        m = lab == int(sizes.argmax())
    m = binary_dilation(m, np.ones((5, 5), bool), iterations=2)
    m = gaussian_filter(m.astype(np.float32), 2.0 * SS)
    return smoothstep(m, 0.35, 0.65)


def blue_ramp(L: np.ndarray) -> np.ndarray:
    """発光色のランプ（深い藍 → 青 → シアン → 白）"""
    t = np.clip(L * 1.12, 0.0, 1.0) ** 0.85
    lo = np.array([0.008, 0.030, 0.090], np.float32)
    mid = np.array([0.045, 0.240, 0.620], np.float32)
    hi = np.array([0.380, 0.780, 1.000], np.float32)
    top = np.array([0.760, 0.960, 1.000], np.float32)
    a = smoothstep(t, 0.0, 0.35)[..., None]
    b = smoothstep(t, 0.35, 0.72)[..., None]
    c = smoothstep(t, 0.72, 1.0)[..., None]
    out = lo + (mid - lo) * a
    out = out + (hi - out) * b
    out = out + (top - out) * c
    return out.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', default=str(HERE / 'art/neon-render.png'))
    ap.add_argument('--webp', default='')
    ap.add_argument('--webp-quality', type=int, default=88)
    ap.add_argument('--webp-alpha-quality', type=int, default=100)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--power', type=float, default=1.0, help='全体の明るさ')
    ap.add_argument('--solid', type=float, default=0.86, help='体を不透明にする強さ')
    ap.add_argument('--cubes', type=int, default=14)
    ap.add_argument('--front-cubes', type=int, default=4)
    ap.add_argument('--waves', type=int, default=22)
    ap.add_argument('--front-waves', type=int, default=10)
    ap.add_argument('--bokeh', type=int, default=220)
    ap.add_argument('--clusters', type=int, default=10)
    ap.add_argument('--exposure', type=float, default=1.10)
    ap.add_argument('--out-scale', type=float, default=1.0, help='1.0 で 1440x672、0.833 で 1200x560')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    src = Image.open(args.src).convert('RGB')
    rgb = np.asarray(src).astype(np.float32) / 255.0
    L = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    w, h = CANVAS_W * SS, CANVAS_H * SS
    rgb = np.stack([neon.resize_f(rgb[..., i], (w, h)) for i in range(3)], axis=-1)
    L = np.stack([neon.resize_f(L, (w, h))], axis=-1)[..., 0]

    # 体は「シルエット × 低いしきい値」、外側は「明るい発光だけ」を残す。
    # こうすると黒地の細かいノイズ（ススのような粒）が透明に落ちる。
    solid = body_mask(L)
    key_body = smoothstep(gaussian_filter(L, 1.0 * SS), 0.012, 0.050)
    key_glow = smoothstep(gaussian_filter(L, 1.0 * SS), 0.110, 0.320)
    alpha = np.maximum(solid * key_body * args.solid, key_glow)
    alpha = np.clip(alpha, 0.0, 1.0)

    # 色は青のランプへ寄せて、灰色の煙を青い光にする
    ramp = blue_ramp(L)
    rgb = np.clip(rgb * 0.38 + ramp * 0.62, 0.0, 1.0)
    # 彩度を上げて、白っぽい面もシアンに寄せる
    lum = neon.luminance(rgb)[..., None]
    rgb = np.clip((rgb - lum) * 1.28 + lum, 0.0, 1.0)
    gains = np.array([0.80, 0.97, 1.18], np.float32)
    rgb = np.clip(rgb * gains, 0.0, 1.0)
    # 局所コントラスト: なめらかな面を沈め、粒と網を浮かせる
    smooth = gaussian_filter(rgb, (2.2, 2.2, 0.0))
    rgb = np.clip(smooth * 0.70 + (rgb - smooth) * 1.42, 0.0, 1.0)
    rgb = rgb ** 1.04

    whale = Layer(w, h)
    whale.em += rgb * alpha[..., None]
    whale.a += alpha

    # なめらかな面に、細かい光の粒を足して「粒子でできている」感じを強める
    sgx, sgy = jitter_grid(w, h, 3.0 * SS, rng)
    ssel = (sample(solid, sgx, sgy) > 0.5) & (rng.random(sgx.size) < 0.55)
    sv = (0.10 + 0.60 * np.clip(sample(L, sgx, sgy) * 1.8, 0.0, 1.0) ** 1.2).astype(np.float32)
    sm = np.zeros((h, w), np.float32)
    stamp(sm, sgy[ssel], sgx[ssel], sv[ssel], 0.9 * SS)
    whale.add(sm, np.array([0.70, 0.93, 1.00], np.float32), 1.0, 0.60)

    # ------------------------------------------------------------------
    # まわりの光（背景）
    # ------------------------------------------------------------------
    bg = Layer(w, h)

    n = args.bokeh
    bx = rng.uniform(0, w, n).astype(np.float32)
    by = rng.uniform(0, h, n).astype(np.float32)
    br = rng.uniform(1.2, 12.0, n).astype(np.float32) * SS
    bi = rng.uniform(0.02, 0.11, n).astype(np.float32) * (0.35 + 0.65 * np.clip(by / h, 0, 1))
    for lo, hi, sig in ((1.2, 3.0, 0.9), (3.0, 7.0, 2.0), (7.0, 13.0, 4.5)):
        sel = (br >= lo * SS) & (br < hi * SS)
        if sel.any():
            m = np.zeros((h, w), np.float32)
            stamp(m, by[sel], bx[sel], bi[sel], sig * SS)
            bg.add(gaussian_filter(m, 0.6), C_BOKEH, 1.0, 0.9)

    for _ in range(args.clusters):
        cxx = rng.uniform(0.03 * w, 0.97 * w)
        cyy = rng.uniform(0.36 * h, 0.96 * h)
        cnt = int(rng.integers(3, 6))
        apts = [(cxx + rng.normal(0, 24 * SS), cyy + rng.normal(0, 20 * SS)) for _ in range(cnt)]
        segs = []
        for i in range(cnt):
            for j in range(i + 1, cnt):
                if rng.random() < 0.6 and math.dist(apts[i], apts[j]) < 100 * SS:
                    segs.append((apts[i][0], apts[i][1], apts[j][0], apts[j][1]))
        if segs:
            bg.add(raster_segments(segs, (w, h), 2), C_NET, 0.20, 0.26)
        m = np.zeros((h, w), np.float32)
        stamp(m, np.array([p[1] for p in apts], np.float32),
              np.array([p[0] for p in apts], np.float32),
              np.full(cnt, 0.55, np.float32), 1.7 * SS)
        bg.add(m, C_HOT, 0.55, 0.8)

    for pts, g in wave_polylines(rng, w, h, args.waves, 0.50, 0.99, 2.5, 9.0):
        bg.add(raster_polylines([pts], (w, h), 2), C_WAVE, 0.15 * g, 0.22 * g)

    cube_specs = []
    for _ in range(args.cubes):
        cxx = rng.uniform(0.02, 0.98) * w
        cyy = rng.uniform(0.42, 0.97) * h
        s = float(np.exp(rng.uniform(math.log(6 * SS), math.log(30 * SS))))
        yaw = rng.uniform(0.0, math.pi)
        cube_specs.append((cxx, cyy, s, yaw, 0.32 + 0.68 * rng.uniform(0.0, 1.0)))
    face_polys = []
    for cxx, cyy, s, yaw, _g in cube_specs:
        _segs, faces = cube_segments(cxx, cyy, s, yaw)
        face_polys.extend(faces)
    bg.add(raster_polys(face_polys, (w, h)), C_CUBE, 0.035, 0.05)
    for cxx, cyy, s, yaw, g in cube_specs:
        segs, _faces = cube_segments(cxx, cyy, s, yaw)
        bg.add(raster_segments(segs, (w, h), 2), C_CUBE, 0.24 * g, 0.28 * g)

    # ------------------------------------------------------------------
    # 手前の光
    # ------------------------------------------------------------------
    fg = Layer(w, h)
    for pts, g in wave_polylines(rng, w, h, args.front_waves, 0.74, 0.99, 1.5, 6.0):
        fg.add(raster_polylines([pts], (w, h), 2), C_WAVE, 0.26 * g, 0.34 * g)
    for _ in range(args.front_cubes):
        cxx = rng.uniform(0.02, 0.98) * w
        cyy = rng.uniform(0.80, 0.96) * h
        s = float(np.exp(rng.uniform(math.log(7 * SS), math.log(20 * SS))))
        yaw = rng.uniform(0.0, math.pi)
        segs, _faces = cube_segments(cxx, cyy, s, yaw)
        fg.add(raster_segments(segs, (w, h), 2), C_CUBE, 0.30, 0.36)

    comp = over(whale, over(fg, bg))
    comp.a = np.clip(comp.a, 0.0, 1.0)

    em = np.stack([neon.resize_f(comp.em[..., i], (CANVAS_W, CANVAS_H)) for i in range(3)], axis=-1)
    al = np.clip(neon.resize_f(comp.a, (CANVAS_W, CANVAS_H)), 0.0, 1.0)

    glow = (gaussian_filter(em, (1.3, 1.3, 0.0)) * 0.55
            + gaussian_filter(em, (4.0, 4.0, 0.0)) * 0.28
            + gaussian_filter(em, (12.0, 12.0, 0.0)) * 0.10)
    em = em + glow * 0.5
    al = np.clip(al + neon.luminance(glow) * 0.30, 0.0, 1.0)
    al = np.clip((al - 0.012) / 0.988, 0.0, 1.0)

    rgb_out = np.clip(em / np.maximum(al, 1e-4)[..., None], 0.0, 1.0)
    rgb_out = 1.0 - np.exp(-rgb_out * args.exposure * 2.1)
    rgb_out = rgb_out ** 1.03
    rgb_out[al < 0.02] = 0.0
    # 左右の端は要素の縁で切れないよう、ごく薄くフェードする
    xf = np.linspace(0.0, 1.0, CANVAS_W, dtype=np.float32)
    fade = smoothstep(xf, 0.0, 0.035) * smoothstep(1.0 - xf, 0.0, 0.035)
    al = al * fade[None, :]

    out = Image.fromarray((np.clip(np.dstack([rgb_out, al]), 0, 1) * 255).astype(np.uint8), 'RGBA')
    if args.out_scale != 1.0:
        ow = int(round(CANVAS_W * args.out_scale))
        oh = int(round(CANVAS_H * args.out_scale))
        out = out.resize((ow, oh), Image.LANCZOS)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    ys, xs = np.nonzero(al > 0.02)
    print(f'png {out_path} {out_path.stat().st_size} B  bbox x {xs.min()}..{xs.max()} y {ys.min()}..{ys.max()}')

    if args.webp:
        wp = Path(args.webp)
        out.save(wp, 'WEBP', quality=args.webp_quality, method=6,
                 alpha_quality=args.webp_alpha_quality, exact=True)
        print(f'webp {wp} {wp.stat().st_size} B  base64 {math.ceil(wp.stat().st_size / 3) * 4} B')


if __name__ == '__main__':
    main()
