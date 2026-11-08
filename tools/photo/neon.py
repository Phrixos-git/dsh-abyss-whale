#!/usr/bin/env python3
"""写真版の大鯨を「発光する粒子と電子の網」で描き直す。

参考イメージ（発光する点の群れ＋星座のような細い網＋ワイヤーフレームの箱＋
流れる光の筋）に寄せる。写真レンダーの形と陰影はそのまま使い、描き方だけを
ネオンの点描へ置き換える。

1. 入力 canvas.png（build.py が置いた写真版）から輝度 L とマスク A を取る
2. 体の内側に、明るい所ほど密な発光粒子をまく（暗い藍 → シアン → 白）
3. 輪郭は等輝度リングを光らせ、その上にも粒を置く（点線の輪郭）
4. 体の上にドロネー網を引き、頂点に光点を置く（星座の網）
5. 背景にワイヤーフレームの箱、流れる光の筋、ボケ、星座のかけらを描く
6. 発光を多段ブルームでにじませ、1440x672（viewBox 1200x560 の 1.2 倍）に収める

使い方（全オプションは --help）:
  ~/AI/ComfyUI/.venv/bin/python neon.py --seed 7 \
      --out art/neon.png --webp art/neon-q88.webp --webp-quality 88
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, label
from scipy.spatial import Delaunay

HERE = Path(__file__).resolve().parent
VIEW_W, VIEW_H = 1200, 560
CANVAS_W, CANVAS_H = 1440, 672          # viewBox 1200x560 の 1.2 倍（既存の絵と同じ）
SS = 2                                  # 内部のスーパーサンプル倍率

# 色（0..1 の RGB。すべて青〜シアンに寄せる）
C_DIM = np.array([0.10, 0.32, 0.92], np.float32)
C_MID = np.array([0.22, 0.64, 1.00], np.float32)
C_HOT = np.array([0.90, 0.99, 1.00], np.float32)
C_RIM = np.array([0.36, 0.84, 1.00], np.float32)
C_NET = np.array([0.18, 0.56, 1.00], np.float32)
C_CUBE = np.array([0.24, 0.80, 1.00], np.float32)
C_WAVE = np.array([0.16, 0.58, 1.00], np.float32)
C_BOKEH = np.array([0.36, 0.78, 1.00], np.float32)
C_FILL_LO = np.array([0.008, 0.022, 0.070], np.float32)
C_FILL_HI = np.array([0.055, 0.195, 0.440], np.float32)
C_BODY_GLOW = np.array([0.06, 0.24, 0.62], np.float32)


def smoothstep(x: np.ndarray, a: float, b: float) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def resize_f(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """float32 の配列を PIL の LANCZOS でリサイズする。"""
    img = Image.fromarray(arr.astype(np.float32), mode='F')
    return np.asarray(img.resize(size, Image.LANCZOS)).astype(np.float32)


def resize_rgb(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.stack([resize_f(arr[..., i], size) for i in range(arr.shape[2])], axis=-1)


def jitter_grid(w: int, h: int, cell: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """セルごとに 1 点をランダムに置く（格子状のムラを消す）。"""
    xs = np.arange(cell * 0.5, w, cell, dtype=np.float32)
    ys = np.arange(cell * 0.5, h, cell, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx + rng.uniform(-cell * 0.45, cell * 0.45, gx.shape).astype(np.float32)
    gy = gy + rng.uniform(-cell * 0.45, cell * 0.45, gy.shape).astype(np.float32)
    return gx.ravel(), gy.ravel()


def sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = field.shape
    ix = np.clip(x.astype(np.int64), 0, w - 1)
    iy = np.clip(y.astype(np.int64), 0, h - 1)
    return field[iy, ix]


def stamp(acc: np.ndarray, ys: np.ndarray, xs: np.ndarray, vals: np.ndarray, sigma: float) -> None:
    """サブピクセル位置にガウスの点を打つ（acc に加算）。"""
    if ys.size == 0:
        return
    h, w = acc.shape
    rad = int(max(1, math.ceil(sigma * 2.2)))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    kern = np.exp(-((yy * yy + xx * xx) / (2.0 * sigma * sigma))).astype(np.float32)
    kern /= max(float(kern.sum()), 1e-6)

    y0 = ys - 0.5
    x0 = xs - 0.5
    iy = np.floor(y0).astype(np.int64)
    ix = np.floor(x0).astype(np.int64)
    fy = (y0 - iy).astype(np.float32)
    fx = (x0 - ix).astype(np.float32)
    corners = (((1.0 - fy) * fx, 0, 1), ((1.0 - fy) * (1.0 - fx), 0, 0),
               (fy * fx, 1, 1), (fy * (1.0 - fx), 1, 0))

    for wgt, dy, dx in corners:
        py = iy + dy
        px = ix + dx
        ok = (py >= 0) & (py < h) & (px >= 0) & (px < w)
        wy, wx, wv = py[ok], px[ok], vals[ok] * wgt[ok]
        for ky in range(-rad, rad + 1):
            ty = wy + ky
            ty_ok = (ty >= 0) & (ty < h)
            if not ty_ok.any():
                continue
            for kx in range(-rad, rad + 1):
                tx = wx + kx
                t_ok = ty_ok & (tx >= 0) & (tx < w)
                if not t_ok.any():
                    continue
                np.add.at(acc, (ty[t_ok], tx[t_ok]), wv[t_ok] * kern[ky + rad, kx + rad])


def raster_segments(segs: list[tuple[float, ...]], size: tuple[int, int], width: int = 2) -> np.ndarray:
    """線分の集まりを 0..1 のマスクにする（重なりは 1 で飽和）。"""
    img = Image.new('L', size, 0)
    d = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in segs:
        d.line([(x0, y0), (x1, y1)], fill=255, width=width)
    return np.asarray(img, dtype=np.float32) / 255.0


def raster_polylines(polys: list[list[tuple[float, float]]], size: tuple[int, int], width: int = 2) -> np.ndarray:
    img = Image.new('L', size, 0)
    d = ImageDraw.Draw(img)
    for p in polys:
        if len(p) > 1:
            d.line(p, fill=255, width=width, joint='curve')
    return np.asarray(img, dtype=np.float32) / 255.0


def raster_polys(polys: list[list[tuple[float, float]]], size: tuple[int, int]) -> np.ndarray:
    img = Image.new('L', size, 0)
    d = ImageDraw.Draw(img)
    for p in polys:
        if len(p) > 2:
            d.polygon(p, fill=255)
    return np.asarray(img, dtype=np.float32) / 255.0


class Layer:
    """発光レイヤー（加算の発光色 em と、遮蔽のための alpha a）。"""

    def __init__(self, w: int, h: int):
        self.em = np.zeros((h, w, 3), np.float32)
        self.a = np.zeros((h, w), np.float32)

    def add(self, mask: np.ndarray, color, gain: float = 1.0, again: float = 1.0) -> None:
        self.em += mask[..., None] * np.asarray(color, np.float32)[None, None, :] * gain
        self.a += mask * again


def over(top: Layer, bottom: Layer) -> Layer:
    out = Layer(top.em.shape[1], top.em.shape[0])
    ta = np.clip(top.a, 0.0, 1.0)
    out.em = top.em + bottom.em * (1.0 - ta)[..., None]
    out.a = ta + bottom.a * (1.0 - ta)
    return out


def cube_segments(cx: float, cy: float, s: float, yaw: float, tilt: float = 0.42) -> tuple[list, list]:
    """ワイヤーフレームの箱。z が縦になるアクソノメトリック投影。"""
    v = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], np.float32)
    ca, sa = math.cos(yaw), math.sin(yaw)
    x = v[:, 0] * ca - v[:, 1] * sa
    y = v[:, 0] * sa + v[:, 1] * ca
    px = cx + x * s
    py = cy + y * s * tilt - v[:, 2] * s * 0.80
    pts = np.stack([px, py], axis=1)
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    segs = [(pts[a, 0], pts[a, 1], pts[b, 0], pts[b, 1]) for a, b in edges]
    faces = [[tuple(pts[i]) for i in (0, 2, 6, 4)],
             [tuple(pts[i]) for i in (1, 3, 7, 5)],
             [tuple(pts[i]) for i in (0, 1, 5, 4)],
             [tuple(pts[i]) for i in (2, 3, 7, 6)]]
    return segs, faces


def wave_polylines(rng: np.random.Generator, w: int, h: int, count: int, y_lo: float, y_hi: float,
                   amp_lo: float, amp_hi: float) -> list[tuple[list[tuple[float, float]], float]]:
    """横に流れる光の筋。太さ（強さ）と細さ（線幅）も持たせる。"""
    out = []
    for _ in range(count):
        base = rng.uniform(y_lo, y_hi) * h
        amp = rng.uniform(amp_lo, amp_hi) * SS
        lam = rng.uniform(90.0, 420.0) * SS
        phase = rng.uniform(0.0, 2.0 * math.pi)
        tilt = rng.uniform(-0.010, 0.010) * w
        xs = np.linspace(-0.05 * w, 1.05 * w, 260)
        ys = (base + amp * np.sin(2.0 * math.pi * xs / lam + phase)
              + amp * 0.45 * np.sin(2.0 * math.pi * xs / (lam * 2.3) + phase * 1.7 + 1.1)
              + tilt * (xs / w - 0.5))
        out.append((list(zip(xs, ys)), 0.35 + 0.65 * rng.random()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(HERE / 'art/canvas.png'))
    ap.add_argument('--out', default=str(HERE / 'art/neon.png'))
    ap.add_argument('--webp', default='')
    ap.add_argument('--webp-quality', type=int, default=88)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--bokeh', type=int, default=280)
    ap.add_argument('--dust', type=int, default=300)
    ap.add_argument('--sparks', type=int, default=1200)
    ap.add_argument('--clusters', type=int, default=18)
    ap.add_argument('--cubes', type=int, default=16)
    ap.add_argument('--front-cubes', type=int, default=4)
    ap.add_argument('--waves', type=int, default=20)
    ap.add_argument('--front-waves', type=int, default=8)
    ap.add_argument('--glow', type=float, default=1.0)
    ap.add_argument('--exposure', type=float, default=1.20)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    src = Image.open(args.src).convert('RGBA')
    big = src.resize((CANVAS_W * SS, CANVAS_H * SS), Image.LANCZOS)
    arr = np.asarray(big).astype(np.float32) / 255.0
    rgb, alpha = arr[..., :3], arr[..., 3]
    L = luminance(rgb)
    w, h = arr.shape[1], arr.shape[0]
    Ln = L / max(float(L.max()), 1e-6)

    # 体（写真では alpha=1 が不透明な体。縁の発光は alpha が薄いので切る）
    # 口の線や眼のまわりなど、写真で暗く落ちた所は alpha が抜けているので、
    # 穴を埋めて 1 つのシルエットにしてから滑らかにする（黒い裂け目を残さない）。
    solid = (alpha > 0.80)
    solid = binary_fill_holes(solid)
    lab, _ = label(solid)
    if lab.max() > 0:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        solid = lab == int(sizes.argmax())
    solid = binary_closing(solid, np.ones((3, 3), bool), iterations=3)
    body = gaussian_filter(solid.astype(np.float32), 2.2 * SS)
    body = smoothstep(body, 0.38, 0.62)
    blur_L = gaussian_filter(Ln, 2.2 * SS)
    # 写真のしわ・溝を線ではなく点で拾うためのエッジ強度
    eg = np.hypot(*np.gradient(blur_L))
    eg = gaussian_filter(eg, 0.8 * SS)
    eg = eg / max(float(eg.max()), 1e-6)
    # 局所コントラスト（暗い側面のしわ・斑も拾えるようにする）
    hp = Ln - gaussian_filter(Ln, 16.0 * SS)
    detail = np.clip(hp * 7.0 + 0.30, 0.0, 1.0)
    detail = gaussian_filter(detail, 0.8 * SS)

    # ------------------------------------------------------------------
    # 背景レイヤー（光の筋・箱・ボケ・かけら）
    # ------------------------------------------------------------------
    bg = Layer(w, h)

    n = args.bokeh
    bx = rng.uniform(0, w, n).astype(np.float32)
    by = rng.uniform(0, h, n).astype(np.float32)
    br = rng.uniform(1.2, 13.0, n).astype(np.float32) * SS
    bi = rng.uniform(0.02, 0.12, n).astype(np.float32) * (0.35 + 0.65 * np.clip(by / h, 0, 1))
    for lo, hi, sig in ((1.2, 3.0, 0.9), (3.0, 7.0, 2.0), (7.0, 14.0, 4.5)):
        sel = (br >= lo * SS) & (br < hi * SS)
        if sel.any():
            m = np.zeros((h, w), np.float32)
            stamp(m, by[sel], bx[sel], bi[sel], sig * SS)
            bg.add(gaussian_filter(m, 0.6), C_BOKEH, 1.0, 0.9)

    for _ in range(args.clusters):
        cxx = rng.uniform(0.04 * w, 0.96 * w)
        cyy = rng.uniform(0.34 * h, 0.96 * h)
        cnt = int(rng.integers(3, 7))
        apts = [(cxx + rng.normal(0, 26 * SS), cyy + rng.normal(0, 22 * SS)) for _ in range(cnt)]
        segs = []
        for i in range(cnt):
            for j in range(i + 1, cnt):
                if rng.random() < 0.55 and math.dist(apts[i], apts[j]) < 110 * SS:
                    segs.append((apts[i][0], apts[i][1], apts[j][0], apts[j][1]))
        if segs:
            bg.add(raster_segments(segs, (w, h), 2), C_NET, 0.22, 0.30)
        m = np.zeros((h, w), np.float32)
        stamp(m, np.array([p[1] for p in apts], np.float32),
              np.array([p[0] for p in apts], np.float32),
              np.full(cnt, 0.6, np.float32), 1.8 * SS)
        bg.add(m, C_HOT, 0.6, 0.85)

    for pts, g in wave_polylines(rng, w, h, args.waves, 0.50, 0.99, 2.5, 9.0):
        bg.add(raster_polylines([pts], (w, h), 2), C_WAVE, 0.17 * g, 0.26 * g)

    cube_specs = []
    for _ in range(args.cubes):
        cxx = rng.uniform(0.02, 0.98) * w
        cyy = rng.uniform(0.40, 0.97) * h
        s = float(np.exp(rng.uniform(math.log(5 * SS), math.log(26 * SS))))
        yaw = rng.uniform(0.0, math.pi)
        depth = rng.uniform(0.0, 1.0)
        cube_specs.append((cxx, cyy, s, yaw, 0.34 + 0.66 * depth))
    face_polys = []
    for cxx, cyy, s, yaw, _g in cube_specs:
        _segs, faces = cube_segments(cxx, cyy, s, yaw)
        face_polys.extend(faces)
    bg.add(raster_polys(face_polys, (w, h)), C_CUBE, 0.035, 0.05)
    for cxx, cyy, s, yaw, g in cube_specs:
        segs, _faces = cube_segments(cxx, cyy, s, yaw)
        bg.add(raster_segments(segs, (w, h), 2), C_CUBE, 0.26 * g, 0.30 * g)

    n = args.dust
    dy = rng.uniform(0, h, n).astype(np.float32)
    dx = rng.uniform(0, w, n).astype(np.float32)
    dv = rng.uniform(0.02, 0.14, n).astype(np.float32)
    m = np.zeros((h, w), np.float32)
    stamp(m, dy, dx, dv, 0.9 * SS)
    bg.add(gaussian_filter(m, 0.5), C_MID, 1.0, 0.85)

    # 鯨のまわりのほのかな水の光（参考画像の「光る水」に寄せる）
    bg.add(gaussian_filter(body, 22.0 * SS) * (0.055 + 0.10 * np.clip(Ln * 1.15, 0.0, 1.0)), C_BODY_GLOW, 1.0, 0.22)

    # ------------------------------------------------------------------
    # 鯨レイヤー
    # ------------------------------------------------------------------
    whale = Layer(w, h)

    # 体の下地: 写真の輝度を「深い藍 → 青 → シアン」のランプに写す。
    # 明るい所（背の稜線・腹・胸びれ）はそのまま光の色になる。
    t = np.clip((Ln * 1.35) ** 0.85, 0.0, 1.0)
    fill = C_FILL_LO[None, None, :] + (C_FILL_HI - C_FILL_LO)[None, None, :] * t[..., None]
    hi = np.clip((t - 0.42) / 0.58, 0.0, 1.0)[..., None] ** 1.4
    fill = fill + (C_MID - fill) * hi * 0.45
    fill_a = body * (0.40 + 0.34 * t)
    whale.em += fill * fill_a[..., None] * 0.78
    whale.a += fill_a

    # 粒子（体の内側に一様にまき、明るさは輝度と局所コントラストで決める）
    gx, gy = jitter_grid(w, h, 4.6 * SS, rng)
    dens = sample(body, gx, gy) * (0.88 + 0.12 * sample(Ln, gx, gy))
    keep = rng.random(gx.size) < np.clip(dens * 1.05, 0.0, 0.99)
    gx, gy = gx[keep], gy[keep]
    lv = (0.32 * np.clip(sample(Ln, gx, gy) * 1.6, 0.0, 1.0) ** 0.8
          + 0.68 * sample(detail, gx, gy))
    lv = np.clip(lv + rng.normal(0, 0.06, gx.size).astype(np.float32), 0.0, 1.0)

    dim = lv < 0.34
    mid = (lv >= 0.34) & (lv < 0.66)
    hot = lv >= 0.66
    for sel, col, sig, base, span, ag in (
            (dim, C_DIM, 1.05, 0.40, 0.70, 0.62),
            (mid, C_MID, 1.35, 0.85, 0.80, 0.78),
            (hot, C_HOT, 2.00, 1.45, 0.70, 0.92)):
        if not sel.any():
            continue
        vals = (base + span * lv[sel]).astype(np.float32)
        m = np.zeros((h, w), np.float32)
        stamp(m, gy[sel], gx[sel], vals, sig * SS)
        whale.add(m, col, 1.0, ag)

    # 白い星（明るい面に散る小さな点）
    n = int(args.sparks)
    sy = rng.uniform(0, h, n * 4).astype(np.float32)
    sx = rng.uniform(0, w, n * 4).astype(np.float32)
    wgt = sample(body, sx, sy) * sample(Ln, sx, sy) ** 2.0
    pick = rng.random(sx.size) < np.clip(wgt * 1.9, 0, 1)
    sx, sy = sx[pick][:n], sy[pick][:n]
    if sx.size:
        m = np.zeros((h, w), np.float32)
        stamp(m, sy, sx, (0.40 + 0.60 * rng.random(sx.size)).astype(np.float32), 1.05 * SS)
        whale.add(m, C_HOT, 1.0, 0.85)

    # しわ・溝の点（写真のエッジ強度から拾う。明るいシアンの細かい点）
    ex, ey = jitter_grid(w, h, 3.6 * SS, rng)
    ev = sample(eg, ex, ey) * sample(body, ex, ey)
    esel = ev > 0.10
    esel &= rng.random(ex.size) < np.clip(ev * 2.4, 0.0, 0.95)
    if esel.any():
        vals = (0.12 + 0.55 * np.clip(sample(blur_L, ex[esel], ey[esel]), 0.0, 1.0)).astype(np.float32)
        m = np.zeros((h, w), np.float32)
        stamp(m, ey[esel], ex[esel], vals, 0.95 * SS)
        whale.add(m, C_MID, 1.0, 0.75)

    # 輪郭（等輝度リング）とその上の粒
    ring = np.exp(-((gaussian_filter(body, 1.6 * SS) - 0.5) / 0.085) ** 2)
    rim_pow = 0.48 + 0.52 * np.clip(blur_L, 0.0, 1.0) ** 1.1
    whale.add(ring * rim_pow, C_RIM, 1.45, 0.92)
    whale.add(gaussian_filter(ring, 3.2 * SS) * rim_pow, C_RIM, 0.75, 0.58)

    rgx, rgy = jitter_grid(w, h, 4.6 * SS, rng)
    rsel = sample(ring, rgx, rgy) > 0.22
    rsel &= rng.random(rgx.size) < 0.8
    rr = (0.30 + 0.70 * sample(blur_L, rgx, rgy)).astype(np.float32)
    m = np.zeros((h, w), np.float32)
    stamp(m, rgy[rsel], rgx[rsel], rr[rsel], 1.3 * SS)
    whale.add(m, C_RIM, 1.0, 0.9)

    # 網（ドロネー）
    nx, ny = jitter_grid(w, h, 27.0 * SS, rng)
    nsel = (sample(body, nx, ny) > 0.8)
    nsel &= rng.random(nx.size) < 0.42
    nx, ny = nx[nsel], ny[nsel]
    if nx.size >= 4:
        pts = np.stack([nx, ny], axis=1)
        tri = Delaunay(pts)
        max_len = 70.0 * SS
        segs_by_gain = ([], [], [])
        gains = (0.14, 0.22, 0.32)
        for a, b, c in tri.simplices:
            for i, j in ((a, b), (b, c), (c, a)):
                x0, y0 = pts[i]
                x1, y1 = pts[j]
                mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
                if math.hypot(x1 - x0, y1 - y0) > max_len:
                    continue
                if sample(body, np.array([mx]), np.array([my]))[0] < 0.5:
                    continue
                if sample(Ln, np.array([mx]), np.array([my]))[0] < 0.16 and rng.random() < 0.55:
                    continue
                g = 0.45 + 0.55 * float(sample(blur_L, np.array([mx]), np.array([my]))[0])
                k = 0 if g < 0.68 else (1 if g < 0.84 else 2)
                segs_by_gain[k].append((x0, y0, x1, y1))
        for segs, g in zip(segs_by_gain, gains):
            if segs:
                whale.add(raster_segments(segs, (w, h), 2), C_NET, g, g * 0.95)
        m = np.zeros((h, w), np.float32)
        nv = (0.40 + 0.60 * sample(blur_L, nx, ny)).astype(np.float32)
        stamp(m, ny, nx, nv, 1.6 * SS)
        whale.add(m, C_HOT, 0.9, 0.85)

    # ------------------------------------------------------------------
    # 手前レイヤー（画面下の光の筋と手前の箱）
    # ------------------------------------------------------------------
    fg = Layer(w, h)
    for pts, g in wave_polylines(rng, w, h, args.front_waves, 0.74, 0.99, 1.5, 6.0):
        fg.add(raster_polylines([pts], (w, h), 2), C_WAVE, 0.30 * g, 0.40 * g)

    for _ in range(args.front_cubes):
        cxx = rng.uniform(0.02, 0.98) * w
        cyy = rng.uniform(0.80, 0.96) * h
        s = float(np.exp(rng.uniform(math.log(7 * SS), math.log(20 * SS))))
        yaw = rng.uniform(0.0, math.pi)
        segs, _faces = cube_segments(cxx, cyy, s, yaw)
        fg.add(raster_segments(segs, (w, h), 2), C_CUBE, 0.34 * (0.6 + 0.4 * rng.random()), 0.40)

    comp = over(whale, over(fg, bg))
    comp.a = np.clip(comp.a, 0.0, 1.0)

    # 最終解像度にしてから多段ブルーム
    em = resize_rgb(comp.em, (CANVAS_W, CANVAS_H))
    al = np.clip(resize_f(comp.a, (CANVAS_W, CANVAS_H)), 0.0, 1.0)

    glow = (gaussian_filter(em, (1.2, 1.2, 0.0)) * 0.80
            + gaussian_filter(em, (3.5, 3.5, 0.0)) * 0.38
            + gaussian_filter(em, (9.0, 9.0, 0.0)) * 0.14
            + gaussian_filter(em, (30.0, 30.0, 0.0)) * 0.06)
    glow *= args.glow

    em = em + glow * 0.55
    al = np.clip(al + luminance(glow) * 0.34, 0.0, 1.0)
    al = np.clip((al - 0.012) / 0.988, 0.0, 1.0)

    rgb_out = np.clip(em / np.maximum(al, 1e-4)[..., None], 0.0, 1.0)
    rgb_out = 1.0 - np.exp(-rgb_out * args.exposure * 2.6)          # やわらかいトーン圧縮
    rgb_out = rgb_out ** 1.05
    rgb_out[al < 0.02] = 0.0

    out = Image.fromarray((np.clip(np.dstack([rgb_out, al]), 0, 1) * 255).astype(np.uint8), 'RGBA')
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(f'png {out_path} {out_path.stat().st_size} B')

    if args.webp:
        wp = Path(args.webp)
        out.save(wp, 'WEBP', quality=args.webp_quality, method=6, alpha_quality=100, exact=True)
        print(f'webp {wp} {wp.stat().st_size} B  base64 {math.ceil(wp.stat().st_size / 3) * 4} B')


if __name__ == '__main__':
    main()
