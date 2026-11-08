#!/usr/bin/env python3
"""写真版の大鯨を「電子の線」で描き直す。

build.py が作った光の鯨（黒い水に浮かぶ写真のレンダー）を入力にして、
形はそのまま、描き方だけを線に置き換える。

1. 外形線: 体のマスクをぼかして 0.5 の等高線をなぞり、一定の太さで光らせる
2. 線画: ぼかした輝度の勾配に非最大抑制（NMS）とヒステリシスをかけて主線を抜く
3. 等高線（アイソフォート）: 体の丸みを線で示す
4. 破線＋パッド点: 線を短い区間に切る（線画らしさ）
5. 回路トレース: 水平に走って 45 度で折れる配線を数本（電子っぽさ）
6. 粒子化（--stipple）: できた線を一定間隔で間引き、ガウスの点で描き直す
   （--stipple-shade で体の内側にも明るさに応じた粒をまける）
7. 発光（ブルーム）と任意の走査線・色ずれを足し、1200x560 viewBox に配置

等高線は入力の粒状ノイズを拾いやすいので、ぼかし＋勾配のゲートで
「面の移り変わり」だけを残す。線はどの処理も --ss 倍に拡大した場で
描いてから縮小するので、1px でも階段状にならず滑らかになる。

使い方（現行の既定の絵。全オプションは --help）:
  ~/AI/ComfyUI/.venv/bin/python lineart.py \
      --src out/whale4_s411.png --bands 7 --blur 16 --sketch 0.55 \
      --rim-mode iso --rim-weight 2.4 --mask-blur 8 \
      --dash-line 0.7 --dash-pitch 60 --dash-dot 1.0 --circuit 6 \
      --ss 4 --scale 1.2 --bottom 496 --length 1060 \
      --out art/lines2.png --webp art/lines2-q85.webp --webp-quality 85
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (binary_erosion, binary_fill_holes, gaussian_filter,
                           label, maximum_filter)

HERE = Path(__file__).resolve().parent
VIEW_W, VIEW_H = 1200, 560


def smoothstep(x: np.ndarray, a: float, b: float) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def key_alpha(rgb: np.ndarray, lo: float = 0.010, hi: float = 0.052) -> np.ndarray:
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return smoothstep(luma, lo, hi)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def disk(radius: int) -> np.ndarray:
    r = max(1, radius)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def resize_f(arr: np.ndarray, size: tuple[int, int], resample) -> np.ndarray:
    """float32 の 2D 配列を PIL でリサイズする（'F' モードは LANCZOS 可）。"""
    img = Image.fromarray(arr.astype(np.float32), mode='F')
    return np.asarray(img.resize(size, resample)).astype(np.float32)


def place_on_canvas(lum: np.ndarray, alpha: np.ndarray, rgb: np.ndarray, args):
    """build.py と同じ手順で 1200x560（x scale）のキャンバスへ置く。"""
    ys, xs = np.where(alpha > 0.02)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    lum = lum[y0:y1 + 1, x0:x1 + 1]
    alpha = alpha[y0:y1 + 1, x0:x1 + 1]
    rgb = rgb[y0:y1 + 1, x0:x1 + 1]

    pad = 12
    lum = np.pad(lum, pad)
    alpha = np.pad(alpha, pad)
    rgb = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)))

    h, w = lum.shape
    target = int(round(args.length * args.scale))
    new_w = target
    new_h = int(round(h * (target / float(w))))

    lum_img = Image.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8), 'L')
    alpha_img = Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8), 'L')
    rgb_img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), 'RGB')
    lum_img = lum_img.resize((new_w, new_h), Image.LANCZOS)
    alpha_img = alpha_img.resize((new_w, new_h), Image.LANCZOS)
    rgb_img = rgb_img.resize((new_w, new_h), Image.LANCZOS)

    cw, ch = int(round(VIEW_W * args.scale)), int(round(VIEW_H * args.scale))
    canvas_l = np.zeros((ch, cw), dtype=np.float32)
    canvas_a = np.zeros((ch, cw), dtype=np.float32)
    canvas_c = np.zeros((ch, cw, 3), dtype=np.float32)
    ox = int(round(args.center * args.scale - new_w / 2.0))
    oy = int(round(args.bottom * args.scale - new_h))

    Lc = np.asarray(lum_img).astype(np.float32) / 255.0
    Ac = np.asarray(alpha_img).astype(np.float32) / 255.0
    Cc = np.asarray(rgb_img).astype(np.float32) / 255.0

    # キャンバスからはみ出す分は切り落として合成する（build.py と同じ見え方）
    y0 = max(oy, 0)
    y1 = min(oy + new_h, ch)
    x0 = max(ox, 0)
    x1 = min(ox + new_w, cw)
    canvas_l[y0:y1, x0:x1] = Lc[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    canvas_a[y0:y1, x0:x1] = Ac[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    canvas_c[y0:y1, x0:x1] = Cc[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return canvas_l, canvas_a, canvas_c


def contour_light(Ls: np.ndarray, body: np.ndarray, args) -> np.ndarray:
    """拡大した場で等高線を引き、縮小して滑らかな線にする。"""
    ss = max(1, args.ss)
    ch, cw = Ls.shape
    hw, hh = cw * ss, ch * ss

    Ls_h = resize_f(Ls, (hw, hh), Image.BICUBIC)
    body_h = resize_f(body, (hw, hh), Image.BILINEAR)
    Ln_h = np.clip((Ls_h - 0.012) / 0.38, 0.0, 1.0)

    q = np.floor(Ln_h * args.bands)
    edge = np.zeros_like(Ln_h, dtype=bool)
    edge[:, 1:] |= q[:, 1:] != q[:, :-1]
    edge[1:, :] |= q[1:, :] != q[:-1, :]

    gy, gx = np.gradient(Ls_h)
    grad = np.hypot(gx, gy)
    edge &= (body_h > 0.15) & (grad > args.gate)
    strength = np.clip(grad / (args.gate * 6.0), 0.0, 1.0)

    e = edge.astype(np.float32) * (0.45 + 0.55 * strength)
    e = maximum_filter(e, footprint=disk((ss + 1) // 2))
    e = gaussian_filter(e, max(ss * args.core_soft, 0.01)) * args.core_gain
    core = resize_f(e, (cw, ch), Image.LANCZOS)
    return np.clip(core, 0.0, 1.0)


def sketch_light(L: np.ndarray, body: np.ndarray, args) -> np.ndarray:
    """輪郭線を「線画」として抜き出す（ペン画の主線の担当）。

    ぼかした輝度の勾配から、勾配方向の非最大抑制（NMS）で 1px まで細める。
    強エッジから弱エッジをたどるヒステリシスで線をつなぎ、拡大した場で
    作業してから縮小するので、線が階段状にならない。
    """
    ss = max(1, args.sketch_ss)
    ch, cw = L.shape
    hw, hh = cw * ss, ch * ss

    base = gaussian_filter(L, args.sketch_blur)
    Lh = resize_f(base, (hw, hh), Image.BICUBIC)
    body_h = resize_f(body, (hw, hh), Image.BILINEAR)

    gy, gx = np.gradient(Lh)
    mag = np.hypot(gx, gy)
    mag = mag / max(float(mag.max()), 1e-6)

    # 勾配方向 8 方向で非最大抑制
    idx = np.round(np.arctan2(gy, gx) / (np.pi / 4.0)).astype(np.int32) % 8
    off = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    keep = np.zeros_like(mag, dtype=bool)
    for k, (dx, dy) in enumerate(off):
        m = idx == k
        if not m.any():
            continue
        n1 = np.roll(mag, (dy, dx), axis=(0, 1))
        n2 = np.roll(mag, (-dy, -dx), axis=(0, 1))
        keep |= m & (mag >= n1) & (mag >= n2)

    keep &= body_h > 0.30
    weak = keep & (mag > args.sketch_low)
    strong = keep & (mag > args.sketch_high)

    # ヒステリシス: 強エッジにつながる弱エッジだけ残す
    lab, n = label(weak)
    if n > 0:
        keep_ids = np.unique(lab[strong])
        keep_ids = keep_ids[keep_ids > 0]
        weak = np.isin(lab, keep_ids) if keep_ids.size else np.zeros_like(weak)

    e = weak.astype(np.float32)
    e = np.clip(gaussian_filter(e, 0.6), 0.0, 1.0)
    core = resize_f(e, (cw, ch), Image.LANCZOS)
    return np.clip(core * args.sketch_gain, 0.0, 1.0)


def circuit_traces(body: np.ndarray, args) -> tuple[np.ndarray, np.ndarray]:
    """体の上に基板の配線のような直線トレースを引く（電子っぽさの担当）。

    水平に長く走り、ときどき 45 度で上下に折れる経路を数本引く。等高線
    （有機的な曲線）と方向が違うので、見た目に「回路」として読める。
    経路の両端には丸いパッド（端子）を打つ。乱数（--circuit-seed）で決まる。
    """
    hh, ww = body.shape
    tr = np.zeros((hh, ww), dtype=np.float32)
    dot = np.zeros((hh, ww), dtype=np.float32)
    if args.circuit <= 0:
        return tr, dot

    rng = np.random.default_rng(args.circuit_seed)
    solid = body > 0.55
    ys, xs = np.nonzero(solid)
    if ys.size == 0:
        return tr, dot

    pads: list[tuple[int, int]] = []
    rows_used: list[int] = []
    made = 0
    attempts = 0
    while made < args.circuit and attempts < args.circuit * 200:
        attempts += 1
        i = int(rng.integers(ys.size))
        y, x = int(ys[i]), int(xs[i])
        if tr[max(0, y - 2):y + 3, max(0, x - 2):x + 3].any():
            continue
        # 同じ高さに何本も並ばないようにする
        if any(abs(y - ry) < 9 for ry in rows_used):
            continue
        sign = 1 if rng.random() < 0.5 else -1
        path = [(y, x)]
        budget = int(rng.integers(110, 300))
        moved = 0
        while budget > 0:
            if rng.random() < 0.28:
                dy, dx = int(rng.choice((-1, 1))), sign      # 45 度のジョグ
                ln = int(rng.integers(8, 20))
            else:
                dy, dx = 0, sign                              # 水平の区間
                ln = int(rng.integers(14, 40))
            ln = min(ln, budget)
            ok = True
            for _step in range(ln):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < hh and 0 <= nx < ww):
                    ok = False
                    break
                if not solid[ny, nx] or tr[ny, nx] > 0:
                    ok = False
                    break
                y, x = ny, nx
                path.append((y, x))
                moved += 1
                budget -= 1
            if not ok or budget <= 0:
                break
        if moved < 24:
            continue
        for (py, px) in path:
            tr[py, px] = 1.0
        pads.extend((path[0], path[-1]))
        rows_used.append(path[0][0])
        made += 1

    # 1px の線は縮小で消えるので、ごく薄く太らせる
    tr = gaussian_filter(tr, 0.8)
    tr = np.clip(tr * 1.6, 0.0, 1.0)

    for (py, px) in pads:
        yy, xx = np.mgrid[-2:3, -2:3]
        r = np.sqrt(yy * yy + xx * xx)
        stamp = np.exp(-(r / 1.15) ** 2)
        y0, y1 = max(py - 2, 0), min(py + 3, hh)
        x0, x1 = max(px - 2, 0), min(px + 3, ww)
        dot[y0:y1, x0:x1] += stamp[(y0 - py + 2):(y1 - py + 2), (x0 - px + 2):(x1 - px + 2)]
    dot = np.clip(dot, 0.0, 1.0)

    return tr * args.circuit_alpha, dot * args.circuit_dot


def stipple_field(light: np.ndarray, Ln: np.ndarray, body: np.ndarray, args) -> np.ndarray:
    """線を光の粒（点描）に置き換える。

    線の画素を間隔 pitch の格子で間引いて 1 粒ずつ拾い、ガウスで描く。
    粒の大きさ・明るさは線の強さに従い、--stipple-scatter で位置を散らすと
    線がほどけて粒子の帯になる。
    """
    hh, ww = light.shape
    field = np.clip(light, 0.0, 1.6)
    ys, xs = np.nonzero(field > args.stipple_thr)
    if ys.size == 0:
        return np.zeros((hh, ww), dtype=np.float32)

    rng = np.random.default_rng(args.stipple_seed)
    order = rng.permutation(ys.size)
    ys, xs = ys[order], xs[order]

    # pitch の格子ごとに 1 粒だけ残す（格子内でランダムに選ぶ）
    pitch = max(0.7, args.stipple_pitch)
    cells = (ys // pitch).astype(np.int64) * 100003 + (xs // pitch).astype(np.int64)
    _, first = np.unique(cells, return_index=True)
    ys, xs = ys[first], xs[first]
    strength = np.clip(field[ys, xs], 0.0, 1.0)
    n = ys.size

    # サブピクセルでゆらす
    jit = args.stipple_jitter
    yf = ys.astype(np.float32) + rng.uniform(-0.5, 0.5, n).astype(np.float32) * jit
    xf = xs.astype(np.float32) + rng.uniform(-0.5, 0.5, n).astype(np.float32) * jit

    # 散らす: 一部の粒を線から離す（ほどけた粒子）
    amp = np.zeros(n, dtype=np.float32)
    if args.stipple_scatter > 0:
        loose = rng.random(n) < args.stipple_loose
        amp = np.where(loose,
                       rng.uniform(0.35, 1.0, n) * args.stipple_scatter * 2.2,
                       rng.uniform(0.0, 1.0, n) * args.stipple_scatter).astype(np.float32)
        strength = np.where(loose, strength * 0.45, strength).astype(np.float32)
        ang = rng.uniform(0.0, 2.0 * np.pi, n).astype(np.float32)
        yf = yf + amp * np.sin(ang)
        xf = xf + amp * np.cos(ang)

    # 粒をガウスの点で描く（サブピクセルは 3x3 の重みに配る）
    acc = np.zeros((hh, ww), dtype=np.float32)
    sigma = max(0.5, args.stipple_size)
    rad = int(max(1, np.ceil(sigma * 2.2)))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    kern = np.exp(-((yy * yy + xx * xx) / (2.0 * sigma * sigma))).astype(np.float32)
    kern = kern / max(float(kern.sum()), 1e-6)
    y0 = yf - 0.5
    x0 = xf - 0.5
    iy = np.floor(y0).astype(np.int64)
    ix = np.floor(x0).astype(np.int64)
    fy = (y0 - iy).astype(np.float32)
    fx = (x0 - ix).astype(np.float32)
    wts = ((1.0 - fy) * fx, (1.0 - fy) * (1.0 - fx), fy * fx, fy * (1.0 - fx))
    corners = ((0, 1), (0, 0), (1, 1), (1, 0))
    amp_val = (strength * args.stipple_gain).astype(np.float32)

    for wgt, (dy, dx) in zip(wts, corners):
        py = iy + dy
        px = ix + dx
        ok = (py >= 0) & (py < hh) & (px >= 0) & (px < ww)
        wy = py[ok]
        wx = px[ok]
        wv = (amp_val[ok] * wgt[ok])
        for ky in range(-rad, rad + 1):
            ty = wy + ky
            ky_ok = (ty >= 0) & (ty < hh)
            if not ky_ok.any():
                continue
            for kx in range(-rad, rad + 1):
                tx = wx + kx
                k_ok = ky_ok & (tx >= 0) & (tx < ww)
                if not k_ok.any():
                    continue
                np.add.at(acc, (ty[k_ok], tx[k_ok]), wv[k_ok] * kern[ky + rad, kx + rad])

    # 陰影の粒: 線ではなく体の内側に、明るい所ほど多くの粒をまく（点描の陰影）
    if args.stipple_shade > 0:
        shade_n = int(args.stipple_shade * body.sum() / max(args.stipple_shade_density, 1.0))
        wgt = (body * (0.18 + 0.82 * np.clip(Ln, 0.0, 1.0) ** 1.6)).ravel()
        tot = float(wgt.sum())
        if tot > 0 and shade_n > 0:
            cdf = np.cumsum(wgt) / tot
            pick = np.searchsorted(cdf, rng.random(shade_n))
            pick = np.clip(pick, 0, wgt.size - 1)
            sy = (pick // ww).astype(np.float32)
            sx = (pick % ww).astype(np.float32)
            sy = sy + rng.uniform(-0.5, 0.5, shade_n).astype(np.float32)
            sx = sx + rng.uniform(-0.5, 0.5, shade_n).astype(np.float32)
            s_val = (args.stipple_shade_gain
                     * (0.35 + 0.65 * np.clip(Ln, 0.0, 1.0).ravel()[pick])).astype(np.float32)
            iy2 = np.floor(sy).astype(np.int64)
            ix2 = np.floor(sx).astype(np.int64)
            fy2 = sy - iy2
            fx2 = sx - ix2
            for wgt2, (dy, dx) in zip(((1.0 - fy2) * fx2, (1.0 - fy2) * (1.0 - fx2),
                                      fy2 * fx2, fy2 * (1.0 - fx2)),
                                     ((0, 1), (0, 0), (1, 1), (1, 0))):
                py2 = iy2 + dy
                px2 = ix2 + dx
                ok2 = (py2 >= 0) & (py2 < hh) & (px2 >= 0) & (px2 < ww)
                wy2 = py2[ok2]
                wx2 = px2[ok2]
                wv2 = s_val[ok2] * wgt2[ok2]
                for ky in range(-rad, rad + 1):
                    ty2 = wy2 + ky
                    ky_ok2 = (ty2 >= 0) & (ty2 < hh)
                    if not ky_ok2.any():
                        continue
                    for kx in range(-rad, rad + 1):
                        tx2 = wx2 + kx
                        k_ok2 = ky_ok2 & (tx2 >= 0) & (tx2 < ww)
                        if not k_ok2.any():
                            continue
                        np.add.at(acc, (ty2[k_ok2], tx2[k_ok2]),
                                  wv2[k_ok2] * kern[ky + rad, kx + rad])

    acc = gaussian_filter(acc, 0.5)
    return np.clip(acc, 0.0, 1.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(HERE / 'out/whale4_s411.png'))
    ap.add_argument('--scale', type=float, default=1.2)
    ap.add_argument('--bottom', type=float, default=496.0)
    ap.add_argument('--length', type=float, default=1060.0)
    ap.add_argument('--center', type=float, default=616.0)
    ap.add_argument('--bands', type=int, default=9, help='等高線の本数（0 で等高線なし）')
    ap.add_argument('--blur', type=float, default=9.0, help='等高線を引く前のぼかし（粒ノイズ対策）')
    ap.add_argument('--gate', type=float, default=0.0006, help='この勾配未満は線を引かない')
    ap.add_argument('--ss', type=int, default=3, help='等高線を引くときの拡大率（2〜4）')
    ap.add_argument('--core-gain', type=float, default=4.0, help='等高線の明るさ補正')
    ap.add_argument('--core-soft', type=float, default=0.5, help='等高線の滲み（ss に対する倍率）')
    ap.add_argument('--scan', type=int, default=6, help='走査線の間隔 px（0 で無効）')
    ap.add_argument('--scan-alpha', type=float, default=0.07, help='走査線の強さ')
    ap.add_argument('--dash', type=float, default=0.0, help='走査線を破線にする強さ 0..1')
    ap.add_argument('--dash-pitch', type=float, default=26.0, help='破線の周期 px')
    ap.add_argument('--slant', type=float, default=0.0, help='走査線の傾き（px/px）')
    ap.add_argument('--vgrid', type=int, default=0, help='縦のグリッド間隔 px（0 で無効）')
    ap.add_argument('--vgrid-alpha', type=float, default=0.10, help='縦グリッドの強さ')
    ap.add_argument('--ribs', type=int, default=0, help='体を横切る細い縦線の間隔 px（0 で無効）')
    ap.add_argument('--rib-alpha', type=float, default=0.16, help='リブの明るさ')
    ap.add_argument('--fill', type=float, default=0.08, help='体の下地の不透明度（0で完全に線だけ）')
    ap.add_argument('--ghost', type=float, default=0.0, help='写真の残像の不透明度（0で線だけ）')
    ap.add_argument('--rim-mode', choices=('iso', 'mask', 'alpha', 'edge', 'grad'), default='iso',
                    help='輪郭の拾い方（iso=拡大してなぞる外形 / mask=体の外形 / alpha=アルファの等高線 / edge=輝度の勾配 / grad=マスクの勾配）')
    ap.add_argument('--mask-blur', type=float, default=2.5, help='マスクのぼかし（浮遊物の粒を消す）')
    ap.add_argument('--rim-level', type=float, default=0.50, help='alpha モードで拾うアルファ値')
    ap.add_argument('--rim-width', type=float, default=0.05, help='輪郭線の太さ（アルファ幅）')
    ap.add_argument('--rim-gain', type=float, default=0.030, help='edge モードで輪郭を拾う勾配の基準')
    ap.add_argument('--rim-blur', type=float, default=1.2, help='mask モードの輪郭の滲み')
    ap.add_argument('--rim-bright', type=float, default=4.2, help='輪郭の明るさ補正')
    ap.add_argument('--rim-weight', type=float, default=1.0, help='輪郭線の太さ px（1 で細い線）')
    ap.add_argument('--rim', type=float, default=0.90, help='輪郭の光の強さ')
    ap.add_argument('--rim-min', type=float, default=0.25, help='暗部での輪郭の残し方')
    ap.add_argument('--glow', type=float, default=0.75, help='発光の強さ')
    ap.add_argument('--chroma', type=float, default=0.0, help='色ずれ（RGB 分離）の強さ 0..1')
    ap.add_argument('--light', type=float, default=1.25, help='線全体の明るさ')
    ap.add_argument('--tone', type=float, default=1.0, help='明るい所ほど線を強くする度合い（0で一様）')
    ap.add_argument('--nodes', type=int, default=0, help='輪郭に打つ節点の間隔 px（0 で無効）')
    ap.add_argument('--node-bright', type=float, default=0.9, help='節点の明るさ')
    ap.add_argument('--sketch', type=float, default=0.0, help='輪郭線画の重ね方 0..1（0 で無効）')
    ap.add_argument('--sketch-blur', type=float, default=5.0, help='線画の前にぼかす強さ')
    ap.add_argument('--sketch-ss', type=int, default=2, help='線画の作業倍率')
    ap.add_argument('--sketch-low', type=float, default=0.030, help='弱いエッジのしきい値')
    ap.add_argument('--sketch-high', type=float, default=0.085, help='強いエッジのしきい値')
    ap.add_argument('--sketch-gain', type=float, default=1.8, help='線画の明るさ補正')
    ap.add_argument('--dash-line', type=float, default=0.0, help='等高線を破線にする強さ 0..1')
    ap.add_argument('--dash-dot', type=float, default=0.0, help='破線の始点に打つ光点の強さ 0..1')
    ap.add_argument('--circuit', type=int, default=0, help='基板のようなトレースの本数（0 で無効）')
    ap.add_argument('--circuit-alpha', type=float, default=0.35, help='トレースの明るさ')
    ap.add_argument('--circuit-dot', type=float, default=1.0, help='端子・接点の点の明るさ')
    ap.add_argument('--circuit-seed', type=int, default=7, help='トレースの乱数シード')
    ap.add_argument('--stipple', type=float, default=0.0, help='線を粒子に置き換える強さ 0..1（0 で線のまま）')
    ap.add_argument('--stipple-pitch', type=float, default=3.0, help='粒の間隔 px')
    ap.add_argument('--stipple-size', type=float, default=1.0, help='粒の大きさ（ガウスの σ）')
    ap.add_argument('--stipple-gain', type=float, default=1.2, help='粒の明るさ')
    ap.add_argument('--stipple-thr', type=float, default=0.10, help='この線の強さ未満は粒にしない')
    ap.add_argument('--stipple-jitter', type=float, default=0.9, help='粒の位置のゆらぎ px')
    ap.add_argument('--stipple-scatter', type=float, default=0.0, help='粒を線から散らす距離 px')
    ap.add_argument('--stipple-loose', type=float, default=0.0, help='散らして漂わせる粒の割合 0..1')
    ap.add_argument('--stipple-seed', type=int, default=13, help='粒の乱数シード')
    ap.add_argument('--stipple-shade', type=float, default=0.0, help='体の内側にまく陰影の粒の量 0..1')
    ap.add_argument('--stipple-shade-density', type=float, default=30.0,
                    help='陰影の粒の粗さ（1粒あたりの面積 px²。大きいほど少ない）')
    ap.add_argument('--stipple-shade-gain', type=float, default=0.55, help='陰影の粒の明るさ')
    ap.add_argument('--tail-fade', type=float, default=0.12, help='尾側を闇へ落とす幅（割合）')
    ap.add_argument('--out', default=str(HERE / 'art/lines.png'))
    ap.add_argument('--webp', default='')
    ap.add_argument('--webp-quality', type=int, default=66)
    args = ap.parse_args()

    im = Image.open(args.src).convert('RGB')
    rgb = np.asarray(im).astype(np.float32) / 255.0
    alpha = key_alpha(rgb)
    lum = luminance(rgb)

    L, A, C = place_on_canvas(lum, alpha, rgb, args)

    # 面の明るさを正規化（鯨の明るい側を 1、水の黒を 0）
    Ls = gaussian_filter(L, args.blur)
    Ln = np.clip((Ls - 0.012) / 0.38, 0.0, 1.0)
    As = gaussian_filter(A, args.mask_blur)
    body = smoothstep(As, 0.18, 0.55)

    # 1) 等高線
    core = contour_light(Ls, body, args) if args.bands > 0 else np.zeros_like(Ls)

    # 1a) 線画: 輪郭の勾配を細らせた主線
    if args.sketch > 0 and args.bands > 0:
        sk = sketch_light(L, body, args)
        core = np.clip(core + sk * args.sketch, 0.0, 1.0)
    elif args.sketch > 0:
        core = sketch_light(L, body, args)
    else:
        sk = None

    # 1b) 破線: 等高線を短い区間に切ると「描いた線」に見える（線画らしさ）
    if args.dash_line > 0 and args.bands > 0:
        hh, ww = Ls.shape
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        ph = np.mod((xx * 0.82 + yy * 0.57) / max(args.dash_pitch, 2.0), 1.0)
        on = smoothstep(ph, 0.0, 0.04) * (1.0 - smoothstep(ph, 0.50, 0.58))
        base_core = core
        core = core * ((1.0 - args.dash_line) + args.dash_line * on)
        if args.dash_dot > 0:
            # 破線の始点に小さな光点を置く（基板のパッドのような電子っぽさ）
            dm = smoothstep(ph, 0.0, 0.02) * (1.0 - smoothstep(ph, 0.04, 0.07))
            dm = gaussian_filter(dm, 0.9)
            core = np.clip(core + base_core * dm * args.dash_dot, 0.0, 1.0)

    # 2) 走査線: 計測器/ホログラムのような横の線
    if args.scan > 0:
        hh, ww = Ls.shape
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        yoff = yy - xx * args.slant
        dist = np.abs(yoff - np.round(yoff / args.scan) * args.scan)
        rows = np.clip(1.0 - dist / 1.1, 0.0, 1.0)
        if args.dash > 0:
            ph = np.mod(xx / args.dash_pitch, 1.0)
            rows = rows * ((1.0 - args.dash) + args.dash * (ph < 0.55))
        scan = rows * body * (0.25 + 0.75 * Ln) * args.scan_alpha
    else:
        scan = np.zeros_like(Ls)

    # 3) 縦のグリッド（任意）: メッシュ感を足す
    if args.vgrid > 0:
        hh, ww = Ls.shape
        xx = np.arange(ww, dtype=np.float32)[None, :]
        distx = np.abs(xx - np.round(xx / args.vgrid) * args.vgrid)
        vcols = np.clip(1.0 - distx / 1.1, 0.0, 1.0)
        vgrid = vcols * body * (0.20 + 0.80 * Ln) * args.vgrid_alpha
    else:
        vgrid = np.zeros_like(Ls)

    # 3b) リブ: 体を横切る細い縦線。ワイヤーフレームらしい構造を足す
    if args.ribs > 0:
        hh, ww = Ls.shape
        xx = np.arange(ww, dtype=np.float32)[None, :]
        distx = np.abs(xx - np.round(xx / args.ribs) * args.ribs)
        cols = np.clip(1.0 - distx / 1.1, 0.0, 1.0)
        # 明るいほど薄くして、背中の等高線とぶつからないようにする
        ribs = cols * body * (1.0 - 0.65 * np.clip(Ln, 0.0, 1.0)) * args.rib_alpha
    else:
        ribs = np.zeros_like(Ls)

    # 4) 輪郭: 体の外縁（アルファの等高線）か、輝度の勾配の強い所
    if args.rim_mode == 'iso':
        # 拡大したマスクで外形をなぞり、縮小して滑らかな1本の線にする
        ss = max(1, args.ss)
        hh, ww = Ls.shape
        As_h = resize_f(As, (ww * ss, hh * ss), Image.BICUBIC)
        m = As_h > args.rim_level
        ring = m & ~binary_erosion(m, iterations=1)
        # 線の太さは「仕上がり px」で指定し、拡大率に合わせて太らせる
        k = max(1, int(round(args.rim_weight * ss / 2.0)))
        e = maximum_filter(ring.astype(np.float32), footprint=disk(k))
        e = gaussian_filter(e, max(ss * 0.30, 0.01)) * args.rim_bright
        rim = resize_f(e, (ww, hh), Image.LANCZOS)
        rim_b = rim > 0.15
        lab, n = label(rim_b)
        if n > 0:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            keep = sizes >= 200
            keep[0] = False
            rim = rim * keep[lab]
        rim = np.clip(rim, 0.0, 1.0)
        rim_l = rim * args.rim * (args.rim_min + (1.0 - args.rim_min) * Ln ** args.tone)
    elif args.rim_mode == 'mask':
        # 体の外形（最大の連結成分）をなぞる → 浮遊物の粒を拾わない
        sil = As > 0.5
        lab, n = label(sil)
        if n > 0:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            sil = lab == sizes.argmax()
        sil = binary_fill_holes(sil)
        rim = sil & ~binary_erosion(sil, iterations=1)
        rim = gaussian_filter(rim.astype(np.float32), args.rim_blur)
        rim = np.clip(rim, 0.0, 1.0)
    elif args.rim_mode == 'grad':
        gA = np.hypot(*np.gradient(As))
        rim = np.clip(gA / args.rim_gain, 0.0, 1.0) ** 1.3
    elif args.rim_mode == 'edge':
        gE = np.hypot(*np.gradient(gaussian_filter(L, 0.9)))
        rim = np.clip(gE / args.rim_gain, 0.0, 1.0) * np.clip(body * 1.4, 0.0, 1.0)
    else:
        rim = np.exp(-((As - args.rim_level) / args.rim_width) ** 2)

    rim_l = rim * args.rim

    # 5) 節点: 輪郭に小さな光点を打つと「なぞった線」に見える
    if args.nodes > 0:
        ring_b = rim > 0.18
        hh, ww = Ls.shape
        dots = np.zeros_like(Ls)
        for yy0 in range(6, hh - 6, args.nodes):
            xs = np.nonzero(ring_b[yy0])[0]
            if xs.size == 0:
                continue
            for xx0 in (int(xs.min()), int(xs.max())):
                dots[yy0 - 1:yy0 + 2, xx0 - 1:xx0 + 2] += 1.0
        dots = gaussian_filter(dots, 1.0) * args.node_bright
        node_light = dots * (0.30 + 0.70 * Ln) * args.rim
    else:
        node_light = np.zeros_like(Ls)

    # 5b) 配線: 基板のような直線トレース（電子っぽさの担当）
    if args.circuit > 0:
        circ, circ_dot = circuit_traces(body, args)
    else:
        circ = np.zeros_like(Ls)
        circ_dot = np.zeros_like(Ls)

    # 6) 線の明るさ: 明るい所ほど強く、暗い所では闇に溶かす
    tone_mod = (1.0 - args.tone * 0.55) + args.tone * 0.55 * Ln ** 1.4
    lines = core * tone_mod + scan + vgrid + ribs
    light = np.clip(lines + rim_l + node_light + circ + circ_dot, 0.0, 1.4)

    # 6b) 粒: 線を点描に置き換える（等高線を消して粒子だけにもできる）
    if args.stipple > 0:
        dots = stipple_field(light, Ln, body, args)
        light = np.clip(light * (1.0 - args.stipple) + dots * args.stipple, 0.0, 1.4)

    # 6) 発光
    glow_src = light if args.stipple > 0 else lines + circ + circ_dot
    g1 = gaussian_filter(glow_src, 2.0)
    g2 = gaussian_filter(glow_src, 10.0)
    glow = np.clip(g1 * 0.50 + g2 * 0.35, 0.0, 1.2) * args.glow
    glow = glow + gaussian_filter(rim_l, 2.0) * 0.30 * args.glow

    # 7) 下地（ごく薄い体。線だけだと形が読みにくいので）
    fill_a = body * args.fill * (0.30 + 0.70 * np.clip(Ln, 0, 1) ** 0.8)
    fill_rgb = np.array([0.07, 0.24, 0.42], dtype=np.float32)

    # 写真の残像（任意）: 下地として薄く敷くと、線だけより体の立体が伝わる
    if args.ghost > 0:
        gains = np.array([0.55, 0.82, 1.00], dtype=np.float32)
        ghost_rgb = np.clip(C * gains, 0.0, 1.0) ** 1.15
        ghost_a = As * args.ghost * (0.45 + 0.55 * np.clip(Ln, 0, 1) ** 0.7)
    else:
        ghost_rgb = np.zeros_like(C)
        ghost_a = np.zeros_like(A)

    # 線の色: 暗部はシアン、明部は白っぽく
    cyan = np.array([0.20, 0.84, 1.00], dtype=np.float32)
    ice = np.array([0.92, 0.99, 1.00], dtype=np.float32)
    t = np.clip(Ln * 1.5, 0.0, 1.0)[..., None]
    color = cyan * (1.0 - t) + ice * t

    total = np.clip(light * 0.92 * args.light + glow * args.light, 0.0, 1.0)
    out_a = np.clip(ghost_a + fill_a + total, 0.0, 1.0)
    acc = (ghost_rgb * ghost_a[..., None] + fill_rgb * fill_a[..., None]
           + color * total[..., None])

    # 色ずれ: 線の縁を左右にずらして RGB を分ける（ブラウン管の収束ずれ）
    if args.chroma > 0:
        sh = max(1, int(round(1.2 * args.scale)))
        fr = total * args.chroma * 0.30
        out_a = np.clip(out_a + fr, 0.0, 1.0)
        acc[..., 0] += np.roll(fr, sh, axis=1) * 0.75
        acc[..., 1] += fr * 0.10
        acc[..., 2] += np.roll(fr, -sh, axis=1) * 0.75

    out_rgb = np.clip(acc / np.maximum(out_a, 1e-4)[..., None], 0.0, 1.0)

    # 8) 尾側を闇へ
    if args.tail_fade > 0:
        w0 = out_a.shape[1]
        xr = np.linspace(0.0, 1.0, w0, dtype=np.float32)
        fade = np.clip(xr / max(args.tail_fade, 1e-6), 0.0, 1.0) ** 1.2
        out_a = out_a * (0.12 + 0.88 * fade)[None, :]

    canvas = Image.fromarray((np.clip(np.dstack([out_rgb, out_a]), 0, 1) * 255).astype(np.uint8), 'RGBA')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)

    ys2, xs2 = np.where(out_a > 0.02)
    bx0, bx1 = xs2.min() / args.scale, xs2.max() / args.scale
    by0, by1 = ys2.min() / args.scale, ys2.max() / args.scale
    print(f'canvas {canvas.size} -> viewBox bbox x {bx0:.1f}..{bx1:.1f} y {by0:.1f}..{by1:.1f}')
    print(f'length {bx1 - bx0:.1f}  height {by1 - by0:.1f}  bottom {by1:.1f}')
    print(f'CSS k = bottom/1200 = {by1 / 1200:.4f}  -> min({by1 / 12:.1f}vw, {by1 * 1900 / 1200:.0f}px)')
    print(f'element mask: top {(by0 / VIEW_H) * 100:.1f}%  bottom {(by1 / VIEW_H) * 100:.1f}%')
    print(f'lines: bands={args.bands} blur={args.blur} ss={args.ss} scan={args.scan}')

    if args.webp:
        wp = Path(args.webp)
        canvas.save(wp, 'WEBP', quality=args.webp_quality, method=6, alpha_quality=55, exact=False)
        print(f'webp {wp} {wp.stat().st_size} B  base64 {math.ceil(wp.stat().st_size / 3) * 4} B')


if __name__ == '__main__':
    main()
