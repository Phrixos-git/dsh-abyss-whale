// 深海の大鯨テーマ: 大鯨（ザトウクジラ）の輪郭プロファイル生成スクリプト。
//
// client.js の WHALE_SVG はこのスクリプトの出力を貼り付けたものです。体の各寸法を
// 変えたいときは下のランドマーク（TOP_HEAD / FIN_SEGS / BOTTOM / FLUKE_SEGS /
// flipper まわりの定数）を直して `node tools/whale-profile.mjs` を実行し、
// ---SVG--- 以降を WHALE_SVG に差し替えます。
// テーマの動作にこのファイルは不要です（実行しなくても client.js だけで動きます）。
//
// 出力の前半は QA 用の数値とシルエット。自己交差がないか、viewBox に収まるか、
// 畝・斑紋・結節・陰影線・尾びれの稜線・胸びれの条線が体の輪郭の内側にあるかを
// 確認する（見た目の確認ではない）。
//
// 比率の根拠（ザトウクジラの実測的な特徴）:
//   - 胸びれ（flipper）は全長のほぼ 1/3（最大 30%）と極端に長い → この絵では 0.29 L
//   - 尾びれ（fluke）は幅が全長の約 1/3 と広く、中央に深い切れ込み、後縁は細かく鋸歯状
//   - 背びれは 2/3 付近の「こぶ」の上にある小さな鎌状のもの、その後ろに段（step）
//   - 腹側の畝（ventral grooves）は 14〜35 本、顎の先からへそのあたりまで
//   - 吻（rostrum）と下顎には結節（tubercles）が並び、頭部にはフジツボが付く
//   - 尾びれには付け根から後縁へ走る稜線、胸びれには長軸方向の条線
// 出典の目安:
//   https://www.seawatchfoundation.org.uk/wp-content/uploads/2025/10/Humpback-Whale.pdf
//   https://www.nps.gov/kefj/learn/nature/humpback-whale.htm
//   https://crru.org.uk/education/species/humpback-whale
//   https://www.oceanicsociety.org/resources/whale-fluke-identification-guide/

const f = (v) => String(Math.round(v * 10) / 10);
const dist = (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1]);

function crSegs(pts, alpha = 0.5) {
  const n = pts.length;
  const p = [pts[0], ...pts, pts[n - 1]];
  const out = [];
  for (let i = 0; i < n - 1; i++) {
    const p0 = p[i], p1 = p[i + 1], p2 = p[i + 2], p3 = p[i + 3];
    const d1 = Math.pow(Math.max(dist(p0, p1), 1e-6), alpha);
    const d2 = Math.pow(Math.max(dist(p1, p2), 1e-6), alpha);
    const d3 = Math.pow(Math.max(dist(p2, p3), 1e-6), alpha);
    out.push({
      c1: [p1[0] + (p2[0] - p0[0]) * d1 / (3 * (d1 + d2)), p1[1] + (p2[1] - p0[1]) * d1 / (3 * (d1 + d2))],
      c2: [p2[0] - (p3[0] - p1[0]) * d2 / (3 * (d2 + d3)), p2[1] - (p3[1] - p1[1]) * d2 / (3 * (d2 + d3))],
      to: p2,
    });
  }
  return out;
}
const seg = (c1, c2, to) => ({ c1, c2, to });
const pack = (segs) => segs.map((s) => `C${f(s.c1[0])} ${f(s.c1[1])} ${f(s.c2[0])} ${f(s.c2[1])} ${f(s.to[0])} ${f(s.to[1])}`).join('');

// ---- 体の輪郭 -------------------------------------------------------------
// 吻端は右（x=1152）、尾柄は左（x=272）。全長 L = 1152 - 272 = 880。
const SNOUT = [1152, 268];
const TOP_HEAD = [
  // 吻の上端（広く丸い吻）から背中へ
  [1152, 268], [1138, 256], [1114, 248], [1082, 242], [1046, 237],
  [1008, 233], [968, 229], [928, 226], [886, 222], [842, 218],
  [798, 214], [754, 209], [712, 203], [672, 196], [640, 189],
];
// 背中のこぶ → 小さな鎌状の背びれ → 背びれの後ろの段（step）
const FIN_SEGS = [
  seg([630, 186], [614, 183], [598, 181]),  // 背中の「こぶ」の頂点（背びれの前付け根）
  seg([584, 174], [560, 164], [546, 158]),  // 背びれ前縁（後方へ傾く鎌状）
  seg([540, 172], [530, 182], [516, 190]),  // 背びれ後縁
  seg([506, 196], [494, 199], [480, 200]),  // 背びれの後ろの段（step）
  seg([470, 200], [460, 201], [448, 202]),
];
const TOP_TAIL = [
  [448, 202], [410, 209], [372, 217], [334, 225], [300, 233], [272, 241],
];
// 腹側（吻端 → 尾柄の順。bottomY() の補間にも使う）
const BOTTOM = [
  [1146, 284], [1138, 295], [1124, 302], [1100, 306], [1072, 310], [1042, 312],
  [1010, 313], [982, 313], [950, 315], [916, 318], [880, 326], [844, 334],
  [806, 342], [768, 348], [730, 352], [690, 353], [650, 351], [610, 347],
  [570, 343], [530, 338], [490, 333], [450, 328], [410, 322], [372, 316],
  [336, 310], [300, 303], [272, 296],
];

// 尾びれ: 幅 226（0.257 L）／先端は尖り／後縁は細かく鋸歯状／中央の切れ込み。
// 切れ込みは弦長 194 の約 38%（先端 x=78 から 74 だけ前方）まで。深すぎると
// 人魚の尾になってしまうため、実際のザトウクジラの切れ込みの深さに合わせた。
const FLUKE_TIP_UP = [78, 155];
const FLUKE_TIP_LO = [78, 381];
const FLUKE_NOTCH = [152, 268];

// 弦 a→b に沿って interior 側へ amp だけ張り出す波（後縁の鋸歯）。
function serrated(a, b, bumps, amp, interior) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  const ux = dx / len, uy = dy / len;
  let nx = -uy, ny = ux;
  const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
  if ((interior[0] - mx) * nx + (interior[1] - my) * ny < 0) { nx = -nx; ny = -ny; }
  const N = bumps * 6 + 1;
  const pts = [];
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1);
    const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * bumps * t);
    pts.push([a[0] + dx * t + nx * amp * w, a[1] + dy * t + ny * amp * w]);
  }
  return pts;
}

const FLUKE_TRAIL_UP = serrated(FLUKE_TIP_UP, FLUKE_NOTCH, 7, 4, [190, 215]);
const FLUKE_TRAIL_LO = serrated(FLUKE_NOTCH, FLUKE_TIP_LO, 7, 4, [186, 322]);
const FLUKE_SEGS = [
  // 上の前縁（尾柄 → 上の先端）
  ...crSegs([[272, 241], [252, 232], [224, 221], [196, 208], [166, 192], [136, 174], [110, 161], FLUKE_TIP_UP]),
  // 上の後縁（上端 → 切れ込み、鋸歯）
  ...crSegs(FLUKE_TRAIL_UP),
  // 下の後縁（切れ込み → 下の先端、鋸歯）
  ...crSegs(FLUKE_TRAIL_LO),
  // 下の前縁（下の先端 → 尾柄）
  ...crSegs([FLUKE_TIP_LO, [110, 376], [136, 362], [166, 347], [196, 331], [224, 318], [252, 306], [272, 296]]),
];

const bodySegs = [
  ...crSegs(TOP_HEAD), ...FIN_SEGS, ...crSegs(TOP_TAIL), ...FLUKE_SEGS,
  ...crSegs([...BOTTOM].reverse()),
  seg([1154, 280], [1156, 270], SNOUT),
];
const bodyD = `M${f(SNOUT[0])} ${f(SNOUT[1])}` + pack(bodySegs);

// ---- 幾何の補助 -----------------------------------------------------------
function sampleCubic(p0, c1, c2, p3, t) {
  const u = 1 - t;
  return [u*u*u*p0[0] + 3*u*u*t*c1[0] + 3*u*t*t*c2[0] + t*t*t*p3[0],
          u*u*u*p0[1] + 3*u*u*t*c1[1] + 3*u*t*t*c2[1] + t*t*t*p3[1]];
}
function flatten(start, segs, steps = 16) {
  const out = [start]; let cur = start;
  for (const s of segs) {
    for (let i = 1; i <= steps; i++) out.push(sampleCubic(cur, s.c1, s.c2, s.to, i / steps));
    cur = s.to;
  }
  return out;
}
const body = flatten(SNOUT, bodySegs);
function inside(poly, pt) {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i], b = poly[j];
    if ((a[1] > pt[1]) !== (b[1] > pt[1]) && pt[0] < ((b[0]-a[0])*(pt[1]-a[1]))/(b[1]-a[1]) + a[0]) c = !c;
  }
  return c;
}
function lineY(landmarks, x) {
  const L = [...landmarks].sort((a, b) => a[0] - b[0]);
  if (x <= L[0][0]) return L[0][1];
  for (let i = 0; i < L.length - 1; i++) {
    if (x >= L[i][0] && x <= L[i+1][0]) {
      const t = (x - L[i][0]) / (L[i+1][0] - L[i][0]);
      return L[i][1] + (L[i+1][1] - L[i][1]) * t * t * (3 - 2 * t);
    }
  }
  return L[L.length - 1][1];
}
const TOPLINE = [...TOP_HEAD];
const bottomY = (x) => lineY(BOTTOM, x);
const topY = (x) => lineY(TOPLINE, x);
// 背びれ・尾びれも含めた上端／下端（陰影線を輪郭の内側に沿わせるため）。
function edgeY(x, dir) {
  let y = null;
  for (const p of body) {
    if (Math.abs(p[0] - x) > 4) continue;
    if (y === null || (dir < 0 ? p[1] < y : p[1] > y)) y = p[1];
  }
  return y === null ? (dir < 0 ? 158 : 380) : y;
}
const topAll = (x) => edgeY(x, -1);
const botAll = (x) => edgeY(x, 1);
// a→b を滑らかに 0→1 へ（a と b の大小は問わない）。
function ramp(v, a, b) {
  const t = Math.min(1, Math.max(0, (v - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

// ---- 腹側の畝（18本） -----------------------------------------------------
// 側面視では顎の先で一点に集まり、腹側の広がりに沿って後方（へそのあたり）へ
// 扇状に走る。本数は実測の 14〜35 本の範囲から 18 本。
const pleats = [];
const NPR = 18;
for (let k = 0; k < NPR; k++) {
  const s = k / (NPR - 1);
  const startX = 1126 - 4 * s;
  const endX = 748 - 9 * s;
  const insetStart = 3 + 3 * s;
  const insetEnd = 5 + 29 * s;
  const N = 14;
  const pts = [];
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1);
    const g = t * t * (3 - 2 * t);
    const x = startX + (endX - startX) * t;
    const inset = insetStart + (insetEnd - insetStart) * g;
    pts.push([x, bottomY(x) - inset]);
  }
  pleats.push({ d: `M${f(pts[0][0])} ${f(pts[0][1])}${pack(crSegs(pts))}`, pts });
}

// ---- 斑紋（背中のまだら／腹側の白い斑） -----------------------------------
const mottles = [
  [1080, 262, 20, 8], [1012, 254, 24, 9], [950, 248, 22, 8], [890, 244, 25, 9],
  [826, 240, 21, 8], [764, 234, 27, 10], [700, 229, 22, 8], [640, 226, 26, 9],
  [584, 226, 20, 7], [524, 236, 18, 7], [464, 246, 15, 6], [410, 254, 13, 5],
];
// 腹側の白斑（塊）。ぼかして淡く重ねる。
const bellyMottles = [
  [1104, 291, 18, 6], [1070, 297, 22, 7], [1034, 301, 20, 6], [998, 305, 24, 8],
  [960, 307, 20, 7], [922, 311, 24, 8], [884, 317, 22, 8], [846, 325, 20, 7],
  [808, 331, 18, 6], [770, 337, 16, 6], [732, 341, 14, 5], [694, 345, 12, 5],
];

// ---- 形の陰影（輪郭に沿う内側の線） ---------------------------------------
// 光の当たる背の稜線と、腹側の丸みを内側からなぞる線。端は細く消す。
const CONTOURS = [];
{
  const pts = [];
  for (let x = 1108; x >= 616; x -= 8) {
    const off = 4 + 5 * ramp(x, 1108, 1030) * ramp(x, 616, 660);
    pts.push([x, topAll(x) + off]);
  }
  CONTOURS.push({ pts, d: `M${f(pts[0][0])} ${f(pts[0][1])}${pack(crSegs(pts))}` });
}
{
  const pts = [];
  for (let x = 1112; x >= 640; x -= 8) {
    const off = 4 + 5 * ramp(x, 1112, 1040) * ramp(x, 640, 720);
    pts.push([x, botAll(x) - off]);
  }
  CONTOURS.push({ pts, d: `M${f(pts[0][0])} ${f(pts[0][1])}${pack(crSegs(pts))}` });
}
{
  // 背びれの後ろ（尾柄）の稜線
  const pts = [];
  for (let x = 500; x >= 400; x -= 8) {
    const off = 3 + 5 * ramp(x, 500, 470) * ramp(x, 400, 430);
    pts.push([x, topAll(x) + off]);
  }
  CONTOURS.push({ pts, d: `M${f(pts[0][0])} ${f(pts[0][1])}${pack(crSegs(pts))}` });
}

// ---- 尾びれの稜線（付け根から後縁へ） -------------------------------------
const ptAt = (list, t) => {
  const i = t * (list.length - 1);
  const i0 = Math.floor(i), i1 = Math.min(list.length - 1, i0 + 1);
  const u = i - i0;
  return [list[i0][0] + (list[i1][0] - list[i0][0]) * u, list[i0][1] + (list[i1][1] - list[i0][1]) * u];
};
const FLUKE_ROOT = [238, 268];
const flukeRidges = [];
for (let i = 0; i < 5; i++) {
  const t = 0.12 + 0.19 * i;
  const targets = [ptAt(FLUKE_TRAIL_UP, t), ptAt(FLUKE_TRAIL_LO, t)];
  for (const target of targets) {
    const end = [FLUKE_ROOT[0] + (target[0] - FLUKE_ROOT[0]) * 0.93,
                 FLUKE_ROOT[1] + (target[1] - FLUKE_ROOT[1]) * 0.93];
    flukeRidges.push({
      pts: [FLUKE_ROOT, [(FLUKE_ROOT[0] + end[0]) / 2, (FLUKE_ROOT[1] + end[1]) / 2], end],
      d: `M${f(FLUKE_ROOT[0])} ${f(FLUKE_ROOT[1])}L${f(end[0])} ${f(end[1])}`,
    });
  }
}
// 尾びれ中央の稜線（尾柄 → 切れ込み）
flukeRidges.push({ pts: [[262, 268], [238, 268], [214, 268]], d: 'M262 268L214 268' });

// ---- 口線・眼・噴気孔 -----------------------------------------------------
// 口線は吻端から長く後方へ弧を描き、口角（眼のすぐ下）で上がる。
const MOUTH_SEGS = [
  seg([1132, 284], [1108, 288], [1080, 289]),
  seg([1046, 289], [1012, 285], [976, 279]),
  seg([950, 274], [920, 269], [892, 268]),
];
const MOUTH_D = 'M1150 272' + pack(MOUTH_SEGS);
const mouthLine = flatten([1150, 272], MOUTH_SEGS, 24);
const mouthY = (x) => lineY(mouthLine.map((p) => [p[0], p[1]]), x);
const MOUTH_CORNER = [892, 268];
const EYE = [872, 256];
const EYE_BROW = `M${f(EYE[0] - 8)} ${f(EYE[1] - 6)}Q${f(EYE[0])} ${f(EYE[1] - 11)} ${f(EYE[0] + 8)} ${f(EYE[1] - 6)}`;
const BLOWHOLE = 'M906 235Q911 229 916 235M922 235Q927 229 932 235';

// ---- 顎のしわ（顎先に集まる短い線） ---------------------------------------
const chinGrooves = [];
for (let i = 0; i < 5; i++) {
  const a = [1146, 281];
  const x1 = 1120 - 9 * i;
  const b = [x1, mouthY(x1) + 8 + 1.6 * i];
  chinGrooves.push({
    pts: [a, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 4], b],
    d: `M${f(a[0])} ${f(a[1])}Q${f((a[0] + b[0]) / 2)} ${f((a[1] + b[1]) / 2 + 4)} ${f(b[0])} ${f(b[1])}`,
  });
}
const CHIN_D = chinGrooves.map((g) => g.d).join('');

// ---- 吻と下顎の結節（tubercles） -----------------------------------------
const knob = (x, y, r, up) => (up
  ? `M${f(x - r)} ${f(y + r * 0.6)}Q${f(x)} ${f(y - r * 0.9)} ${f(x + r)} ${f(y + r * 0.6)}`
  : `M${f(x - r)} ${f(y - r * 0.6)}Q${f(x)} ${f(y + r * 0.9)} ${f(x + r)} ${f(y - r * 0.6)}`);
const tubercles = [];
const tubercleMarks = [];
const addTubercle = (x, y, r, up) => {
  tubercles.push(knob(x, y, r, up));
  tubercleMarks.push([x, y, r, up]);
};
// 吻の背（正中と側面の2列）
for (let i = 0; i < 6; i++) { const x = 1134 - 17 * i; addTubercle(x, topY(x) + 7, 4.6, true); }
for (let i = 0; i < 4; i++) { const x = 1122 - 19 * i; addTubercle(x, topY(x) + 15, 3.8, true); }
// 下顎（口線のすぐ下）
for (let i = 0; i < 5; i++) { const x = 1122 - 13 * i; addTubercle(x, mouthY(x) + 10, 4.2, false); }
// 口角の後ろの小さな結節
addTubercle(906, 262, 3.4, false);
addTubercle(884, 258, 3.0, false);
const TUBERCLE_D = tubercles.join('');

// ---- 胸びれ（0.29 L、前縁に結節、先端は尖る） ----------------------------
// ザトウクジラの胸びれは全長の約 1/3。付け根は眼のすぐ後下（肩）、先端は
// 胴の中央下まで後方へ流し、腹の輪郭から下へ突き出す。前縁には結節の凹凸。
const flipper = (() => {
  const root = [890, 304], tip = [660, 414];
  const dx = tip[0] - root[0], dy = tip[1] - root[1];
  const len = Math.hypot(dx, dy);
  const ux = dx / len, uy = dy / len;
  const nx = -uy, ny = ux; // 前縁側（上）
  // 弦の半幅（先端へ向けて細る）。最大弦 31 単位 ≒ 0.12 L。
  const HW = [[0, 13], [0.18, 15.5], [0.36, 14], [0.55, 11], [0.75, 7], [0.92, 3.4], [1, 1.4]];
  const hw = (t) => {
    for (let i = 0; i < HW.length - 1; i++) {
      const [t0, w0] = HW[i], [t1, w1] = HW[i + 1];
      if (t >= t0 && t <= t1) return w0 + (w1 - w0) * (t - t0) / (t1 - t0);
    }
    return 1.4;
  };
  const P = (t, side, extra = 0) => [
    root[0] + ux * len * t + nx * side * (hw(t) + extra),
    root[1] + uy * len * t + ny * side * (hw(t) + extra),
  ];
  const LE = [0, 0.13, 0.26, 0.39, 0.52, 0.65, 0.78, 0.9, 1];
  const TE = [1, 0.86, 0.7, 0.54, 0.38, 0.22, 0.08, 0];
  const a0 = P(0, 1);
  let d = `M${f(a0[0])} ${f(a0[1])}`;
  for (let i = 1; i < LE.length; i++) {
    const a = P(LE[i - 1], 1), b = P(LE[i], 1);
    // 前縁の結節: 制御点を外側へずらして凹凸を出す
    const cx = (a[0] + b[0]) / 2 + nx * 8, cy = (a[1] + b[1]) / 2 + ny * 8;
    d += `Q${f(cx)} ${f(cy)} ${f(b[0])} ${f(b[1])}`;
  }
  // 先端を丸める
  const tipL = P(1, 1), tipR = P(1, -1);
  d += `C${f(tipL[0] + ux * 5)} ${f(tipL[1] + uy * 5)} ${f(tipR[0] + ux * 5)} ${f(tipR[1] + uy * 5)} ${f(tipR[0])} ${f(tipR[1])}`;
  // 後縁は滑らかな曲線（スプライン）で戻す
  d += pack(crSegs(TE.map((t) => P(t, -1))));
  d += 'Z';
  // QA 用に外形のサンプル点（前縁は結節の張り出しを含む）
  const pts = [];
  for (let i = 0; i <= 24; i++) {
    const t = i / 24;
    pts.push(P(t, 1, 8), P(t, -1, 0));
  }
  // 条線（fin rays）: 長軸方向に走る細い線
  const rays = [];
  for (const s0 of [-0.62, -0.41, -0.2, 0, 0.2, 0.41, 0.62]) {
    const p = [];
    for (let t = 0.28; t <= 0.961; t += 0.11) p.push(P(t, s0));
    rays.push({ pts: p, d: `M${f(p[0][0])} ${f(p[0][1])}${pack(crSegs(p))}` });
  }
  // 前縁の結節（丸いこぶ）の位置
  const knobs = [0.09, 0.22, 0.35, 0.48, 0.61, 0.74, 0.87].map((t) => P(t, 1, 1.6));
  // 先端側に付くフジツボ
  const flecks = [P(0.72, 0.4), P(0.8, -0.2), P(0.88, 0.25)];
  return { d, root, tip, len, pts, P, hw, rays, knobs, flecks };
})();
const FLIPPER_D = flipper.d;

// ---- 頭部・尾びれのフジツボ ----------------------------------------------
const barnacles = [
  [1124, topY(1124) + 6, 2.1, 0.22],
  [1103, topY(1103) + 7, 1.7, 0.18],
  [1082, topY(1082) + 8, 1.4, 0.15],
  [1116, mouthY(1116) + 6, 1.9, 0.18],
  [1094, mouthY(1094) + 7, 1.5, 0.15],
  [92, 165, 1.6, 0.18],
  [92, 370, 1.6, 0.18],
];

// ---- QA -------------------------------------------------------------------
const bbox = body.reduce((b, p) => ({ x0: Math.min(b.x0,p[0]), x1: Math.max(b.x1,p[0]), y0: Math.min(b.y0,p[1]), y1: Math.max(b.y1,p[1]) }), { x0: 1e9, x1: -1e9, y0: 1e9, y1: -1e9 });
function segInt(a,b,c,d){const s=(p,q,r)=>(q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0]);const d1=s(c,d,a),d2=s(c,d,b),d3=s(a,b,c),d4=s(a,b,d);return ((d1>0&&d2<0)||(d1<0&&d2>0))&&((d3>0&&d4<0)||(d3<0&&d4>0));}
let crossings = 0;
for (let i = 0; i < body.length; i++) for (let j = i+2; j < body.length; j++) {
  if (i === 0 && j === body.length-1) continue;
  if (segInt(body[i], body[(i+1)%body.length], body[j], body[(j+1)%body.length])) crossings++;
}
const L = 1152 - 272; // 吻端 → 尾柄
// 体高は背びれ・尾びれを含めず、胴の範囲（x >= 680）で測る。
let maxDepth = 0, at = 0;
for (let x = 680; x <= 1120; x += 2) {
  const ys = body.filter((p) => Math.abs(p[0]-x) < 2.5).map((p) => p[1]);
  if (ys.length > 1) { const d = Math.max(...ys) - Math.min(...ys); if (d > maxDepth) { maxDepth = d; at = x; } }
}
let finTop = 1e9, finTopX = 0;
for (const p of body) if (p[0] > 500 && p[0] < 620 && p[1] < finTop) { finTop = p[1]; finTopX = p[0]; }
function maskAlpha(yPct) {
  // CSS mask: linear-gradient(to bottom, transparent 18%, rgba(0,0,0,0.5) 36%, #000 64%)
  if (yPct <= 18) return 0;
  if (yPct <= 36) return 0.5 * (yPct - 18) / 18;
  if (yPct <= 64) return 0.5 + 0.5 * (yPct - 36) / 28;
  return 1;
}
const bad = (list) => (pts) => pts.filter((q) => !inside(body, q)).map((q) => [+q[0].toFixed(1), +q[1].toFixed(1)]);
const badP = pleats.map((p,k) => ({ k, out: bad(p.pts)(p.pts) })).filter((r) => r.out.length);
const badC = CONTOURS.map((c, k) => ({ k, out: bad(c.pts)(c.pts) })).filter((r) => r.out.length);
const badFR = flukeRidges.map((c, k) => ({ k, out: bad(c.pts)(c.pts) })).filter((r) => r.out.length);
const badCG = chinGrooves.map((c, k) => ({ k, out: bad(c.pts)(c.pts) })).filter((r) => r.out.length);
const badTB = tubercleMarks.map((m, k) => ({ k, pt: [m[0], m[1] + (m[3] ? -m[2] * 0.9 : m[2] * 0.9)] }))
  .filter((r) => !inside(body, r.pt));
const badBN = barnacles.filter((b) => !inside(body, [b[0], b[1]]));
const badM = mottles.filter(([x,y,rx,ry]) => !inside(body,[x,y]) || !inside(body,[x+rx*0.7,y]) || !inside(body,[x-rx*0.7,y]) || !inside(body,[x,y+ry*0.6]) || !inside(body,[x,y-ry*0.6]));
const badBM = bellyMottles.filter(([x,y,rx,ry]) => !inside(body,[x,y]) || !inside(body,[x+rx*0.7,y]) || !inside(body,[x-rx*0.7,y]) || !inside(body,[x,y+ry*0.5]) || !inside(body,[x,y-ry*0.5]));
// 胸びれの外形が viewBox 内か
const flipPts = flipper.pts;
const badFlip = flipPts.filter(([x, y]) => x < 0 || x > 1200 || y < 0 || y > 560);
const badRays = flipper.rays.flatMap((r) => r.pts).filter(([x, y]) => x < 0 || x > 1200 || y < 0 || y > 560);
const badKnobs = flipper.knobs.filter(([x, y]) => x < 0 || x > 1200 || y < 0 || y > 560);
const flipLowest = flipPts.reduce((m, p) => Math.max(m, p[1]), -1e9);
const flipHighest = flipPts.reduce((m, p) => Math.min(m, p[1]), 1e9);
const artLowest = Math.max(bbox.y1, flipLowest);
const artTop = Math.min(bbox.y0, flipHighest);
console.log(JSON.stringify({
  bbox: { x0: +bbox.x0.toFixed(1), x1: +bbox.x1.toFixed(1), y0: +bbox.y0.toFixed(1), y1: +bbox.y1.toFixed(1) },
  crossings,
  L,
  maxDepth: +maxDepth.toFixed(1),
  maxDepthAt: at,
  depthRatio: +(maxDepth / L).toFixed(3),
  flukeSpan: FLUKE_TIP_LO[1] - FLUKE_TIP_UP[1],
  flukeSpanRatio: +((FLUKE_TIP_LO[1] - FLUKE_TIP_UP[1]) / L).toFixed(3),
  flukeChord: 272 - FLUKE_TIP_UP[0],
  notchDepth: FLUKE_NOTCH[0] - FLUKE_TIP_UP[0],
  notchDepthRatio: +((FLUKE_NOTCH[0] - FLUKE_TIP_UP[0]) / (272 - FLUKE_TIP_UP[0])).toFixed(3),
  finApex: [+finTopX.toFixed(1), +finTop.toFixed(1)],
  finBaseCenterFromSnout: +(((1152 - 557) / L)).toFixed(3),
  headToGape: +(((1152 - MOUTH_CORNER[0]) / L)).toFixed(3),
  flipperLen: +flipper.len.toFixed(1),
  flipperLenRatio: +(flipper.len / L).toFixed(3),
  flipperTip: [+flipper.tip[0].toFixed(1), +flipper.tip[1].toFixed(1)],
  vboxTopPct: +(bbox.y0 / 560 * 100).toFixed(1),
  vboxBottomPct: +(bbox.y1 / 560 * 100).toFixed(1),
  // .aw-whale の mask（transparent 18% → 0.5 36% → #000 64%）に対する位置
  bodyTopMaskAlpha: +maskAlpha(bbox.y0 / 560 * 100).toFixed(2),
  bodyBottomMaskAlpha: +maskAlpha(bbox.y1 / 560 * 100).toFixed(2),
  // CSS の top: calc(95% - min(35vw, 665px)) が前提にする「幅に対する下端位置」。
  artLowest: +artLowest.toFixed(1),
  artTop: +artTop.toFixed(1),
  artBottomPct: +((artLowest + 3.5) / 560 * 100).toFixed(1),
  artBottomFracOfWidth: +((artLowest + 3.5) / 1200).toFixed(3),
  badMottles: badM,
  badBellyMottles: badBM,
  badPleatPoints: badP,
  badContours: badC,
  badFlukeRidges: badFR,
  badChinGrooves: badCG,
  badTubercles: badTB,
  badBarnacles: badBN,
  mouthIns: inside(body, MOUTH_CORNER),
  eyeIns: inside(body, EYE),
  blowholeIns: inside(body, [919, 233]),
  tubercleCount: tubercles.length,
  chinGrooveCount: chinGrooves.length,
  barnacleCount: barnacles.length,
  badFlipperPts: badFlip,
  flipperYSpan: [+flipHighest.toFixed(1), +flipLowest.toFixed(1)],
  badFinRays: badRays,
  badFlipperKnobs: badKnobs,
  pleatCount: pleats.length,
  ridgeCount: flukeRidges.length,
  bodyDlen: bodyD.length,
  pleatLens: pleats.map(p=>p.d.length),
}, null, 1));

// ---- シルエット（幾何の目視確認用。見た目の確認ではない） -------------------
const W = 200, cellW = (bbox.x1 - bbox.x0) / W, cellH = cellW * 2;
const H = Math.ceil((bbox.y1 - bbox.y0) / cellH);
const rows = [];
for (let r = 0; r < H; r++) {
  let line = '';
  for (let c = 0; c < W; c++) {
    line += inside(body, [bbox.x0 + (c+0.5)*cellW, bbox.y0 + (r+0.5)*cellH]) ? '#' : ' ';
  }
  rows.push(line);
}
function put(x, y, ch) {
  const c = Math.floor((x - bbox.x0) / cellW), r = Math.floor((y - bbox.y0) / cellH);
  if (rows[r]) { const s = rows[r].split(''); s[c] = ch; rows[r] = s.join(''); }
}
for (const [x,y] of mottles) put(x, y, 'o');
for (const p of pleats) for (const [x,y] of p.pts) put(x, y, '-');
for (const c of CONTOURS) for (const [x,y] of c.pts) put(x, y, '~');
for (const c of flukeRidges) for (const [x,y] of c.pts) put(x, y, '*');
put(EYE[0], EYE[1], 'E'); put(MOUTH_CORNER[0], MOUTH_CORNER[1], 'M');
put(558, 160, 'F'); put(flipper.tip[0], flipper.tip[1], 'P'); put(FLUKE_NOTCH[0], FLUKE_NOTCH[1], 'N');
console.log(rows.join('\n'));

// ---- 最終的な SVG ---------------------------------------------------------
const f1 = (v) => String(Math.round(v * 10) / 10);
const mottleSvg = mottles.map(([x, y, rx, ry]) =>
  `    <ellipse cx="${f1(x)}" cy="${f1(y)}" rx="${f1(rx)}" ry="${f1(ry)}"/>`).join('\n');
const bellySvg = bellyMottles.map(([x, y, rx, ry]) =>
  `      <ellipse cx="${f1(x)}" cy="${f1(y)}" rx="${f1(rx)}" ry="${f1(ry)}"/>`).join('\n');
const pleatSvg = pleats.map((p) => `    <path d="${p.d}"/>`).join('\n');
const contourSvg = CONTOURS.map((c) => `    <path d="${c.d}"/>`).join('\n');
const ridgeSvg = flukeRidges.map((c) => `    <path d="${c.d}"/>`).join('\n');
const finRaySvg = flipper.rays.map((r) => `      <path d="${r.d}"/>`).join('\n');
const flipperFleckSvg = flipper.flecks.map(([x, y]) => `      <circle cx="${f1(x)}" cy="${f1(y)}" r="1.7"/>`).join('\n');
const knobSvg = flipper.knobs.map(([x, y]) => `  <circle cx="${f1(x)}" cy="${f1(y)}" r="2.3"/>`).join('\n');
const barnacleSvg = barnacles.map(([x, y, r, o]) =>
  `      <circle cx="${f1(x)}" cy="${f1(y)}" r="${f1(r)}" fill-opacity="${o}"/>`).join('\n');

const svg = `<svg viewBox="0 0 1200 560" xmlns="http://www.w3.org/2000/svg" focusable="false" aria-hidden="true">
  <defs>
    <linearGradient id="aw-whale-body" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#9fe8ff" stop-opacity="0.13"/>
      <stop offset="0.46" stop-color="#4fb4e0" stop-opacity="0.075"/>
      <stop offset="1" stop-color="#1a4f80" stop-opacity="0.025"/>
    </linearGradient>
    <linearGradient id="aw-whale-rim" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#c8f8ff" stop-opacity="0.6"/>
      <stop offset="0.52" stop-color="#58c8f5" stop-opacity="0.3"/>
      <stop offset="1" stop-color="#2b7fa6" stop-opacity="0.08"/>
    </linearGradient>
    <linearGradient id="aw-whale-fin" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#9fe8ff" stop-opacity="0.08"/>
      <stop offset="1" stop-color="#2b7fa6" stop-opacity="0.02"/>
    </linearGradient>
    <linearGradient id="aw-whale-fin-white" x1="0.12" y1="0.1" x2="0.9" y2="0.95">
      <stop offset="0" stop-color="#eafcff" stop-opacity="0"/>
      <stop offset="0.4" stop-color="#eafcff" stop-opacity="0.05"/>
      <stop offset="1" stop-color="#f2feff" stop-opacity="0.13"/>
    </linearGradient>
    <linearGradient id="aw-whale-belly" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#dff7ff" stop-opacity="0"/>
      <stop offset="0.34" stop-color="#cfefff" stop-opacity="0.03"/>
      <stop offset="1" stop-color="#eafcff" stop-opacity="0.1"/>
    </linearGradient>
    <radialGradient id="aw-whale-halo" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0" stop-color="#4ec8ff" stop-opacity="0.12"/>
      <stop offset="1" stop-color="#4ec8ff" stop-opacity="0"/>
    </radialGradient>
    <filter id="aw-whale-blur" x="-15%" y="-15%" width="130%" height="130%">
      <feGaussianBlur stdDeviation="9"/>
    </filter>
    <clipPath id="aw-whale-clip">
      <path d="${bodyD}"/>
    </clipPath>
    <clipPath id="aw-whale-flipper-clip">
      <path d="${FLIPPER_D}"/>
    </clipPath>
  </defs>

  <ellipse cx="610" cy="280" rx="600" ry="230" fill="url(#aw-whale-halo)"/>

  <path d="${bodyD}" fill="url(#aw-whale-body)"/>

  <g clip-path="url(#aw-whale-clip)">
    <rect x="0" y="240" width="1200" height="320" fill="url(#aw-whale-belly)"/>

    <g fill="#bfeaff" fill-opacity="0.055">
${mottleSvg}
    </g>

    <g fill="#eafcff" fill-opacity="0.05" filter="url(#aw-whale-blur)">
${bellySvg}
    </g>

    <g fill="none" stroke="#9fe8ff" stroke-opacity="0.15" stroke-width="1.9" stroke-linecap="round">
${pleatSvg}
    </g>

    <g fill="none" stroke="#bfeaff" stroke-opacity="0.09" stroke-width="1.8" stroke-linecap="round">
${contourSvg}
    </g>

    <g fill="none" stroke="#dff7ff" stroke-opacity="0.1" stroke-width="1.6" stroke-linecap="round">
${ridgeSvg}
    </g>

    <path d="${MOUTH_D}" fill="none" stroke="#c9f6ff" stroke-opacity="0.3" stroke-width="2.2" stroke-linecap="round"/>
    <path d="${EYE_BROW}" fill="none" stroke="#c9f6ff" stroke-opacity="0.18" stroke-width="1.5" stroke-linecap="round"/>
    <path d="${BLOWHOLE}" fill="none" stroke="#c9f6ff" stroke-opacity="0.26" stroke-width="2" stroke-linecap="round"/>

    <g fill="none" stroke="#c9f6ff" stroke-opacity="0.2" stroke-width="1.8" stroke-linecap="round">
${TUBERCLE_D}
    </g>

    <g fill="none" stroke="#bfeaff" stroke-opacity="0.14" stroke-width="1.5" stroke-linecap="round">
${CHIN_D}
    </g>

    <g fill="#dff7ff">
${barnacleSvg}
    </g>
  </g>

  <path d="${bodyD}" fill="none" stroke="#8fe9ff" stroke-opacity="0.2" stroke-width="7" stroke-linejoin="round" filter="url(#aw-whale-blur)"/>

  <path d="${bodyD}" fill="none" stroke="url(#aw-whale-rim)" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>

  <path d="${FLIPPER_D}" fill="url(#aw-whale-fin)"/>
  <g clip-path="url(#aw-whale-flipper-clip)">
    <path d="${FLIPPER_D}" fill="url(#aw-whale-fin-white)"/>
    <g fill="none" stroke="#dff7ff" stroke-opacity="0.1" stroke-width="1.4" stroke-linecap="round">
${finRaySvg}
    </g>
    <g fill="#eafcff" fill-opacity="0.2">
${flipperFleckSvg}
    </g>
  </g>
  <path d="${FLIPPER_D}" fill="none" stroke="url(#aw-whale-rim)" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round"/>
  <path d="${FLIPPER_D}" fill="none" stroke="#8fe9ff" stroke-opacity="0.26" stroke-width="6" stroke-linejoin="round" stroke-linecap="round" filter="url(#aw-whale-blur)"/>
  <g fill="#c9f6ff" fill-opacity="0.22">
${knobSvg}
  </g>

  <circle cx="${EYE[0]}" cy="${EYE[1]}" r="3.6" fill="#eaffff" fill-opacity="0.9"/>
  <circle cx="${EYE[0]}" cy="${EYE[1]}" r="6.4" fill="none" stroke="#c9f6ff" stroke-opacity="0.2" stroke-width="1.2"/>
  <circle cx="${EYE[0]}" cy="${EYE[1]}" r="10" fill="#c9f6ff" fill-opacity="0.1"/>
</svg>`;
// The SVG is printed to stdout: paste it into WHALE_SVG in ../client.js.
console.log('\n---SVG---\n' + svg);
