// Dry-fire flag tracker.
//
// Detects the printed bore-insert flag — a 4×4 checkerboard (8 dark tiles)
// with one round orientation dot — in a grayscale camera frame and reduces it
// to a small feature vector describing the flag's pose.
//
// Deliberately NO full 3D pose recovery (no camera intrinsics, no solvePnP):
// the aim calibration (aimModel.ts) fits a direct regression from these
// features to on-screen aim coordinates, which absorbs the camera intrinsics,
// the screen's pose, the bore-to-card transform, AND the shooter's zero in
// one step. The features just have to respond smoothly and distinctly to the
// flag's translation and tilt — an affine frame fitted to the tile
// constellation does exactly that.
//
// Tile selection is SCALE-FREE: real scenes are full of dark blobs (the gun
// itself, monitor bezels, furniture), so no fixed size band can anchor on
// "the 8 largest". Instead, every blob proposes a scale hypothesis, and a
// valid candidate is any tight cluster of 8 similar-area blobs — each cluster
// is then tried through the full geometric pipeline.
//
// Correspondence is solved projectively: the four tiles on the checker's main
// diagonal — (0,0),(1,1),(2,2),(3,3) — are COLLINEAR, and collinearity
// survives any perspective. That line plus the side split identifies every
// tile, and the dot resolves the remaining 180°/mirror ambiguity.

export type Point = { x: number; y: number };

// Mean observed RGB of each colored quadrant — the card's appearance under
// THIS room's lighting, learned at lock and used by track mode.
export type TrackedColors = {
  red: [number, number, number];
  green: [number, number, number];
  blue: [number, number, number];
};

export type FlagObservation = {
  tiles: Point[]; // 8 dark-tile centroids, ordered by template index
  dot: Point;
  center: Point; // affine origin = pattern center, image px
  cellPx: number; // approximate cell size in image px
  residual: number; // affine fit RMS as a fraction of cellPx (quality gate)
  features: number[]; // [1, bx, by, a11, a12, a21, a22], normalized by normPx
  quadColors?: TrackedColors; // color mode: sampled quadrant appearance
  // Mean RGB read from the card's WHITE quadrant (off the dot) — a built-in
  // gray card: its chroma IS the current illuminant at the card. null when
  // the reading was implausible (glare, occlusion, mis-sample).
  whiteRGB?: [number, number, number] | null;
  // Number of patches synthesized by the rigid-coherence hold (0 = every
  // patch genuinely seen). The caller caps consecutive soft frames.
  softPatches?: number;
};

// Everything the detector saw, stage by stage — for the on-screen diagnostics
// view when the flag "isn't being found". When several clusters were tried,
// the debug reflects the attempt that got FURTHEST through the pipeline.
export type FlagFailStage =
  | "ok"
  | "too-few-blobs"
  | "too-few-tiles"
  | "no-collinear-quad"
  | "bad-side-split"
  | "no-dot-candidates"
  | "affine-fail"
  | "tiny-cell"
  | "dot-mismatch"
  | "high-residual"
  // Color-quadrant card stages:
  | "missing-color"
  | "bad-color-layout"
  // Black/white shape card stages:
  | "missing-shape"
  | "bad-shape-layout";

export type FlagDebug = {
  failStage: FlagFailStage;
  threshold: number; // Otsu gray cut — everything below counts as "dark"
  blobCount: number;
  blobs: { x: number; y: number; area: number }[]; // capped, offset applied
  clusterCount: number; // how many 8-tile cluster hypotheses were tried
  tileCandidates: Point[]; // best attempt's 8 tiles, offset applied
  quad: Point[] | null; // the collinear diagonal, if found
  quadScore: number | null; // lower = straighter (gate: 0.12)
  dotCandidates: Point[];
  dotErrCells: number | null; // dot mismatch in cell units (gate: 0.8)
  cellPx: number | null;
  residual: number | null; // affine RMS in cells (gate: 0.35)
  // Color-quadrant mode: how many blobs each classifier found.
  colorCounts: { red: number; green: number; blue: number } | null;
  // Color-quadrant mode: which layout gate combinations died at, plus where
  // the dot was EXPECTED for the best geometric triple.
  colorGates: {
    triples: number; // R×G×B combinations tried
    sizeOk: number; // survived the quadrant size-similarity gate
    areaOk: number; // survived the blob-area vs affine-scale gate
    nearestDotErrCells: number | null; // best dot miss distance, in quadrants
    predictedDot: Point | null; // where the dot should be (best triple)
    dotless: boolean; // accepted via the geometry-only fallback
    strengths: [number, number, number] | null; // chosen triple's color purity (r,g,b)
    chiralityOk: boolean | null; // red→green→blue sweeps the expected direction
    dotBoxHit: boolean; // a blob's bounding box contains the predicted dot
    dotBox: { x: number; y: number; w: number; h: number } | null; // that blob's bbox (full-frame)
    anchors: number | null; // shape card: edge anchor dots found (of 4)
  } | null;
};

// Template: dark-cell centers of the 4×4 checker in card units (1 = one
// cell), centered on the pattern. y is "up the card"; the per-frame affine
// absorbs the image's y-down convention. Order is the canonical index used
// everywhere: 4 diagonal tiles, then the two i−j=+2 tiles, then the two
// i−j=−2 tiles.
const TEMPLATE: readonly (readonly [number, number])[] = [
  [-1.5, -1.5],
  [-0.5, -0.5],
  [0.5, 0.5],
  [1.5, 1.5],
  [0.5, -1.5],
  [1.5, -0.5],
  [-1.5, 0.5],
  [-0.5, 1.5],
];
const DOT_TEMPLATE: readonly [number, number] = [-1.5, 1.5];

type Blob = {
  x: number;
  y: number;
  area: number;
  w: number;
  h: number;
  x0: number;
  y0: number;
  // Diagonal-extreme corners (max/min of x+y and x−y) — the convex quad's
  // corners under ANY rotation, unlike the axis-aligned bbox. Present on
  // blobs from maskBlobs; synthetic blobs omit them.
  corners?: [Point, Point, Point, Point]; // TL, TR, BR, BL order (image coords)
  // How square the blob is, 0..1: pixel count ÷ corner-quad area lands at
  // ~1.0 ONLY for a filled quadrilateral (keystone included). A circle
  // overshoots (~1.57 — its corner quad underestimates it), a concave or
  // sparse blob undershoots (<0.6); squareness measures closeness to 1.
  squareness?: number;
};

// Neutral squareness for synthetic/legacy blobs — neither reward nor damn.
function sqOf(b: Blob): number {
  return b.squareness ?? 0.7;
}

// Build a tracker seed pose from three HUMAN-IDENTIFIED patch centers
// (guided taps on a frozen frame): yellow ("red" slot), cyan ("green"),
// magenta ("blue"), in processing-frame coordinates.
export function seedFromThreePoints(
  yellowPt: Point,
  cyanPt: Point,
  magentaPt: Point,
): { affine: [number, number, number, number]; center: Point; cellPx: number } {
  const tpl = templateFor("cmy");
  const A = affineFrom3Pts(tpl.p1, tpl.p2, tpl.p3, yellowPt, cyanPt, magentaPt);
  const cellPx = (Math.hypot(A.a11, A.a21) + Math.hypot(A.a12, A.a22)) / 2;
  return { affine: [A.a11, A.a12, A.a21, A.a22], center: { x: A.bx, y: A.by }, cellPx };
}

// "Is this spot a BRIGHT, NEUTRAL (white) patch?" — sampled with stride 2,
// scored +1 for clean white … −1 for strongly colored. The card's fourth
// quadrant is its most unusual feature: backgrounds that fake three color
// squares almost never also present a white square completing the layout.
// Used as a SCORING bonus in both acquisition paths, never a gate.
function whiteQuadScore(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  cx: number,
  cy: number,
  half: number,
): number {
  const x0 = Math.max(0, Math.round(cx - half));
  const x1 = Math.min(width - 1, Math.round(cx + half));
  const y0 = Math.max(0, Math.round(cy - half));
  const y1 = Math.min(height - 1, Math.round(cy + half));
  let cnt = 0;
  let good = 0;
  let colored = 0;
  for (let y = y0; y <= y1; y += 2)
    for (let x = x0; x <= x1; x += 2) {
      const o = (y * width + x) * 4;
      const r = rgba[o];
      const g = rgba[o + 1];
      const b = rgba[o + 2];
      const sum = r + g + b;
      cnt += 1;
      if (sum < 40) continue;
      const dN = Math.hypot(r / sum - 1 / 3, g / sum - 1 / 3);
      if (sum > 330 && dN < 0.08) good += 1;
      else if (dN > 0.14) colored += 1;
    }
  return cnt > 0 ? (good - colored) / cnt : 0;
}

// HARD acquisition gate: a candidate patch big enough to judge must be
// reasonably quad-shaped. Shirts, foliage, and skin make amorphous blobs
// (squareness < ~0.4); a printed patch — even blurred or keystoned —
// stays well above. Small blobs are exempt (shape is noise at that size),
// and TRACKING never uses this (a held lock stays geometry-relaxed).
const SQUARENESS_GATE = 0.45;
function isSquareEnough(b: Blob): boolean {
  return b.area < 60 || (b.squareness ?? 0.7) >= SQUARENESS_GATE;
}

function otsuThreshold(gray: Uint8Array): number {
  const hist = new Uint32Array(256);
  for (let i = 0; i < gray.length; i += 1) hist[gray[i]] += 1;
  const total = gray.length;
  let sum = 0;
  for (let i = 0; i < 256; i += 1) sum += i * hist[i];
  let sumB = 0;
  let wB = 0;
  let best = 0;
  let threshold = 127;
  for (let t = 0; t < 256; t += 1) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > best) {
      best = between;
      threshold = t;
    }
  }
  return threshold;
}

// Dark connected components (4-neighborhood), centroids + bounds.
function darkBlobs(gray: Uint8Array, width: number, height: number, threshold: number, minArea: number): Blob[] {
  const size = width * height;
  const visited = new Uint8Array(size);
  const stack = new Int32Array(size);
  const blobs: Blob[] = [];
  for (let start = 0; start < size; start += 1) {
    if (visited[start] || gray[start] >= threshold) continue;
    let top = 0;
    stack[top++] = start;
    visited[start] = 1;
    let area = 0;
    let sx = 0;
    let sy = 0;
    let minX = width;
    let maxX = 0;
    let minY = height;
    let maxY = 0;
    while (top > 0) {
      const idx = stack[--top];
      const x = idx % width;
      const y = (idx / width) | 0;
      area += 1;
      sx += x;
      sy += y;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (x > 0 && !visited[idx - 1] && gray[idx - 1] < threshold) {
        visited[idx - 1] = 1;
        stack[top++] = idx - 1;
      }
      if (x < width - 1 && !visited[idx + 1] && gray[idx + 1] < threshold) {
        visited[idx + 1] = 1;
        stack[top++] = idx + 1;
      }
      if (y > 0 && !visited[idx - width] && gray[idx - width] < threshold) {
        visited[idx - width] = 1;
        stack[top++] = idx - width;
      }
      if (y < height - 1 && !visited[idx + width] && gray[idx + width] < threshold) {
        visited[idx + width] = 1;
        stack[top++] = idx + width;
      }
    }
    if (area >= minArea) {
      blobs.push({ x: sx / area + 0.5, y: sy / area + 0.5, area, w: maxX - minX + 1, h: maxY - minY + 1, x0: minX, y0: minY });
    }
  }
  return blobs;
}

// Least-squares affine (card units → image px) from ordered correspondences.
type Affine = { a11: number; a12: number; a21: number; a22: number; bx: number; by: number };

function fitAffine(imagePts: Point[]): Affine | null {
  let suu = 0;
  let suv = 0;
  let svv = 0;
  let su = 0;
  let sv = 0;
  const n = imagePts.length;
  let sxu = 0;
  let sxv = 0;
  let sx = 0;
  let syu = 0;
  let syv = 0;
  let sy = 0;
  for (let i = 0; i < n; i += 1) {
    const [u, v] = TEMPLATE[i];
    const { x, y } = imagePts[i];
    suu += u * u;
    suv += u * v;
    svv += v * v;
    su += u;
    sv += v;
    sxu += x * u;
    sxv += x * v;
    sx += x;
    syu += y * u;
    syv += y * v;
    sy += y;
  }
  const m = [
    [suu, suv, su],
    [suv, svv, sv],
    [su, sv, n],
  ];
  const solve3 = (r0: number, r1: number, r2: number): [number, number, number] | null => {
    const a = m.map((row) => row.slice());
    const b = [r0, r1, r2];
    for (let col = 0; col < 3; col += 1) {
      let pivot = col;
      for (let row = col + 1; row < 3; row += 1) if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
      if (Math.abs(a[pivot][col]) < 1e-9) return null;
      if (pivot !== col) {
        const tmpRow = a[col];
        a[col] = a[pivot];
        a[pivot] = tmpRow;
        const tmpB = b[col];
        b[col] = b[pivot];
        b[pivot] = tmpB;
      }
      for (let row = col + 1; row < 3; row += 1) {
        const factor = a[row][col] / a[col][col];
        for (let k = col; k < 3; k += 1) a[row][k] -= factor * a[col][k];
        b[row] -= factor * b[col];
      }
    }
    const out: [number, number, number] = [0, 0, 0];
    for (let row = 2; row >= 0; row -= 1) {
      let acc = b[row];
      for (let k = row + 1; k < 3; k += 1) acc -= a[row][k] * out[k];
      out[row] = acc / a[row][row];
    }
    return out;
  };
  const rowX = solve3(sxu, sxv, sx);
  const rowY = solve3(syu, syv, sy);
  if (!rowX || !rowY) return null;
  return { a11: rowX[0], a12: rowX[1], bx: rowX[2], a21: rowY[0], a22: rowY[1], by: rowY[2] };
}

function affineApply(A: Affine, u: number, v: number): Point {
  return { x: A.a11 * u + A.a12 * v + A.bx, y: A.a21 * u + A.a22 * v + A.by };
}

// Scale-free tile clustering: every blob proposes a scale; a candidate is a
// tight cluster of 8 similar-area blobs (the pattern spans ~4.2 cells, so
// members must sit within ~9 estimated cell-widths of the seed). Returns up
// to `maxClusters` distinct hypotheses, larger scales first (a closer flag
// beats background texture).
function findTileClusters(blobs: Blob[], maxClusters = 6): Blob[][] {
  const seeds = [...blobs].sort((a, b) => b.area - a.area);
  const clusters: Blob[][] = [];
  const signatures = new Set<string>();
  for (const seed of seeds) {
    const cellEst = Math.sqrt(seed.area);
    const members = blobs.filter(
      (b) =>
        b.area >= seed.area * 0.45 &&
        b.area <= seed.area * 2.4 &&
        Math.hypot(b.x - seed.x, b.y - seed.y) <= cellEst * 9,
    );
    if (members.length < 8) continue;
    let chosen = members;
    if (chosen.length > 8) {
      const areas = chosen.map((b) => b.area).sort((x, y) => x - y);
      const median = areas[areas.length >> 1];
      chosen = [...chosen]
        .sort((a, b) => Math.abs(Math.log(a.area / median)) - Math.abs(Math.log(b.area / median)))
        .slice(0, 10);
      let cx = 0;
      let cy = 0;
      for (const b of chosen) {
        cx += b.x;
        cy += b.y;
      }
      cx /= chosen.length;
      cy /= chosen.length;
      chosen = chosen.sort((a, b) => Math.hypot(a.x - cx, a.y - cy) - Math.hypot(b.x - cx, b.y - cy)).slice(0, 8);
    }
    const signature = chosen
      .map((b) => `${Math.round(b.x)},${Math.round(b.y)}`)
      .sort()
      .join(";");
    if (signatures.has(signature)) continue;
    signatures.add(signature);
    clusters.push(chosen);
    if (clusters.length >= maxClusters) break;
  }
  return clusters;
}

// One cluster attempt: constellation solve → dot check → affine + gates.
type Attempt = {
  stage: FlagFailStage;
  tiles: Point[];
  quad: Point[] | null;
  quadScore: number | null;
  dots: Point[];
  dotErrCells: number | null;
  cellPx: number | null;
  residual: number | null;
  result: { orderedTiles: Point[]; dot: Point; affine: Affine; cellPx: number; residual: number } | null;
};

const STAGE_RANK: Record<FlagFailStage, number> = {
  "too-few-blobs": 0,
  "too-few-tiles": 1,
  "missing-color": 1,
  "missing-shape": 1,
  "no-collinear-quad": 2,
  "bad-color-layout": 2,
  "bad-shape-layout": 2,
  "bad-side-split": 3,
  "no-dot-candidates": 4,
  "affine-fail": 5,
  "tiny-cell": 6,
  "dot-mismatch": 7,
  "high-residual": 8,
  ok: 9,
};

function solveCluster(pts: Blob[], allBlobs: Blob[]): Attempt {
  const attempt: Attempt = {
    stage: "no-collinear-quad",
    tiles: pts.map((b) => ({ x: b.x, y: b.y })),
    quad: null,
    quadScore: null,
    dots: [],
    dotErrCells: null,
    cellPx: null,
    residual: null,
    result: null,
  };

  // Collinear diagonal among the 8 (projective-invariant).
  let bestQuad: number[] | null = null;
  let bestQuadScore = Infinity;
  for (let a = 0; a < 5; a += 1)
    for (let b = a + 1; b < 6; b += 1)
      for (let c = b + 1; c < 7; c += 1)
        for (let d = c + 1; d < 8; d += 1) {
          const quad = [a, b, c, d];
          let mx = 0;
          let my = 0;
          for (const i of quad) {
            mx += pts[i].x;
            my += pts[i].y;
          }
          mx /= 4;
          my /= 4;
          let sxx = 0;
          let sxy = 0;
          let syy = 0;
          for (const i of quad) {
            const dx = pts[i].x - mx;
            const dy = pts[i].y - my;
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
          }
          const tr = sxx + syy;
          const det = sxx * syy - sxy * sxy;
          const lambda = tr / 2 - Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
          const spread = tr - lambda;
          if (spread <= 1e-6) continue;
          const score = Math.sqrt(Math.max(0, lambda) / 4) / Math.sqrt(spread / 4);
          if (score < bestQuadScore) {
            bestQuadScore = score;
            bestQuad = quad;
          }
        }
  attempt.quadScore = bestQuad ? bestQuadScore : null;
  if (bestQuad) attempt.quad = bestQuad.map((i) => ({ x: pts[i].x, y: pts[i].y }));
  if (!bestQuad || bestQuadScore > 0.12) return attempt;

  const quadSet = new Set(bestQuad);
  const sideIdx = [0, 1, 2, 3, 4, 5, 6, 7].filter((i) => !quadSet.has(i));
  const p0 = pts[bestQuad[0]];
  const p3 = pts[bestQuad[bestQuad.length - 1]];
  let dirX = p3.x - p0.x;
  let dirY = p3.y - p0.y;
  const dirLen = Math.hypot(dirX, dirY) || 1;
  dirX /= dirLen;
  dirY /= dirLen;
  const along = (p: Point) => p.x * dirX + p.y * dirY;
  const across = (p: Point) => -(p.x * dirY) + p.y * dirX;
  const diagSorted = [...bestQuad].sort((a, b) => along(pts[a]) - along(pts[b]));
  const sideA = sideIdx.filter((i) => across(pts[i]) >= 0).sort((a, b) => along(pts[a]) - along(pts[b]));
  const sideB = sideIdx.filter((i) => across(pts[i]) < 0).sort((a, b) => along(pts[a]) - along(pts[b]));
  attempt.stage = "bad-side-split";
  if (sideA.length !== 2 || sideB.length !== 2) return attempt;

  // Dot candidates: small dark blobs near this cluster, sized well below its
  // tiles (scale from THIS cluster, not the whole frame).
  const areas = pts.map((b) => b.area).sort((x, y) => x - y);
  const medianArea = areas[areas.length >> 1];
  const cellEst = Math.sqrt(medianArea);
  let ccx = 0;
  let ccy = 0;
  for (const b of pts) {
    ccx += b.x;
    ccy += b.y;
  }
  ccx /= pts.length;
  ccy /= pts.length;
  const dots = allBlobs.filter(
    (b) =>
      b.area >= medianArea * 0.03 &&
      b.area <= medianArea * 0.4 &&
      Math.hypot(b.x - ccx, b.y - ccy) <= cellEst * 7,
  );
  attempt.dots = dots.map((b) => ({ x: b.x, y: b.y }));
  attempt.stage = "no-dot-candidates";
  if (dots.length === 0) return attempt;

  // Correspondence hypotheses (diagonal direction × side labeling); the dot
  // picks the right one.
  let best: { tiles: Point[]; affine: Affine; dot: Point; dotErr: number } | null = null;
  for (const flipDiag of [false, true])
    for (const swapSides of [false, true]) {
      const diag = flipDiag ? [...diagSorted].reverse() : diagSorted;
      const sp = swapSides ? sideB : sideA;
      const sn = swapSides ? sideA : sideB;
      const spOrdered = flipDiag ? [...sp].reverse() : sp;
      const snOrdered = flipDiag ? [...sn].reverse() : sn;
      const orderedIdx = [...diag, ...spOrdered, ...snOrdered];
      const tilePts = orderedIdx.map((i) => ({ x: pts[i].x, y: pts[i].y }));
      const affine = fitAffine(tilePts);
      if (!affine) continue;
      const predictedDot = affineApply(affine, DOT_TEMPLATE[0], DOT_TEMPLATE[1]);
      let nearest: Point | null = null;
      let nearestDist = Infinity;
      for (const dot of dots) {
        const dist = Math.hypot(dot.x - predictedDot.x, dot.y - predictedDot.y);
        if (dist < nearestDist) {
          nearestDist = dist;
          nearest = { x: dot.x, y: dot.y };
        }
      }
      if (!nearest) continue;
      if (!best || nearestDist < best.dotErr) best = { tiles: tilePts, affine, dot: nearest, dotErr: nearestDist };
    }
  attempt.stage = "affine-fail";
  if (!best) return attempt;

  const cellPx = (Math.hypot(best.affine.a11, best.affine.a21) + Math.hypot(best.affine.a12, best.affine.a22)) / 2;
  attempt.cellPx = cellPx;
  attempt.dotErrCells = cellPx > 0 ? best.dotErr / cellPx : null;
  attempt.stage = "tiny-cell";
  if (cellPx < 3) return attempt;
  attempt.stage = "dot-mismatch";
  if (best.dotErr > cellPx * 0.8) return attempt;

  let residSq = 0;
  for (let i = 0; i < 8; i += 1) {
    const projected = affineApply(best.affine, TEMPLATE[i][0], TEMPLATE[i][1]);
    residSq += (projected.x - best.tiles[i].x) ** 2 + (projected.y - best.tiles[i].y) ** 2;
  }
  const residual = Math.sqrt(residSq / 8) / cellPx;
  attempt.residual = residual;
  attempt.stage = "high-residual";
  if (residual > 0.35) return attempt;

  attempt.stage = "ok";
  attempt.tiles = best.tiles;
  attempt.result = { orderedTiles: best.tiles, dot: best.dot, affine: best.affine, cellPx, residual };
  return attempt;
}

// Detect the flag in a grayscale patch. `normPx` is the constant the features
// are normalized by (use the full processing frame width for the whole
// session so calibration and prediction agree). Offsets shift reported
// coordinates back into full-frame space when a ROI patch is passed.
export function detectFlag(
  gray: Uint8Array,
  width: number,
  height: number,
  normPx: number,
  offsetX = 0,
  offsetY = 0,
): FlagObservation | null {
  return analyzeFlag(gray, width, height, normPx, offsetX, offsetY).observation;
}

// ARCHIVED (checkerboard card): retired from the trainer UI in favor of the
// CMY color card and the red shapes card; kept functional for old prints.
// Same as detectFlag, but also reports every intermediate stage for the
// diagnostics overlay.
export function analyzeFlag(
  gray: Uint8Array,
  width: number,
  height: number,
  normPx: number,
  offsetX = 0,
  offsetY = 0,
): { observation: FlagObservation | null; debug: FlagDebug } {
  const threshold = otsuThreshold(gray);
  const blobs = darkBlobs(gray, width, height, threshold, 6);
  const shift = (p: Point): Point => ({ x: p.x + offsetX, y: p.y + offsetY });
  const debug: FlagDebug = {
    failStage: "too-few-blobs",
    threshold,
    blobCount: blobs.length,
    blobs: blobs
      .slice()
      .sort((a, b) => b.area - a.area)
      .slice(0, 80)
      .map((b) => ({ x: b.x + offsetX, y: b.y + offsetY, area: b.area })),
    clusterCount: 0,
    tileCandidates: [],
    quad: null,
    quadScore: null,
    dotCandidates: [],
    dotErrCells: null,
    cellPx: null,
    residual: null,
    colorCounts: null,
    colorGates: null,
  };
  if (blobs.length < 9) return { observation: null, debug };

  const clusters = findTileClusters(blobs);
  debug.clusterCount = clusters.length;
  if (clusters.length === 0) {
    debug.failStage = "too-few-tiles";
    return { observation: null, debug };
  }

  // Try every cluster; keep whichever attempt got furthest for diagnostics.
  let bestAttempt: Attempt | null = null;
  for (const cluster of clusters) {
    const attempt = solveCluster(cluster, blobs);
    if (!bestAttempt || STAGE_RANK[attempt.stage] > STAGE_RANK[bestAttempt.stage]) bestAttempt = attempt;
    if (attempt.stage === "ok") break;
  }
  if (bestAttempt) {
    debug.failStage = bestAttempt.stage;
    debug.tileCandidates = bestAttempt.tiles.map(shift);
    debug.quad = bestAttempt.quad ? bestAttempt.quad.map(shift) : null;
    debug.quadScore = bestAttempt.quadScore;
    debug.dotCandidates = bestAttempt.dots.map(shift);
    debug.dotErrCells = bestAttempt.dotErrCells;
    debug.cellPx = bestAttempt.cellPx;
    debug.residual = bestAttempt.residual;
  }
  if (!bestAttempt || !bestAttempt.result) return { observation: null, debug };

  const { orderedTiles, dot, affine, cellPx, residual } = bestAttempt.result;
  const observation: FlagObservation = {
    tiles: orderedTiles.map(shift),
    dot: { x: dot.x + offsetX, y: dot.y + offsetY },
    center: { x: affine.bx + offsetX, y: affine.by + offsetY },
    cellPx,
    residual,
    features: featuresFromAffine(affine, offsetX, offsetY, normPx),
  };
  return { observation, debug };
}

// ============================================================================
// COLOR-QUADRANT card (the AMS-printed variant, and the recommended one).
//
// The card face is four big quadrants — facing the card (the camera's view):
// GREEN top-left, RED top-right, BLUE bottom-right, and the bare WHITE
// quadrant with a blue dot bottom-left. Saturated
// R/G/B patches are rare in a room and the color IS the correspondence, so
// detection collapses to "find the biggest red, green, and blue blobs that
// form the right layout" — no constellation solving, no orientation
// ambiguity (an affine handles the camera's mirror-or-not for free), and the
// patches are ~2× larger than checker cells, so it works farther away.
// ============================================================================

export type FlagPatternMode = "checker" | "color" | "shape";

// Quadrant centers in CARD-LOCAL units (1 unit = one quadrant side, y up,
// x = the gun's right). FACING the card — as the camera does — left/right
// mirror, so the camera sees: GREEN top-left, RED top-right, BLUE
// bottom-right, DOT bottom-left. Handedness never matters to the math: the
// 3-point affine absorbs any mirror, and the dot check only relies on the
// mirror-proof identity dot = green + blue − red (the parallelogram's fourth
// corner, diagonal from red).
const COLOR_TEMPLATE = {
  red: [-0.5, 0.5] as const,
  green: [0.5, 0.5] as const,
  blue: [-0.5, -0.5] as const,
  dot: [0.5, -0.5] as const,
};

// Per-pixel classification: 1=red, 2=green, 3=blue, 0=none.
//
// The card sits at 45° to the camera, lit mostly by ambient light — far
// darker than a face-on card in bright light — so the gates can't demand
// bright pixels. Each level combines three tests per color: a low absolute
// floor (rejects near-black noise), a channel RATIO (scale-invariant chroma),
// and an absolute dominance MARGIN over the other channels (protects the
// ratio test where gamma compresses dark pixels). Looser levels admit more
// background strays, but the layout gates (similar-size triple + dot /
// clear fourth quadrant) do the real filtering.
export type ColorSensitivity = "strict" | "normal" | "forgiving";

const COLOR_GATES: Record<
  ColorSensitivity,
  {
    rFloor: number; rRatio: number; rMargin: number;
    gFloor: number; gRatioR: number; gRatioB: number; gMargin: number;
    bFloor: number; bRatioR: number; bRatioG: number; bMargin: number;
  }
> = {
  strict: {
    rFloor: 70, rRatio: 1.45, rMargin: 24,
    gFloor: 55, gRatioR: 1.25, gRatioB: 1.12, gMargin: 14,
    bFloor: 55, bRatioR: 1.3, bRatioG: 1.12, bMargin: 14,
  },
  normal: {
    rFloor: 42, rRatio: 1.28, rMargin: 16,
    gFloor: 36, gRatioR: 1.16, gRatioB: 1.08, gMargin: 10,
    bFloor: 36, bRatioR: 1.16, bRatioG: 1.05, bMargin: 8,
  },
  forgiving: {
    rFloor: 30, rRatio: 1.2, rMargin: 11,
    gFloor: 26, gRatioR: 1.1, gRatioB: 1.05, gMargin: 7,
    bFloor: 26, bRatioR: 1.1, bRatioG: 1.03, bMargin: 5,
  },
};

// Which filament set is printed on the card. Slots keep their template
// positions; only the classifier changes:
//   "rgb" — red in the red slot, green, blue (+ blue dot). Classic.
//   "cmy" — facing the card: CYAN top-left (the green slot), YELLOW
//     top-right (the red slot), MAGENTA bottom-right (the blue slot), and
//     the white quadrant bottom-left with a BLACK dot. CMY patches light
//     TWO subpixels each, so they're ~2× brighter than RGB in dim rooms.
export type ColorPalette = "rgb" | "cmy";

// CMY gates: dom = the two lit channels' MINIMUM, rival = the excluded
// channel. Requiring BOTH lit channels up is inherently more selective than
// a single-channel test. Yellow's margin runs 1.5× — warm light, wood, and
// off-white walls all live near yellow.
const CMY_GATES: Record<ColorSensitivity, { floor: number; ratio: number; margin: number }> = {
  strict: { floor: 90, ratio: 1.35, margin: 40 },
  normal: { floor: 68, ratio: 1.2, margin: 26 },
  forgiving: { floor: 52, ratio: 1.12, margin: 16 },
};

// ---- Adjustable pickup -------------------------------------------------------
// The UI's dual-handle sliders expose the gate table directly: per color slot,
// [floor, margin] — the lower handle is the brightness FLOOR the channel must
// clear, the upper handle is floor + MARGIN, the absolute amount it must beat
// the strongest rival channel by. The ratio gates are derived from the margin
// with the same proportions the presets use, so a preset and its extracted
// tuning classify identically.
export type ColorGateTuning = {
  red: [number, number]; // [floor, margin] — cyan slot under the CMY palette
  green: [number, number]; // yellow slot under CMY
  blue: [number, number]; // magenta slot under CMY
};
export type ColorPickup = ColorSensitivity | ColorGateTuning;

// Extract a preset's floors/margins — seeds the UI sliders.
// Palette defaults are CMY everywhere: the app calibrates and trains on CMY
// cards only — RGB support remains in the classifier for older prints, but a
// caller must ask for it explicitly.
export function tuningFromPreset(s: ColorSensitivity, palette: ColorPalette = "cmy"): ColorGateTuning {
  if (palette === "cmy") {
    const g = CMY_GATES[s];
    // Yellow (red slot) runs a 1.5× margin — warm rooms live near yellow.
    return { red: [g.floor, Math.round(g.margin * 1.5)], green: [g.floor, g.margin], blue: [g.floor, g.margin] };
  }
  const g = COLOR_GATES[s];
  return { red: [g.rFloor, g.rMargin], green: [g.gFloor, g.gMargin], blue: [g.bFloor, g.bMargin] };
}

// Rebuild a full RGB gate set from a tuning: ratios follow the margin with
// the preset table's proportions (e.g. normal red margin 16 ↔ ratio 1.28).
function gatesFromTuning(t: ColorGateTuning) {
  return {
    rFloor: t.red[0], rRatio: 1 + t.red[1] / 56, rMargin: t.red[1],
    gFloor: t.green[0], gRatioR: 1 + t.green[1] / 62, gRatioB: 1 + t.green[1] / 130, gMargin: t.green[1],
    bFloor: t.blue[0], bRatioR: 1 + t.blue[1] / 50, bRatioG: 1 + t.blue[1] / 160, bMargin: t.blue[1],
  };
}

export function classifyRGB(
  r: number,
  g: number,
  b: number,
  sensitivity: ColorPickup = "forgiving",
  palette: ColorPalette = "cmy",
): 0 | 1 | 2 | 3 {
  if (palette === "cmy") {
    // Per-slot gates. A preset expands to one shared gate (yellow's margin
    // 1.5× — warm rooms live near yellow); a custom tuning drives each slot
    // directly, ratios derived with the CMY presets' proportion
    // (normal: margin 26 ↔ ratio 1.2).
    const slot = (pair: [number, number]) => ({ floor: pair[0], ratio: 1 + pair[1] / 128, margin: pair[1] });
    const preset = typeof sensitivity === "string" ? CMY_GATES[sensitivity] : null;
    const yeGate = preset
      ? { floor: preset.floor, ratio: preset.ratio, margin: preset.margin * 1.5 }
      : slot((sensitivity as ColorGateTuning).red);
    const cyGate = preset ?? slot((sensitivity as ColorGateTuning).green);
    const maGate = preset ?? slot((sensitivity as ColorGateTuning).blue);
    const cy = Math.min(g, b); // cyan: green+blue up, red down
    const ma = Math.min(r, b); // magenta: red+blue up, green down
    const ye = Math.min(r, g); // yellow: red+green up, blue down
    const cyOk = cy >= cyGate.floor && cy >= r * cyGate.ratio && cy - r >= cyGate.margin;
    const maOk = ma >= maGate.floor && ma >= g * maGate.ratio && ma - g >= maGate.margin;
    const yeOk = ye >= yeGate.floor && ye >= b * yeGate.ratio && ye - b >= yeGate.margin;
    // Best margin wins if several pass (they're near-exclusive in practice).
    // Facing the card: yellow top-right (red slot), cyan top-left (green
    // slot), magenta bottom-right (blue slot; the dot is BLACK).
    const scores: [number, 0 | 1 | 2 | 3][] = [
      [yeOk ? ye - b : -1, 1], // yellow lives in the red slot
      [cyOk ? cy - r : -1, 2], // cyan in the green slot
      [maOk ? ma - g : -1, 3], // magenta in the blue slot
    ];
    scores.sort((p, q) => q[0] - p[0]);
    return scores[0][0] >= 0 ? scores[0][1] : 0;
  }
  const gate = typeof sensitivity === "string" ? COLOR_GATES[sensitivity] : gatesFromTuning(sensitivity);
  if (r >= gate.rFloor && r >= g * gate.rRatio && r >= b * gate.rRatio && r >= Math.max(g, b) + gate.rMargin) return 1;
  if (g >= gate.gFloor && g >= r * gate.gRatioR && g >= b * gate.gRatioB && g >= Math.max(r, b) + gate.gMargin) return 2;
  if (b >= gate.bFloor && b >= r * gate.bRatioR && b >= g * gate.bRatioG && b >= Math.max(r, g) + gate.bMargin) return 3;
  return 0;
}

// Classification margin (dominance strength) for a class under a palette —
// feeds centroid weighting and the purity ranking.
export function classMargin(r: number, g: number, b: number, cls: 1 | 2 | 3, palette: ColorPalette): number {
  if (palette === "cmy") {
    if (cls === 1) return Math.min(r, g) - b; // yellow (red slot)
    if (cls === 2) return Math.min(g, b) - r; // cyan (green slot)
    return Math.min(r, b) - g; // magenta (blue slot)
  }
  if (cls === 1) return r - Math.max(g, b);
  if (cls === 2) return g - Math.max(r, b);
  return b - Math.max(r, g);
}

// Connected components over a 0/1 mask (4-neighborhood).
// weights (optional): per-pixel classification confidence 1..255 — centroids
// become confidence-weighted (sub-pixel accuracy: strong core pixels count
// more than fringe pixels that barely passed the gate).
function maskBlobs(mask: Uint8Array, width: number, height: number, minArea: number, weights?: Uint8Array): Blob[] {
  const size = width * height;
  const visited = new Uint8Array(size);
  const stack = new Int32Array(size);
  const blobs: Blob[] = [];
  for (let start = 0; start < size; start += 1) {
    if (visited[start] || !mask[start]) continue;
    let top = 0;
    stack[top++] = start;
    visited[start] = 1;
    let area = 0;
    let wsum = 0;
    let sx = 0;
    let sy = 0;
    let minX = width;
    let maxX = 0;
    let minY = height;
    let maxY = 0;
    // Diagonal extremes → true quad corners under any rotation.
    let minSum = Infinity;
    let maxSum = -Infinity;
    let minDif = Infinity;
    let maxDif = -Infinity;
    let cTL = { x: 0, y: 0 };
    let cBR = { x: 0, y: 0 };
    let cBL = { x: 0, y: 0 };
    let cTR = { x: 0, y: 0 };
    while (top > 0) {
      const idx = stack[--top];
      const x = idx % width;
      const y = (idx / width) | 0;
      area += 1;
      const wt = weights ? weights[idx] : 1;
      wsum += wt;
      sx += x * wt;
      sy += y * wt;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      const s = x + y;
      const d = x - y;
      if (s < minSum) {
        minSum = s;
        cTL = { x, y };
      }
      if (s > maxSum) {
        maxSum = s;
        cBR = { x, y };
      }
      if (d > maxDif) {
        maxDif = d;
        cTR = { x, y };
      }
      if (d < minDif) {
        minDif = d;
        cBL = { x, y };
      }
      if (x > 0 && !visited[idx - 1] && mask[idx - 1]) {
        visited[idx - 1] = 1;
        stack[top++] = idx - 1;
      }
      if (x < width - 1 && !visited[idx + 1] && mask[idx + 1]) {
        visited[idx + 1] = 1;
        stack[top++] = idx + 1;
      }
      if (y > 0 && !visited[idx - width] && mask[idx - width]) {
        visited[idx - width] = 1;
        stack[top++] = idx - width;
      }
      if (y < height - 1 && !visited[idx + width] && mask[idx + width]) {
        visited[idx + width] = 1;
        stack[top++] = idx + width;
      }
    }
    if (area >= minArea && wsum > 0) {
      // Shoelace area of the corner quad (TL→TR→BR→BL) → squareness.
      const quadArea =
        Math.abs(
          cTL.x * cTR.y -
            cTR.x * cTL.y +
            (cTR.x * cBR.y - cBR.x * cTR.y) +
            (cBR.x * cBL.y - cBL.x * cBR.y) +
            (cBL.x * cTL.y - cTL.x * cBL.y),
        ) / 2;
      blobs.push({
        x: sx / wsum + 0.5,
        y: sy / wsum + 0.5,
        area,
        w: maxX - minX + 1,
        h: maxY - minY + 1,
        x0: minX,
        y0: minY,
        corners: [cTL, cTR, cBR, cBL],
        // Too small to judge shape → neutral (0.7), never a penalty.
        squareness:
          area >= 60 ? Math.max(0, 1 - Math.abs(1 - area / Math.max(1, quadArea)) * 1.6) : 0.7,
      });
    }
  }
  return blobs;
}

// One step more forgiving — the seeded "swipe" check relaxes the color gate
// when hunting a quadrant that the strict classifier already failed to see.
// Custom tunings relax proportionally (the ~step between adjacent presets).
function relaxOne(sensitivity: ColorPickup): ColorPickup {
  if (typeof sensitivity !== "string") {
    const soften = (p: [number, number]): [number, number] => [Math.round(p[0] * 0.72), Math.round(p[1] * 0.68)];
    return { red: soften(sensitivity.red), green: soften(sensitivity.green), blue: soften(sensitivity.blue) };
  }
  return sensitivity === "strict" ? "normal" : "forgiving";
}

// Centroid + extent of pixels of ONE color class inside a window — the
// "swipe": a quick local sweep at a predicted quadrant position, far more
// tolerant than blob extraction because it needs no connectivity and can
// use a relaxed gate.
function localColorCentroid(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  cx: number,
  cy: number,
  half: number,
  cls: 1 | 2 | 3,
  sensitivity: ColorPickup,
  palette: ColorPalette,
): Blob | null {
  const x0 = Math.max(0, Math.round(cx - half));
  const x1 = Math.min(width - 1, Math.round(cx + half));
  const y0 = Math.max(0, Math.round(cy - half));
  const y1 = Math.min(height - 1, Math.round(cy + half));
  let n = 0;
  let sx = 0;
  let sy = 0;
  let minX = width;
  let maxX = 0;
  let minY = height;
  let maxY = 0;
  for (let y = y0; y <= y1; y += 1) {
    for (let x = x0; x <= x1; x += 1) {
      const o = (y * width + x) * 4;
      if (classifyRGB(rgba[o], rgba[o + 1], rgba[o + 2], sensitivity, palette) !== cls) continue;
      n += 1;
      sx += x;
      sy += y;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  if (n < 8) return null;
  return { x: sx / n + 0.5, y: sy / n + 0.5, area: n, w: maxX - minX + 1, h: maxY - minY + 1, x0: minX, y0: minY };
}

// Template positions per palette (card-local units). The CMY card swaps the
// class-3 patch (magenta) with the white/dot quadrant — facing the card:
// cyan top-left, yellow top-right, MAGENTA bottom-left, white + BLACK dot
// bottom-right.
function templateFor(palette: ColorPalette): {
  p1: readonly [number, number];
  p2: readonly [number, number];
  p3: readonly [number, number];
  dot: readonly [number, number];
} {
  return palette === "cmy"
    ? { p1: COLOR_TEMPLATE.red, p2: COLOR_TEMPLATE.green, p3: COLOR_TEMPLATE.dot, dot: COLOR_TEMPLATE.blue }
    : { p1: COLOR_TEMPLATE.red, p2: COLOR_TEMPLATE.green, p3: COLOR_TEMPLATE.blue, dot: COLOR_TEMPLATE.dot };
}

// Exact affine from any three template↔image correspondences:
// A = [P2−P1, P3−P1] · [T2−T1, T3−T1]⁻¹, b = P1 − A·T1.
function affineFrom3Pts(
  t1: readonly [number, number],
  t2: readonly [number, number],
  t3: readonly [number, number],
  p1: Point,
  p2: Point,
  p3: Point,
): Affine {
  const du1 = t2[0] - t1[0];
  const dv1 = t2[1] - t1[1];
  const du2 = t3[0] - t1[0];
  const dv2 = t3[1] - t1[1];
  const det = du1 * dv2 - du2 * dv1 || 1e-9;
  const dx1 = p2.x - p1.x;
  const dy1 = p2.y - p1.y;
  const dx2 = p3.x - p1.x;
  const dy2 = p3.y - p1.y;
  const a11 = (dx1 * dv2 - dx2 * dv1) / det;
  const a12 = (dx2 * du1 - dx1 * du2) / det;
  const a21 = (dy1 * dv2 - dy2 * dv1) / det;
  const a22 = (dy2 * du1 - dy1 * du2) / det;
  return {
    a11,
    a12,
    a21,
    a22,
    bx: p1.x - a11 * t1[0] - a12 * t1[1],
    by: p1.y - a21 * t1[0] - a22 * t1[1],
  };
}

// Exact affine from the three colored quadrant centers (6 equations, 6
// unknowns): card units → image px.
function affineFrom3(red: Point, green: Point, blue: Point): Affine {
  // Template: red (−.5,.5), green (.5,.5), blue (−.5,−.5).
  // u axis: green − red spans Δu = 1 at same v; v axis: red − blue spans Δv = 1.
  const a11 = green.x - red.x;
  const a21 = green.y - red.y;
  const a12 = red.x - blue.x;
  const a22 = red.y - blue.y;
  const bx = red.x + 0.5 * a11 - 0.5 * a12;
  const by = red.y + 0.5 * a21 - 0.5 * a22;
  return { a11, a12, a21, a22, bx, by };
}

// ---- Homography (perspective) pose --------------------------------------
// With 4+ correspondences we can fit a full 8-DOF homography instead of a
// 6-DOF affine. The two perspective terms (h31, h32) become features 7 and 8
// of the aim vector — exactly what changes as the gun tilts toward the
// camera, which the affine can't express. Affine-only frames use zeros for
// those terms (an affine IS a homography with h31 = h32 = 0), so the feature
// space stays consistent across paths.
export const FLAG_FEATURE_COUNT = 9;

function gaussSolve(m: number[][], b: number[]): number[] | null {
  const n = b.length;
  const a = m.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    if (Math.abs(a[pivot][col]) < 1e-12) return null;
    if (pivot !== col) {
      const tmp = a[col];
      a[col] = a[pivot];
      a[pivot] = tmp;
    }
    for (let row = col + 1; row < n; row += 1) {
      const f = a[row][col] / a[col][col];
      for (let k = col; k <= n; k += 1) a[row][k] -= f * a[col][k];
    }
  }
  const out = new Array<number>(n).fill(0);
  for (let row = n - 1; row >= 0; row -= 1) {
    let acc = a[row][n];
    for (let k = row + 1; k < n; k += 1) acc -= a[row][k] * out[k];
    out[row] = acc / a[row][row];
  }
  return out;
}

// Least-squares homography (card units → image px), h33 = 1. Image points
// are normalized (centroid + scale) for conditioning, then denormalized.
type Homography = [number, number, number, number, number, number, number, number]; // h11..h32

function solveHomographyPairs(cardPts: number[][], imagePts: Point[]): Homography | null {
  const n = imagePts.length;
  if (n < 4) return null;
  const mx = imagePts.reduce((s, p) => s + p.x, 0) / n;
  const my = imagePts.reduce((s, p) => s + p.y, 0) / n;
  const scale = Math.max(1e-6, imagePts.reduce((s, p) => s + Math.hypot(p.x - mx, p.y - my), 0) / n);
  // Normal equations AᵀA h = Aᵀb over the 2n DLT rows.
  const ata = Array.from({ length: 8 }, () => new Array<number>(8).fill(0));
  const atb = new Array<number>(8).fill(0);
  const addRow = (row: number[], rhs: number) => {
    for (let i = 0; i < 8; i += 1) {
      atb[i] += row[i] * rhs;
      for (let j = i; j < 8; j += 1) ata[i][j] += row[i] * row[j];
    }
  };
  for (let k = 0; k < n; k += 1) {
    const u = cardPts[k][0];
    const v = cardPts[k][1];
    const x = (imagePts[k].x - mx) / scale;
    const y = (imagePts[k].y - my) / scale;
    addRow([u, v, 1, 0, 0, 0, -x * u, -x * v], x);
    addRow([0, 0, 0, u, v, 1, -y * u, -y * v], y);
  }
  for (let i = 0; i < 8; i += 1) for (let j = 0; j < i; j += 1) ata[i][j] = ata[j][i];
  const h = gaussSolve(ata, atb);
  if (!h) return null;
  // Physical sanity: our card never tilts hard enough for perspective terms
  // this large — a fit like that is a noise artifact, not a pose.
  if (Math.abs(h[6]) + Math.abs(h[7]) > 0.8) return null;
  // Denormalize: H = T⁻¹·H' where T normalizes image coords.
  return [
    scale * h[0] + mx * h[6],
    scale * h[1] + mx * h[7],
    scale * h[2] + mx,
    scale * h[3] + my * h[6],
    scale * h[4] + my * h[7],
    scale * h[5] + my,
    h[6],
    h[7],
  ];
}

// Reprojection error of one known card point under an affine pose.
function reproErrAffine(A: Affine, t: number[], p: Point): number {
  const x = A.a11 * t[0] + A.a12 * t[1] + A.bx;
  const y = A.a21 * t[0] + A.a22 * t[1] + A.by;
  return Math.hypot(p.x - x, p.y - y);
}

function reproErrH(h: Homography, t: number[], p: Point): number {
  const w = h[6] * t[0] + h[7] * t[1] + 1;
  const x = (h[0] * t[0] + h[1] * t[1] + h[2]) / w;
  const y = (h[3] * t[0] + h[4] * t[1] + h[5]) / w;
  return Math.hypot(p.x - x, p.y - y);
}

// Robust pose from known correspondences: fit, REJECT any point whose
// reprojection error exceeds 2.5× the median (a glare-shifted centroid or
// half-occluded anchor would otherwise drag the whole pose), refit on the
// survivors, and only accept the homography if it genuinely beats the
// affine's residual — the known geometry is the referee.
function robustPose(
  cardPts: number[][],
  imgPts: Point[],
): { affine: Affine | null; h: Homography | null; rmsPx: number | null } {
  let useCard = cardPts;
  let usePts = imgPts;
  if (usePts.length < 3) return { affine: null, h: null, rmsPx: null };
  let A = fitAffinePairs(useCard, usePts);
  if (!A) return { affine: null, h: null, rmsPx: null };
  if (usePts.length >= 5) {
    const errs = usePts.map((p, i) => reproErrAffine(A as Affine, useCard[i], p));
    const sorted = [...errs].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const keep = errs.map((e) => e <= Math.max(1, median) * 2.5);
    const kept = keep.filter(Boolean).length;
    if (kept >= 4 && kept < usePts.length) {
      useCard = useCard.filter((_, i) => keep[i]);
      usePts = usePts.filter((_, i) => keep[i]);
      A = fitAffinePairs(useCard, usePts) ?? A;
    }
  }
  const aErrs = usePts.map((p, i) => reproErrAffine(A as Affine, useCard[i], p));
  const aRms = Math.sqrt(aErrs.reduce((s, e) => s + e * e, 0) / aErrs.length);
  let h: Homography | null = null;
  if (usePts.length >= 4) {
    const cand = solveHomographyPairs(useCard, usePts);
    if (cand) {
      const hErrs = usePts.map((p, i) => reproErrH(cand, useCard[i], p));
      const hRms = Math.sqrt(hErrs.reduce((s, e) => s + e * e, 0) / hErrs.length);
      // Exact 4-point fits have ~zero residual by construction; for 5+ the
      // homography must actually explain the data better than the affine.
      if (usePts.length === 4 || hRms <= aRms * 1.2 + 0.5) h = cand;
    }
  }
  return { affine: A, h, rmsPx: aRms };
}

// The 9-feature aim vector. Affine-only: perspective terms are zero.
function featuresFromAffine(A: Affine, offsetX: number, offsetY: number, normPx: number): number[] {
  return [
    1,
    (A.bx + offsetX) / normPx,
    (A.by + offsetY) / normPx,
    A.a11 / normPx,
    A.a12 / normPx,
    A.a21 / normPx,
    A.a22 / normPx,
    0,
    0,
  ];
}

function featuresFromHomography(h: Homography, offsetX: number, offsetY: number, normPx: number): number[] {
  // h13/h23 are the card-center image position (u=v=0), like bx/by; the
  // linear terms match the affine's; h31/h32 are dimensionless perspective.
  return [
    1,
    (h[2] + offsetX) / normPx,
    (h[5] + offsetY) / normPx,
    h[0] / normPx,
    h[1] / normPx,
    h[3] / normPx,
    h[4] / normPx,
    h[6],
    h[7],
  ];
}

// ---- CMY preliminary scan --------------------------------------------------
// Cheap full-frame pass run BEFORE acquisition. Every 2nd pixel is scored for
// how strongly cyan, yellow, and magenta it is (opponent measures — e.g.
// cyan-ness = (min(g,b) − r)/sum — which rank saturation, not just "passes a
// gate"), and the strengths pool into a coarse grid. The winner is the
// neighborhood whose WEAKEST channel is strongest: a wall or shirt can flood
// one channel across half the frame, but only the card has all three inks
// side by side. Acquisition then focuses on that region instead of taking
// the first plausible layout anywhere in frame.
export type CMYPrescan = {
  roi: { x: number; y: number; w: number; h: number };
  center: Point;
  score: number; // weakest-channel pooled strength of the winning window
};

const PRESCAN_CELL = 16; // px per grid cell
const PRESCAN_POOL_R = 3; // pooling radius, cells (window ≈ 112 px)
const PRESCAN_MIN_PIXEL = 0.05; // per-pixel strength floor to accumulate
const PRESCAN_MIN_SCORE = 0.8; // weakest channel must pool at least this

export function prescanCMY(rgba: Uint8ClampedArray, width: number, height: number): CMYPrescan | null {
  const gw = Math.max(1, Math.ceil(width / PRESCAN_CELL));
  const gh = Math.max(1, Math.ceil(height / PRESCAN_CELL));
  const accC = new Float32Array(gw * gh);
  const accY = new Float32Array(gw * gh);
  const accM = new Float32Array(gw * gh);
  for (let y = 0; y < height; y += 2) {
    const gy = (y / PRESCAN_CELL) | 0;
    for (let x = 0; x < width; x += 2) {
      const o = (y * width + x) * 4;
      const r = rgba[o];
      const g = rgba[o + 1];
      const b = rgba[o + 2];
      const sum = r + g + b;
      if (sum < 60 || sum > 740) continue; // too dark / blown out
      const gi = gy * gw + ((x / PRESCAN_CELL) | 0);
      const c = (Math.min(g, b) - r) / sum;
      const ye = (Math.min(r, g) - b) / sum;
      const m = (Math.min(r, b) - g) / sum;
      if (c > PRESCAN_MIN_PIXEL) accC[gi] += c;
      if (ye > PRESCAN_MIN_PIXEL) accY[gi] += ye;
      if (m > PRESCAN_MIN_PIXEL) accM[gi] += m;
    }
  }
  // Pool each channel over a (2R+1)² cell window and take the min across
  // channels — "all three inks near here" — keeping the best window.
  let best = 0;
  let bestGx = -1;
  let bestGy = -1;
  for (let gy = 0; gy < gh; gy += 1) {
    const y0 = Math.max(0, gy - PRESCAN_POOL_R);
    const y1 = Math.min(gh - 1, gy + PRESCAN_POOL_R);
    for (let gx = 0; gx < gw; gx += 1) {
      const x0 = Math.max(0, gx - PRESCAN_POOL_R);
      const x1 = Math.min(gw - 1, gx + PRESCAN_POOL_R);
      let pc = 0;
      let py = 0;
      let pm = 0;
      for (let yy = y0; yy <= y1; yy += 1)
        for (let xx = x0; xx <= x1; xx += 1) {
          const gi = yy * gw + xx;
          pc += accC[gi];
          py += accY[gi];
          pm += accM[gi];
        }
      const tri = Math.min(pc, py, pm);
      if (tri > best) {
        best = tri;
        bestGx = gx;
        bestGy = gy;
      }
    }
  }
  if (best < PRESCAN_MIN_SCORE || bestGx < 0) return null;
  const cx = (bestGx + 0.5) * PRESCAN_CELL;
  const cy = (bestGy + 0.5) * PRESCAN_CELL;
  // Focus region: generously larger than the pooling window so a close-up
  // card (bigger than the window) still fits with margin for the analyzer.
  const side = Math.round(PRESCAN_CELL * (2 * PRESCAN_POOL_R + 1) * 2.5);
  const x = Math.max(0, Math.min(width - Math.min(side, width), Math.round(cx - side / 2)));
  const y = Math.max(0, Math.min(height - Math.min(side, height), Math.round(cy - side / 2)));
  return {
    roi: { x, y, w: Math.min(side, width - x), h: Math.min(side, height - y) },
    center: { x: cx, y: cy },
    score: best,
  };
}

// ---- CYM beacon -------------------------------------------------------------
// Very cheap "where is the ink RIGHT NOW" pass (stride 8, well under a
// millisecond): per-channel histogram top-K strength scales, then the
// centroid of each channel's strong samples. If all three centroids sit
// near one another, that spot is almost certainly the card. Used to steer
// ---- Ink quadtree -----------------------------------------------------------
// Minimize the screen area the EXPENSIVE full-resolution analysis touches.
// A coarse ink-activity grid feeds an integral image; a quadtree then
// recursively PRUNES empty quadrants in O(1) each (never descending into
// screen with no colored ink), so we quickly isolate the small region(s)
// that actually contain C/Y/M. The costly per-pixel classification and
// flood-fill run only inside those boxes — not the whole frame.
export type InkRegion = { x: number; y: number; w: number; h: number; strength: number };

export function inkQuadtree(rgba: Uint8ClampedArray, width: number, height: number): InkRegion[] {
  const CELL = 10; // px per grid cell (coarse activity map)
  const gw = Math.max(1, Math.floor(width / CELL));
  const gh = Math.max(1, Math.floor(height / CELL));
  // Per-cell ink activity = max(cyan, yellow, magenta) opponent strength at
  // the cell's center sample. Cheap: one pixel per cell.
  const act = new Float32Array(gw * gh);
  for (let gy = 0; gy < gh; gy += 1) {
    const py = Math.min(height - 1, gy * CELL + (CELL >> 1));
    const base = py * width;
    for (let gx = 0; gx < gw; gx += 1) {
      const o = (base + Math.min(width - 1, gx * CELL + (CELL >> 1))) * 4;
      const r = rgba[o];
      const g = rgba[o + 1];
      const b = rgba[o + 2];
      const sum = r + g + b;
      if (sum < 60 || sum > 720) continue;
      const s = Math.max((Math.min(g, b) - r) / sum, (Math.min(r, g) - b) / sum, (Math.min(r, b) - g) / sum);
      if (s > 0.07) act[gy * gw + gx] = s;
    }
  }
  // Integral image of the grid → any quadrant's total activity in O(1).
  const IW = gw + 1;
  const II = new Float32Array(IW * (gh + 1));
  for (let y = 0; y < gh; y += 1) {
    let row = 0;
    const oCur = (y + 1) * IW;
    const oPrev = y * IW;
    for (let x = 0; x < gw; x += 1) {
      row += act[y * gw + x];
      II[oCur + x + 1] = II[oPrev + x + 1] + row;
    }
  }
  const sumOf = (x: number, y: number, w: number, h: number): number =>
    II[(y + h) * IW + x + w] - II[y * IW + x + w] - II[(y + h) * IW + x] + II[y * IW + x];

  // Quadtree: descend, pruning quadrants below the ink floor. Active leaves'
  // cells are marked; empty screen is never visited past its parent's O(1)
  // rejection.
  const mark = new Uint8Array(gw * gh);
  const LEAF = 8; // stop subdividing at this grid size
  const PRUNE = 0.8; // a quadrant totaling less ink than this is "empty"
  const rec = (x: number, y: number, w: number, h: number): void => {
    if (sumOf(x, y, w, h) < PRUNE) return; // ← the area-skipping prune
    if (w <= LEAF && h <= LEAF) {
      for (let yy = y; yy < y + h; yy += 1)
        for (let xx = x; xx < x + w; xx += 1) if (act[yy * gw + xx] > 0) mark[yy * gw + xx] = 1;
      return;
    }
    const hw = Math.max(1, w >> 1);
    const hh = Math.max(1, h >> 1);
    rec(x, y, hw, hh);
    if (w - hw > 0) rec(x + hw, y, w - hw, hh);
    if (h - hh > 0) rec(x, y + hh, hw, h - hh);
    if (w - hw > 0 && h - hh > 0) rec(x + hw, y + hh, w - hw, h - hh);
  };
  rec(0, 0, gw, gh);

  // Connected components (8-neighbour) over marked cells → disjoint ink
  // regions, so a distant distractor is its own box rather than inflating
  // the card's. Strongest first.
  const visited = new Uint8Array(gw * gh);
  const stack = new Int32Array(gw * gh);
  const regions: InkRegion[] = [];
  for (let start = 0; start < gw * gh; start += 1) {
    if (!mark[start] || visited[start]) continue;
    let top = 0;
    stack[top++] = start;
    visited[start] = 1;
    let minX = gw;
    let minY = gh;
    let maxX = 0;
    let maxY = 0;
    let strength = 0;
    while (top > 0) {
      const idx = stack[--top];
      const cx = idx % gw;
      const cy = (idx / gw) | 0;
      strength += act[idx];
      if (cx < minX) minX = cx;
      if (cx > maxX) maxX = cx;
      if (cy < minY) minY = cy;
      if (cy > maxY) maxY = cy;
      for (let dy = -1; dy <= 1; dy += 1)
        for (let dx = -1; dx <= 1; dx += 1) {
          const nx = cx + dx;
          const ny = cy + dy;
          if (nx < 0 || ny < 0 || nx >= gw || ny >= gh) continue;
          const ni = ny * gw + nx;
          if (mark[ni] && !visited[ni]) {
            visited[ni] = 1;
            stack[top++] = ni;
          }
        }
    }
    regions.push({
      x: minX * CELL,
      y: minY * CELL,
      w: (maxX - minX + 1) * CELL,
      h: (maxY - minY + 1) * CELL,
      strength,
    });
  }
  regions.sort((a, b) => b.strength - a.strength);
  return regions;
}

export function cymBeacon(rgba: Uint8ClampedArray, width: number, height: number): Point | null {
  const STRIDE = 8;
  const HB = 64;
  const HS = HB / 0.36;
  const hists = [new Uint32Array(HB), new Uint32Array(HB), new Uint32Array(HB)];
  const raw = [0, 0, 0];
  for (let y = 0; y < height; y += STRIDE) {
    for (let x = 0; x < width; x += STRIDE) {
      const o = (y * width + x) * 4;
      const r = rgba[o];
      const g = rgba[o + 1];
      const b = rgba[o + 2];
      const sum = r + g + b;
      if (sum < 40) continue;
      const v0 = (Math.min(r, g) - b) / sum;
      const v1 = (Math.min(g, b) - r) / sum;
      const v2 = (Math.min(r, b) - g) / sum;
      if (v0 > 0) {
        hists[0][Math.min(HB - 1, (v0 * HS) | 0)] += 1;
        if (v0 > raw[0]) raw[0] = v0;
      }
      if (v1 > 0) {
        hists[1][Math.min(HB - 1, (v1 * HS) | 0)] += 1;
        if (v1 > raw[1]) raw[1] = v1;
      }
      if (v2 > 0) {
        hists[2][Math.min(HB - 1, (v2 * HS) | 0)] += 1;
        if (v2 > raw[2]) raw[2] = v2;
      }
    }
  }
  const scaleOf = (h: Uint32Array, fallback: number): number => {
    let acc = 0;
    for (let bin = HB - 1; bin >= 0; bin -= 1) {
      acc += h[bin];
      if (acc >= 6) return (bin + 0.5) / HS; // ≈ a 20×20 patch at stride 8
    }
    return fallback;
  };
  const scales = [scaleOf(hists[0], raw[0]), scaleOf(hists[1], raw[1]), scaleOf(hists[2], raw[2])];
  if (Math.min(scales[0], scales[1], scales[2]) < 0.06) return null;
  // Second pass: centroid of each channel's strong samples.
  const cx = [0, 0, 0];
  const cy = [0, 0, 0];
  const cnt = [0, 0, 0];
  for (let y = 0; y < height; y += STRIDE) {
    for (let x = 0; x < width; x += STRIDE) {
      const o = (y * width + x) * 4;
      const r = rgba[o];
      const g = rgba[o + 1];
      const b = rgba[o + 2];
      const sum = r + g + b;
      if (sum < 40) continue;
      const vs = [(Math.min(r, g) - b) / sum, (Math.min(g, b) - r) / sum, (Math.min(r, b) - g) / sum];
      for (let k = 0; k < 3; k += 1) {
        if (vs[k] > scales[k] * 0.55) {
          cx[k] += x;
          cy[k] += y;
          cnt[k] += 1;
        }
      }
    }
  }
  if (cnt[0] < 2 || cnt[1] < 2 || cnt[2] < 2) return null;
  const pts = [0, 1, 2].map((k) => ({ x: cx[k] / cnt[k], y: cy[k] / cnt[k] }));
  // All three inks must sit near one another — that's what makes it the card.
  const near = Math.max(80, Math.min(width, height) * 0.25);
  for (let i = 0; i < 3; i += 1)
    for (let j = i + 1; j < 3; j += 1) {
      if (Math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y) > near) return null;
    }
  return { x: (pts[0].x + pts[1].x + pts[2].x) / 3, y: (pts[0].y + pts[1].y + pts[2].y) / 3 };
}

// ---- "Obvious card" acquisition ---------------------------------------------
// The MOST FORGIVING lock path, for the common reality: the three inks are
// wildly distinct on camera and nothing else in the scene looks like them.
// No absolute color gates at all. Each pixel is scored by the same opponent
// measures as the prescan (how cyan / yellow / magenta it is), and each
// channel keeps everything within striking distance of ITS OWN in-frame
// maximum — "the most yellow stuff here", whatever the lighting did to it.
// The top few blobs per channel then compete combinatorially for the card
// layout, held to the geometry that can't be relaxed: correct chirality
// (front face, any rotation), sibling patch sizes, sane anisotropy, and a
// preference for a dark dot at the predicted fourth corner.
const LOOSE_ABS_FLOOR = 0.045; // strength floor so a gray scene can't rank noise
const LOOSE_REL_FRAC = 0.4; // keep pixels ≥ this × the channel's frame max
const LOOSE_TOP_BLOBS = 8; // per-channel candidates entering the layout search

export function analyzeColorFlagLoose(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  normPx: number,
  offsetX = 0,
  offsetY = 0,
  fg: ForegroundGrid | null = null,
): { observation: FlagObservation | null; debug: FlagDebug } {
  const n = width * height;
  // Pass 1 (stride 2): each channel's strength SCALE via histogram — the
  // strength of roughly the 24th-strongest sample, not the raw maximum, so
  // a single hot/saturated pixel can't define "most yellow" for the whole
  // frame. Falls back to the true max when the card is so small/distant
  // that fewer than 24 samples carry ink. Slot order matches the
  // classifier: 0 = yellow ("red" slot), 1 = cyan ("green"), 2 = magenta
  // ("blue").
  const HBINS = 96;
  const HSCALE = HBINS / 0.36; // opponent strengths top out near 1/3
  const hists = [new Uint32Array(HBINS), new Uint32Array(HBINS), new Uint32Array(HBINS)];
  const rawMax = [0, 0, 0];
  for (let y = 0; y < height; y += 2) {
    for (let x = 0; x < width; x += 2) {
      // Motion veto: static cells don't get to define "most yellow here".
      if (fg && !fg.mask[(((offsetY + y) / fg.cell) | 0) * fg.gw + (((offsetX + x) / fg.cell) | 0)]) continue;
      const o = (y * width + x) * 4;
      const r = rgba[o];
      const g = rgba[o + 1];
      const b = rgba[o + 2];
      const sum = r + g + b;
      if (sum < 40) continue;
      const ye = (Math.min(r, g) - b) / sum;
      const c = (Math.min(g, b) - r) / sum;
      const m = (Math.min(r, b) - g) / sum;
      if (ye > 0) {
        hists[0][Math.min(HBINS - 1, (ye * HSCALE) | 0)] += 1;
        if (ye > rawMax[0]) rawMax[0] = ye;
      }
      if (c > 0) {
        hists[1][Math.min(HBINS - 1, (c * HSCALE) | 0)] += 1;
        if (c > rawMax[1]) rawMax[1] = c;
      }
      if (m > 0) {
        hists[2][Math.min(HBINS - 1, (m * HSCALE) | 0)] += 1;
        if (m > rawMax[2]) rawMax[2] = m;
      }
    }
  }
  const TOP_K = 96; // ≈ a 20×20 native-px patch at stride 2 (shooter is close)
  const scaleOf = (h: Uint32Array, fallback: number): number => {
    let acc = 0;
    for (let bin = HBINS - 1; bin >= 0; bin -= 1) {
      acc += h[bin];
      if (acc >= TOP_K) return (bin + 0.5) / HSCALE;
    }
    return fallback;
  };
  const maxs = [scaleOf(hists[0], rawMax[0]), scaleOf(hists[1], rawMax[1]), scaleOf(hists[2], rawMax[2])];
  const debug: FlagDebug = {
    failStage: "missing-color",
    threshold: 0,
    blobCount: 0,
    blobs: [],
    clusterCount: 0,
    tileCandidates: [],
    quad: null,
    quadScore: null,
    dotCandidates: [],
    dotErrCells: null,
    cellPx: 0,
    residual: null,
    colorCounts: { red: 0, green: 0, blue: 0 },
    colorGates: null,
  };
  // THREE-COLOR RULE: all three inks must register somewhere in frame.
  const present = [maxs[0] >= 0.06, maxs[1] >= 0.06, maxs[2] >= 0.06];
  if (!(present[0] && present[1] && present[2])) return { observation: null, debug };
  // Pass 2: mask everything near its channel's own maximum. Channels with
  // no real signal stay empty rather than ranking noise.
  const thr = maxs.map((mx) => Math.max(LOOSE_ABS_FLOOR, mx * LOOSE_REL_FRAC));
  const masks = [new Uint8Array(n), new Uint8Array(n), new Uint8Array(n)];
  for (let i = 0; i < n; i += 1) {
    if (fg) {
      const gx = ((offsetX + (i % width)) / fg.cell) | 0;
      const gy = ((offsetY + ((i / width) | 0)) / fg.cell) | 0;
      if (!fg.mask[gy * fg.gw + gx]) continue;
    }
    const o = i * 4;
    const r = rgba[o];
    const g = rgba[o + 1];
    const b = rgba[o + 2];
    const sum = r + g + b;
    if (sum < 40) continue;
    if (present[0] && (Math.min(r, g) - b) / sum > thr[0]) masks[0][i] = 1;
    if (present[1] && (Math.min(g, b) - r) / sum > thr[1]) masks[1][i] = 1;
    if (present[2] && (Math.min(r, b) - g) / sum > thr[2]) masks[2][i] = 1;
  }
  const minArea = Math.max(6, Math.round(n * 0.8e-5));
  const cands = masks.map((mask) =>
    maskBlobs(mask, width, height, minArea)
      .filter(isSquareEnough) // fabric/foliage look-alikes die HERE
      .sort((a, b) => b.area - a.area)
      .slice(0, LOOSE_TOP_BLOBS),
  );
  debug.colorCounts = { red: cands[0].length, green: cands[1].length, blue: cands[2].length };
  debug.blobCount = cands[0].length + cands[1].length + cands[2].length;
  if (cands[0].length === 0 || cands[1].length === 0 || cands[2].length === 0)
    return { observation: null, debug };
  debug.failStage = "bad-color-layout";

  const tpl = templateFor("cmy");
  const minDim = Math.min(width, height);
  const shift = (p: Point): Point => ({ x: p.x + offsetX, y: p.y + offsetY });
  // Dark fraction in a window — the black dot at the predicted 4th corner.
  const darkFracAt = (cx: number, cy: number, half: number): number => {
    const x0 = Math.max(0, Math.round(cx - half));
    const x1 = Math.min(width - 1, Math.round(cx + half));
    const y0 = Math.max(0, Math.round(cy - half));
    const y1 = Math.min(height - 1, Math.round(cy + half));
    let cnt = 0;
    let dark = 0;
    for (let y = y0; y <= y1; y += 2)
      for (let x = x0; x <= x1; x += 2) {
        const o = (y * width + x) * 4;
        const gray = (rgba[o] * 77 + rgba[o + 1] * 150 + rgba[o + 2] * 29) >> 8;
        cnt += 1;
        if (gray < 90) dark += 1;
      }
    return cnt > 0 ? dark / cnt : 0;
  };

  let best: { score: number; affine: Affine; picks: [Blob, Blob, Blob]; quadPx: number } | null = null;
  for (const pY of cands[0])
    for (const pC of cands[1])
      for (const pM of cands[2]) {
        const affine = affineFrom3Pts(tpl.p1, tpl.p2, tpl.p3, pY, pC, pM);
        // NO chirality gate (by request): if the three colors sit close
        // together in a square pattern, that IS the card — grab the area.
        // Sibling sizes, skew, squareness, and keystone agreement do the
        // discriminating; the arrangement's handedness is not policed.
        const s1 = Math.hypot(affine.a11, affine.a21);
        const s2 = Math.hypot(affine.a12, affine.a22);
        const quadPx = (s1 + s2) / 2;
        // Upper bound is generous — a close-up card legitimately spans most
        // of the frame.
        if (quadPx < 3.5 || quadPx > minDim * 0.85) continue;
        if (Math.max(s1, s2) > Math.min(s1, s2) * 3.0) continue; // extreme skew
        const aMax = Math.max(pY.area, pC.area, pM.area);
        const aMin = Math.max(1, Math.min(pY.area, pC.area, pM.area));
        if (aMax > aMin * 12) continue; // patches are siblings, not a mural + specks
        // Score: sibling-sized, cell-filling, bigger-is-better, dot-confirmed,
        // SQUARE (keystoned quads, not amorphous blobs), and keystone-agreed.
        const fill = (pY.area + pC.area + pM.area) / 3 / Math.max(1, quadPx * quadPx);
        const dotP = {
          x: affine.bx + affine.a11 * tpl.dot[0] + affine.a12 * tpl.dot[1],
          y: affine.by + affine.a21 * tpl.dot[0] + affine.a22 * tpl.dot[1],
        };
        const sqAvg = (sqOf(pY) + sqOf(pC) + sqOf(pM)) / 3;
        // WHITE-QUADRANT bonus: a bright neutral square where the fourth
        // quadrant belongs is strong confirmation; ink there is a penalty.
        // (Sampled 30% outward of its center so the black dot can't bias it.)
        const wpt = {
          x: affine.bx + affine.a11 * tpl.dot[0] * 1.3 + affine.a12 * tpl.dot[1] * 1.3,
          y: affine.by + affine.a21 * tpl.dot[0] * 1.3 + affine.a22 * tpl.dot[1] * 1.3,
        };
        const whiteBonus = whiteQuadScore(rgba, width, height, wpt.x, wpt.y, Math.max(2, quadPx * 0.25));
        let score =
          aMax / aMin - Math.min(1, fill) * 2 - Math.log2(Math.max(4, quadPx)) - sqAvg * 1.2 - whiteBonus * 1.5;
        // Shared keystone: three patches on ONE plane rotate together — the
        // top edges of their corner quads must point the same way.
        if (pY.corners && pC.corners && pM.corners && aMin >= 60) {
          const edgeAngle = (b: Blob): number => {
            const c = b.corners as [Point, Point, Point, Point];
            return Math.atan2(c[1].y - c[0].y, c[1].x - c[0].x);
          };
          const a1 = edgeAngle(pY);
          const a2 = edgeAngle(pC);
          const a3 = edgeAngle(pM);
          const spread = (u: number, v: number) => {
            const dAng = Math.abs(u - v) % Math.PI;
            return Math.min(dAng, Math.PI - dAng);
          };
          if (Math.max(spread(a1, a2), spread(a1, a3), spread(a2, a3)) < 0.22) score -= 0.8;
        }
        if (darkFracAt(dotP.x, dotP.y, Math.max(3, quadPx * 0.3)) > 0.12) score -= 2;
        if (!best || score < best.score) best = { score, affine, picks: [pY, pC, pM], quadPx };
      }
  // NOTE: the two-color PAIR LOCK was removed — THREE-COLOR RULE. Pairs
  // plus a corroborating dark spot hallucinated locks on cardless scenes;
  // all three inks in the correct sweep are the minimum evidence to lock.
  if (!best) return { observation: null, debug };

  const { affine, picks, quadPx } = best;
  debug.failStage = "ok";
  debug.cellPx = quadPx;
  debug.tileCandidates = picks.map((b) => shift({ x: b.x, y: b.y }));
  const dotLocal = {
    x: affine.bx + affine.a11 * tpl.dot[0] + affine.a12 * tpl.dot[1],
    y: affine.by + affine.a21 * tpl.dot[0] + affine.a22 * tpl.dot[1],
  };
  let features = featuresFromAffine(affine, offsetX, offsetY, normPx);
  {
    // Dark-centroid 4th correspondence → homography, same as the tracker.
    const half = Math.max(3, Math.round(quadPx * 0.3));
    const x0 = Math.max(0, Math.round(dotLocal.x - half));
    const x1 = Math.min(width - 1, Math.round(dotLocal.x + half));
    const y0 = Math.max(0, Math.round(dotLocal.y - half));
    const y1 = Math.min(height - 1, Math.round(dotLocal.y + half));
    let cnt = 0;
    let sx = 0;
    let sy = 0;
    for (let y = y0; y <= y1; y += 1)
      for (let x = x0; x <= x1; x += 1) {
        const o = (y * width + x) * 4;
        const gray = (rgba[o] * 77 + rgba[o + 1] * 150 + rgba[o + 2] * 29) >> 8;
        if (gray >= 85) continue;
        cnt += 1;
        sx += x;
        sy += y;
      }
    if (cnt >= 4 && cnt <= quadPx * quadPx * 0.25) {
      const h = solveHomographyPairs(
        [
          [tpl.p1[0], tpl.p1[1]],
          [tpl.p2[0], tpl.p2[1]],
          [tpl.p3[0], tpl.p3[1]],
          [tpl.dot[0], tpl.dot[1]],
        ],
        [
          { x: picks[0].x, y: picks[0].y },
          { x: picks[1].x, y: picks[1].y },
          { x: picks[2].x, y: picks[2].y },
          { x: sx / cnt + 0.5, y: sy / cnt + 0.5 },
        ],
      );
      if (h) features = featuresFromHomography(h, offsetX, offsetY, normPx);
    }
  }
  const observation: FlagObservation = {
    tiles: picks.map((b) => shift({ x: b.x, y: b.y })),
    dot: shift(dotLocal),
    center: { x: affine.bx + offsetX, y: affine.by + offsetY },
    cellPx: quadPx,
    // Rank-based fit has no reprojection residual to report — a modest
    // constant keeps the feature filter appropriately skeptical of these
    // frames without rejecting them.
    residual: 0.15,
    features,
    quadColors: {
      red: sampleMeanRGB(rgba, width, height, picks[0].x, picks[0].y),
      green: sampleMeanRGB(rgba, width, height, picks[1].x, picks[1].y),
      blue: sampleMeanRGB(rgba, width, height, picks[2].x, picks[2].y),
    },
  };
  return { observation, debug };
}

// ---- Motion prior -----------------------------------------------------------
// Optional foreground grid from the camera loop's background model (see
// backgroundModel.ts): mask[cell] = 1 where the scene CHANGED vs the learned
// background. The camera is stationary and the card is handheld, so the card
// is always foreground — while a poster, a colored bin, or a monitor bezel
// is not. When supplied, acquisition ignores pixels in static cells, which
// prunes candidate blobs before the layout search and lets the loose
// analyzer's relative scales rank only MOVING ink. Coordinates are
// full-frame: analyzers add their patch offset before the cell lookup.
export type ForegroundGrid = { mask: Uint8Array; gw: number; gh: number; cell: number };

// Detect the color-quadrant card in an RGBA patch.
export function analyzeColorFlag(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  normPx: number,
  offsetX = 0,
  offsetY = 0,
  sensitivity: ColorPickup = "forgiving",
  palette: ColorPalette = "cmy",
  // REFERENCE mode: sampled/locked patch colors. When present, pixels are
  // classified by chromaticity distance to these references (brightness-
  // normalized, so the card can brighten/darken by ±refBrightTol without
  // losing pickup) instead of the generic gates — assumes the shooter and
  // lighting haven't fundamentally changed since the reference was taken.
  refColors: TrackedColors | null = null,
  refBrightTol = 0.5,
  fg: ForegroundGrid | null = null,
  // User-adjustable COLOR WINDOW: acceptance radius (chroma units) around
  // the locked reference colors. Tighter = fewer look-alikes admitted.
  refWindow = 0.09,
  // Live illuminant gains from the white quadrant (see FlagTrackPrior).
  refGain: [number, number, number] | null = null,
  // Accumulated observed shades per ink (see ShadePools).
  refShades?: ShadePools,
): { observation: FlagObservation | null; debug: FlagDebug } {
  const n = width * height;
  const redMask = new Uint8Array(n);
  const greenMask = new Uint8Array(n);
  const blueMask = new Uint8Array(n);
  // Per-pixel confidence (dominant channel's margin over the runner-up) —
  // weights the blob centroids sub-pixel: saturated core pixels count more
  // than fringe pixels that barely passed the gate.
  const conf = new Uint8Array(n);
  // CMY cards carry a BLACK confirmation dot (maximal contrast on the white
  // quadrant, immune to every color shift) — collect a darkness mask for it.
  const darkMask = palette === "cmy" ? new Uint8Array(n) : null;
  const darkConf = palette === "cmy" ? new Uint8Array(n) : null;
  // Wide by design: the card's inks are far from anything else in a normal
  // scene, so err toward pickup — the layout/chirality gates downstream do
  // the discriminating, not the per-pixel color gate.
  const REF_THR = refWindow;
  // Illuminant correction first (white-quadrant gains), then the locus LUT
  // (2700–6500 K synthetic sweep, neutral competition baked in) + per-ink
  // loci for the confidence computation of classified pixels.
  const refLit =
    refColors && refGain
      ? {
          red: [refColors.red[0] * refGain[0], refColors.red[1] * refGain[1], refColors.red[2] * refGain[2]] as [number, number, number],
          green: [refColors.green[0] * refGain[0], refColors.green[1] * refGain[1], refColors.green[2] * refGain[2]] as [number, number, number],
          blue: [refColors.blue[0] * refGain[0], refColors.blue[1] * refGain[1], refColors.blue[2] * refGain[2]] as [number, number, number],
        }
      : refColors;
  const refLUT = refLit ? buildInkLUT(refLit, REF_THR, refShades) : null;
  const refLoci = refLit
    ? [
        buildChromaLocus(refLit.red).concat(refShades?.[0] ?? []),
        buildChromaLocus(refLit.green).concat(refShades?.[1] ?? []),
        buildChromaLocus(refLit.blue).concat(refShades?.[2] ?? []),
      ]
    : null;
  const refSums = refLit
    ? [refLit.red, refLit.green, refLit.blue].map((c) => Math.max(1, c[0] + c[1] + c[2]))
    : null;
  for (let i = 0; i < n; i += 1) {
    // Motion veto: pixels in cells that match the background can't be the
    // handheld card — skip them before any color test.
    if (fg) {
      const gx = ((offsetX + (i % width)) / fg.cell) | 0;
      const gy = ((offsetY + ((i / width) | 0)) / fg.cell) | 0;
      if (!fg.mask[gy * fg.gw + gx]) continue;
    }
    const o = i * 4;
    const r = rgba[o];
    const g = rgba[o + 1];
    const b = rgba[o + 2];
    if (darkMask && darkConf) {
      const gray = (r * 77 + g * 150 + b * 29) >> 8;
      if (gray < 85) {
        darkMask[i] = 1;
        darkConf[i] = 255 - gray;
      }
    }
    if (refLUT && refLoci && refSums) {
      const sum = r + g + b;
      if (sum < 30) continue;
      const cr = r / sum;
      const cg = g / sum;
      // Locus LUT decides the class; the per-slot brightness window then
      // vets it; confidence from the exact distance-to-locus.
      const best = refLUT[lutIndex(cr, cg)];
      if (best === 0) continue;
      const k = best - 1;
      if (sum < refSums[k] * (1 - refBrightTol) || sum > refSums[k] * (1 + refBrightTol)) continue;
      const bestD = locusDist2(cr, cg, refLoci[k]);
      if (best === 1) redMask[i] = 1;
      else if (best === 2) greenMask[i] = 1;
      else blueMask[i] = 1;
      conf[i] = Math.max(1, Math.round(255 * (1 - Math.min(1, Math.sqrt(bestD) / REF_THR))));
      continue;
    }
    const cls = classifyRGB(r, g, b, sensitivity, palette);
    if (cls === 0) continue;
    if (cls === 1) redMask[i] = 1;
    else if (cls === 2) greenMask[i] = 1;
    else blueMask[i] = 1;
    conf[i] = Math.min(255, Math.max(1, classMargin(r, g, b, cls, palette)));
  }
  // Per-blob color STRENGTH: how decisively the blob is its color, measured as
  // the dominant channel's margin over the runner-up, normalized by brightness
  // (sampled at the centroid). "Most red" beats "barely red" regardless of size.
  const strengthOf = (blob: Blob, cls: 1 | 2 | 3): number => {
    const [r, g, b] = sampleMeanRGB(rgba, width, height, blob.x, blob.y, 3);
    const sum = Math.max(1, r + g + b);
    return classMargin(r, g, b, cls, palette) / sum;
  };
  type Scored = Blob & { strength: number };
  const scoreAndPick = (blobsIn: Blob[], cls: 1 | 2 | 3, keep: number): Scored[] =>
    blobsIn
      .map((b) => ({ ...b, strength: strengthOf(b, cls) }))
      // Rank by purity × (log-ish) size, so vivid quadrants beat big murky
      // blobs, but a speck never beats a real quadrant.
      .sort((a, b) => b.strength * Math.sqrt(b.area) - a.strength * Math.sqrt(a.area))
      .slice(0, keep);

  // Quadrant candidates: the strongest few per color. The DOT gets its own
  // pool of ALL blue blobs down to tiny — it's the smallest blue thing on the
  // card and must never be crowded out of a top-N list by background strays.
  const allBlues = maskBlobs(blueMask, width, height, 5, conf).sort((a, b) => b.area - a.area);
  // WINNER-TAKE-ALL when reference colors are locked: rank each channel's
  // candidates by exact color closeness to the locked ink (illuminant-locus
  // distance) and keep ONLY the closest — a dynamic threshold set by the
  // best blob itself. Window-squeakers are discarded before the layout
  // search ever sees them.
  const topByColor = (list: Scored[], locus: ChromaLocus | null): Scored[] => {
    if (!locus || list.length <= 1) return list;
    let best: Scored | null = null;
    let bestD = Infinity;
    for (const b of list) {
      const [r, g, bl] = sampleMeanRGB(rgba, width, height, b.x, b.y, 3);
      const s = Math.max(1, r + g + bl);
      const d = locusDist2(r / s, g / s, locus);
      if (d < bestD) {
        bestD = d;
        best = b;
      }
    }
    return best ? [best] : list;
  };
  const reds = topByColor(
    scoreAndPick(maskBlobs(redMask, width, height, 12, conf).sort((a, b) => b.area - a.area).slice(0, 8), 1, 4),
    refLoci ? refLoci[0] : null,
  );
  const greens = topByColor(
    scoreAndPick(maskBlobs(greenMask, width, height, 12, conf).sort((a, b) => b.area - a.area).slice(0, 8), 2, 4),
    refLoci ? refLoci[1] : null,
  );
  const blues = topByColor(
    scoreAndPick(allBlues.filter((b) => b.area >= 12).slice(0, 8), 3, 4),
    refLoci ? refLoci[2] : null,
  );
  // Dot candidates: RGB card → small blue blobs; CMY card → small BLACK
  // blobs (the dot prints black there).
  const dotPool =
    darkMask && darkConf
      ? maskBlobs(darkMask, width, height, 4, darkConf)
          .sort((a, b) => b.area - a.area)
          .slice(0, 32)
      : allBlues.slice(0, 32);
  const shift = (p: Point): Point => ({ x: p.x + offsetX, y: p.y + offsetY });
  // Card-local template for this palette (CMY swaps magenta ↔ white/dot).
  const tpl = templateFor(palette);

  // ---- FAST SEEDED CHECK ("swipe") -----------------------------------------
  // The mount holds the card roughly upright in the image, so any ONE strong
  // quadrant predicts where its partners must be — facing the card: green one
  // quadrant LEFT of red, blue one BELOW red, layout rigid. When a color pool
  // is empty (washed out, below blob threshold), take the strongest seed we
  // do have, jump to each missing partner's predicted spot, and sweep a local
  // window for the expected color at a relaxed gate. A hit is verified for
  // anticipated size against the seed and injected as a candidate — then the
  // normal layout gates, chirality, and dot check still have to pass.
  if (reds.length === 0 || greens.length === 0 || blues.length === 0) {
    // Image positions in quadrant units, facing view. RGB: green TL, red TR,
    // dot BL, blue BR. CMY: cyan TL, yellow TR, magenta BL, white+dot BR.
    const POS: Record<1 | 2 | 3, [number, number]> =
      palette === "cmy" ? { 1: [1, 0], 2: [0, 0], 3: [0, 1] } : { 1: [1, 0], 2: [0, 0], 3: [1, 1] };
    const seeds: { cls: 1 | 2 | 3; blob: Scored }[] = [];
    if (reds[0]) seeds.push({ cls: 1, blob: reds[0] });
    if (greens[0]) seeds.push({ cls: 2, blob: greens[0] });
    if (blues[0]) seeds.push({ cls: 3, blob: blues[0] });
    seeds.sort((a, b) => b.blob.strength * Math.sqrt(b.blob.area) - a.blob.strength * Math.sqrt(a.blob.area));
    const seed = seeds[0];
    if (seed) {
      const quadPxEst = Math.sqrt(seed.blob.area); // quadrants are full squares
      const missing: (1 | 2 | 3)[] = [];
      if (reds.length === 0) missing.push(1);
      if (greens.length === 0) missing.push(2);
      if (blues.length === 0) missing.push(3);
      for (const cls of missing) {
        const px = seed.blob.x + (POS[cls][0] - POS[seed.cls][0]) * quadPxEst;
        const py = seed.blob.y + (POS[cls][1] - POS[seed.cls][1]) * quadPxEst;
        const found = localColorCentroid(rgba, width, height, px, py, quadPxEst * 0.7, cls, relaxOne(sensitivity), palette);
        // Anticipated shape: quadrant-sized relative to the seed, not a stray.
        if (found && found.area >= seed.blob.area * 0.25 && found.area <= seed.blob.area * 2.5) {
          const rescued: Scored = { ...found, strength: strengthOf(found, cls) };
          if (cls === 1) reds.push(rescued);
          else if (cls === 2) greens.push(rescued);
          else blues.push(rescued);
        }
      }
    }
  }

  const debug: FlagDebug = {
    failStage: "missing-color",
    threshold: 0,
    blobCount: reds.length + greens.length + allBlues.length,
    blobs: [...reds, ...greens, ...allBlues.slice(0, 16)].map((b) => ({
      x: b.x + offsetX,
      y: b.y + offsetY,
      area: b.area,
    })),
    clusterCount: 0,
    tileCandidates: [],
    quad: null,
    quadScore: null,
    dotCandidates: [],
    dotErrCells: null,
    cellPx: null,
    residual: null,
    colorCounts: { red: reds.length, green: greens.length, blue: allBlues.length },
    colorGates: null,
  };
  if (reds.length === 0 || greens.length === 0 || blues.length === 0) return { observation: null, debug };

  type Triple = {
    red: Scored;
    green: Scored;
    blue: Scored;
    affine: Affine;
    quadPx: number;
    det: number;
    score: number; // lower = better: area consistency + color purity + chirality
    chiralityOk: boolean;
    predictedDot: Point;
  };
  const gates = { triples: 0, sizeOk: 0, areaOk: 0, nearestDotErr: Infinity };
  let bestGeom: Triple | null = null;
  let bestDot: { triple: Triple; dot: Point; dotErr: number; dotBlob: Blob; boxHit: boolean } | null = null;

  debug.failStage = "bad-color-layout";
  for (const red of reds)
    for (const green of greens)
      for (const blue of blues) {
        gates.triples += 1;
        // HARD squareness gate: printed quads only — an amorphous fabric
        // or foliage blob can't enter a lock no matter how good its color.
        if (!isSquareEnough(red) || !isSquareEnough(green) || !isSquareEnough(blue)) continue;
        // Quadrants must be similar sizes. (Winner-take-all already keeps
        // murals and specks out, so this only rejects the absurd.)
        const areas = [red.area, green.area, blue.area];
        const maxA = Math.max(...areas);
        const minA = Math.min(...areas);
        if (maxA > minA * 4.5) continue;
        gates.sizeOk += 1;
        const affine = affineFrom3Pts(tpl.p1, tpl.p2, tpl.p3, red, green, blue);
        const quadPx = (Math.hypot(affine.a11, affine.a21) + Math.hypot(affine.a12, affine.a22)) / 2;
        if (quadPx < 4) continue;
        // Blob areas must be consistent with the affine's area scale.
        const detSigned = affine.a11 * affine.a22 - affine.a12 * affine.a21;
        const det = Math.abs(detSigned);
        if (det < 1) continue;
        // NO chirality gate (by request): the sweep direction is not
        // policed — three same-sized colors in a square pattern lock.
        // chiralityOk is still reported for the diagnostics readout.
        const chiralityOk = detSigned > 0;
        let areaScore = 0;
        let plausible = true;
        for (const area of areas) {
          const ratio = area / det;
          if (ratio < 0.3 || ratio > 2.2) plausible = false;
          areaScore += Math.abs(Math.log(Math.max(1e-3, ratio)));
        }
        if (!plausible) continue;
        gates.areaOk += 1;
        // Purity: "most red / most green / most blue" wins over marginal
        // classifications. Squareness: printed quadrilaterals beat
        // amorphous look-alikes (neutral 0.7 when too small to judge).
        const purityPenalty =
          Math.max(0, 0.22 - red.strength) * 3 +
          Math.max(0, 0.22 - green.strength) * 3 +
          Math.max(0, 0.22 - blue.strength) * 3;
        const squarePenalty = (1 - (sqOf(red) + sqOf(green) + sqOf(blue)) / 3) * 0.8;
        const predictedDot = {
          x: affine.a11 * tpl.dot[0] + affine.a12 * tpl.dot[1] + affine.bx,
          y: affine.a21 * tpl.dot[0] + affine.a22 * tpl.dot[1] + affine.by,
        };
        // WHITE-QUADRANT bonus (scoring only, never a gate): bright neutral
        // where the fourth quadrant belongs confirms the card; ink there
        // penalizes. Sampled 30% outward so the black dot can't bias it.
        const wq = {
          x: affine.a11 * tpl.dot[0] * 1.3 + affine.a12 * tpl.dot[1] * 1.3 + affine.bx,
          y: affine.a21 * tpl.dot[0] * 1.3 + affine.a22 * tpl.dot[1] * 1.3 + affine.by,
        };
        const whiteBonus = whiteQuadScore(rgba, width, height, wq.x, wq.y, Math.max(2, quadPx * 0.25));
        const score = areaScore + purityPenalty + squarePenalty - whiteBonus * 0.9;
        const triple: Triple = { red, green, blue, affine, quadPx, det, score, chiralityOk, predictedDot };
        if (!bestGeom || score < bestGeom.score) bestGeom = triple;
        // The dot: a SMALL blue blob near the white quadrant's center. Pool is
        // every blue blob, not just the quadrant candidates.
        for (const cand of dotPool) {
          if (cand === blue) continue;
          // The dot may render as a SECOND blue square (print bloom, close
          // range, blur merging it outward) — allow anything up to ~60% of a
          // quadrant's area, as long as it sits where the dot belongs. Two
          // blue squares in the right arrangement are the card, not a
          // conflict.
          if (cand.area > det * 0.6) continue;
          const err = Math.hypot(cand.x - predictedDot.x, cand.y - predictedDot.y);
          if (err / quadPx < gates.nearestDotErr) gates.nearestDotErr = err / quadPx;
          // Two ways to confirm: the blob's centroid lands near the predicted
          // dot, OR the predicted dot falls inside the blob's BOUNDING BOX.
          // The box test survives perspective (which drags the centroid off
          // the affine prediction) and a dot that prints/blurs into a small
          // square — the box still contains the predicted point.
          const dotSized = Math.max(cand.w, cand.h) <= quadPx * 0.85;
          const boxHit =
            dotSized &&
            predictedDot.x >= cand.x0 - 2 &&
            predictedDot.x <= cand.x0 + cand.w + 2 &&
            predictedDot.y >= cand.y0 - 2 &&
            predictedDot.y <= cand.y0 + cand.h + 2;
          if (err > quadPx * 0.6 && !boxHit) continue;
          const effErr = boxHit ? Math.min(err, quadPx * 0.25) : err;
          if (!bestDot || effErr + score * quadPx * 0.1 < bestDot.dotErr + bestDot.triple.score * bestDot.triple.quadPx * 0.1) {
            bestDot = { triple, dot: { x: cand.x, y: cand.y }, dotErr: effErr, dotBlob: cand, boxHit };
          }
        }
      }

  // Geometry-only fallback: when no dot confirms (washed out, below
  // resolution), accept the best triple under TIGHTER gates, provided the
  // white quadrant is free of large WRONG-colored blobs. Blue is exempt: a
  // blue blob there is the dot (possibly printed as a second blue square) —
  // evidence FOR the card, never against it.
  let chosen: {
    triple: Triple;
    dot: Point;
    dotErr: number;
    dotless: boolean;
    dotBlob?: Blob;
    boxHit?: boolean;
  } | null = bestDot ? { ...bestDot, dotless: false } : null;
  if (!chosen && bestGeom) {
    const t = bestGeom;
    // REFERENCE MODE ACCEPTS OUTRIGHT: when the match ran against the
    // human-committed / locked reference colors, identity was already
    // settled upstream — winner-take-all against illuminant-corrected loci
    // in a tight window, squareness, white-quadrant scoring. Demanding the
    // dot or a generic purity number on top was the single most
    // restrictive filter in the pipeline, vetoing frames that visibly
    // matched. The dot stays as a pure bonus (homography upgrade).
    if (refColors) {
      chosen = { triple: t, dot: t.predictedDot, dotErr: t.quadPx * 0.6, dotless: true };
    } else if (
      // GENERIC path (no reference): HIGH-CONFIDENCE colors need no dot —
      // three strong, size-consistent patches ARE the card.
      Math.min(t.red.strength, t.green.strength, t.blue.strength) >= 0.2 &&
      t.score <= 0.6 &&
      t.quadPx >= 6
    ) {
      chosen = { triple: t, dot: t.predictedDot, dotErr: t.quadPx * 0.6, dotless: true };
    } else {
      // Medium confidence: the original tighter dotless gates.
      const areas = [t.red.area, t.green.area, t.blue.area];
      const tightSize = Math.max(...areas) <= Math.min(...areas) * 2.2;
      const tightArea = areas.every((a) => a / t.det >= 0.45 && a / t.det <= 1.9);
      const fourthClear = ![...reds, ...greens].some(
        (b) =>
          b !== t.red &&
          b !== t.green &&
          b.area > t.det * 0.3 &&
          Math.hypot(b.x - t.predictedDot.x, b.y - t.predictedDot.y) < t.quadPx * 0.45,
      );
      // (No chirality requirement — square-pattern evidence suffices.)
      if (tightSize && tightArea && fourthClear) {
        chosen = { triple: t, dot: t.predictedDot, dotErr: t.quadPx * 0.6, dotless: true };
      }
    }
  }

  const reference = chosen?.triple ?? bestGeom;
  if (reference) {
    debug.tileCandidates = [reference.red, reference.green, reference.blue].map((b) => shift({ x: b.x, y: b.y }));
    debug.cellPx = reference.quadPx;
  }
  debug.colorGates = {
    triples: gates.triples,
    sizeOk: gates.sizeOk,
    areaOk: gates.areaOk,
    nearestDotErrCells: Number.isFinite(gates.nearestDotErr) ? gates.nearestDotErr : null,
    predictedDot: reference ? shift(reference.predictedDot) : null,
    dotless: chosen?.dotless ?? false,
    strengths: reference
      ? [reference.red.strength, reference.green.strength, reference.blue.strength]
      : null,
    chiralityOk: reference ? reference.chiralityOk : null,
    dotBoxHit: chosen?.boxHit ?? false,
    dotBox: chosen?.dotBlob
      ? { x: chosen.dotBlob.x0 + offsetX, y: chosen.dotBlob.y0 + offsetY, w: chosen.dotBlob.w, h: chosen.dotBlob.h }
      : null,
    anchors: null,
  };
  if (chosen) {
    debug.dotCandidates = [shift(chosen.dot)];
    debug.dotErrCells = chosen.dotErr / chosen.triple.quadPx;
    debug.residual = chosen.dotErr / chosen.triple.quadPx;
  }
  if (!chosen) return { observation: null, debug };

  debug.failStage = "ok";
  const A = chosen.triple.affine;
  // Perspective upgrade: with a real dot we have 4 correspondences — enough
  // for a full homography, whose h31/h32 terms carry the tilt information an
  // affine can't. Dotless locks fall back to affine (perspective = 0).
  let features = featuresFromAffine(A, offsetX, offsetY, normPx);
  if (!chosen.dotless) {
    const h = solveHomographyPairs(
      [
        [tpl.p1[0], tpl.p1[1]],
        [tpl.p2[0], tpl.p2[1]],
        [tpl.p3[0], tpl.p3[1]],
        [tpl.dot[0], tpl.dot[1]],
      ],
      [
        { x: chosen.triple.red.x, y: chosen.triple.red.y },
        { x: chosen.triple.green.x, y: chosen.triple.green.y },
        { x: chosen.triple.blue.x, y: chosen.triple.blue.y },
        chosen.dot,
      ],
    );
    if (h) features = featuresFromHomography(h, offsetX, offsetY, normPx);
  }
  const observation: FlagObservation = {
    tiles: [chosen.triple.red, chosen.triple.green, chosen.triple.blue].map((b) => shift({ x: b.x, y: b.y })),
    dot: shift(chosen.dot),
    center: { x: A.bx + offsetX, y: A.by + offsetY },
    cellPx: chosen.triple.quadPx,
    residual: chosen.dotErr / chosen.triple.quadPx,
    features,
    quadColors: {
      red: sampleMeanRGB(rgba, width, height, chosen.triple.red.x, chosen.triple.red.y),
      green: sampleMeanRGB(rgba, width, height, chosen.triple.green.x, chosen.triple.green.y),
      blue: sampleMeanRGB(rgba, width, height, chosen.triple.blue.x, chosen.triple.blue.y),
    },
  };
  return { observation, debug };
}

// Mean RGB in a small window around a point (patch coordinates).
function sampleMeanRGB(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  cx: number,
  cy: number,
  radius = 2,
): [number, number, number] {
  let r = 0;
  let g = 0;
  let b = 0;
  let count = 0;
  const x0 = Math.max(0, Math.round(cx) - radius);
  const x1 = Math.min(width - 1, Math.round(cx) + radius);
  const y0 = Math.max(0, Math.round(cy) - radius);
  const y1 = Math.min(height - 1, Math.round(cy) + radius);
  for (let y = y0; y <= y1; y += 1)
    for (let x = x0; x <= x1; x += 1) {
      const o = (y * width + x) * 4;
      r += rgba[o];
      g += rgba[o + 1];
      b += rgba[o + 2];
      count += 1;
    }
  if (count === 0) return [128, 128, 128];
  return [r / count, g / count, b / count];
}

// ============================================================================
// TRACK MODE — after acquisition locks, the card becomes the PRIME candidate.
//
// Instead of generic color gates and combinatorial layout solving, track mode
// classifies pixels against the card's LEARNED quadrant colors (chromaticity
// distance — very forgiving, because it matches "this filament under this
// lighting"), and assigns each quadrant to the blob nearest its PREDICTED
// position (previous affine + motion). Gates are loose; the sanity check is
// agreement with the prediction, which also prevents track hijacking by
// look-alike objects elsewhere in the frame.
// ============================================================================

export type FlagTrackPrior = {
  colors: TrackedColors;
  predictedCenter: Point; // full-frame processing coords
  cellPx: number;
  affine: [number, number, number, number]; // previous [a11, a12, a21, a22], px
  // Card palette — the CMY card's template swaps magenta ↔ the white/dot
  // quadrant, so the tracker must predict positions with the right layout.
  palette?: ColorPalette;
  // 0 = strict (fresh lock), 1 = fully relaxed (lock held ~a second; the
  // region is trusted, so gates open up to forgive lighting variance and
  // brief angle changes instead of dropping the track).
  relax?: number;
  // Consecutive missed frames on this track. While re-seeking, the pick /
  // center gates widen with the miss count — the dead-reckoned prediction
  // drifts, so a rigid acceptance radius would refuse the real card even
  // when it's back in view.
  misses?: number;
  // User-adjustable COLOR WINDOW (chroma units) around the learned colors.
  chromaWindow?: number;
  // Live illuminant gains (mean-normalized, from the white quadrant): the
  // committed identity colors are multiplied by these before classification,
  // so lighting is tracked as its own variable while identity stays frozen.
  illumGain?: [number, number, number];
  // Accumulated observed shades per ink (see ShadePools) — extra accepted
  // chroma points on top of the synthetic locus.
  shades?: ShadePools;
};

function chroma(c: readonly [number, number, number]): [number, number] {
  const sum = Math.max(1, c[0] + c[1] + c[2]);
  return [c[0] / sum, c[1] / sum];
}

// ---- Illuminant locus -------------------------------------------------------
// SYNTHETIC lighting sweep: instead of accepting a disk around the captured
// chroma, precompute where that ink SHOULD sit under any plausible room
// light (2700 K warm bulb → 6500 K daylight) and accept distance-to-locus.
// Von Kries adaptation applied in camera RGB (gamma-encoded — approximate,
// which the acceptance radius absorbs), with PARTIAL adaptation levels
// because webcam auto-white-balance cancels an unknown fraction of the
// illuminant before we ever see it.

// Planckian white point (Kang et al. approximation) → linear sRGB ratios,
// normalized to green.
function whitePointRGB(cctK: number): [number, number, number] {
  const T = cctK;
  const x =
    T <= 4000
      ? -0.2661239e9 / T ** 3 - 0.2343589e6 / T ** 2 + 0.8776956e3 / T + 0.17991
      : -3.0258469e9 / T ** 3 + 2.1070379e6 / T ** 2 + 0.2226347e3 / T + 0.24039;
  const y = -3 * x * x + 2.87 * x - 0.275;
  const X = x / y;
  const Z = (1 - x - y) / y;
  const r = Math.max(0.05, 3.2406 * X - 1.5372 - 0.4986 * Z);
  const g = Math.max(0.05, -0.9689 * X + 1.8758 + 0.0415 * Z);
  const b = Math.max(0.05, 0.0557 * X - 0.204 + 1.057 * Z);
  return [r / g, 1, b / g];
}

export type ChromaLocus = [number, number][];

const LOCUS_CCTS = [2700, 3400, 4300, 5500, 6500];
const LOCUS_ALPHAS = [0.5, 1]; // partial → full illuminant shift (AWB residual)

// The expected chroma positions of one captured ink across the CCT sweep.
export function buildChromaLocus(rgb: readonly [number, number, number]): ChromaLocus {
  const pts: ChromaLocus = [];
  const push = (r: number, g: number, b: number) => {
    const s = Math.max(1, r + g + b);
    pts.push([r / s, g / s]);
  };
  push(rgb[0], rgb[1], rgb[2]); // exactly as captured
  for (const T of LOCUS_CCTS) {
    const [wr, , wb] = whitePointRGB(T);
    for (const a of LOCUS_ALPHAS) {
      push(rgb[0] * (1 + (wr - 1) * a), rgb[1], rgb[2] * (1 + (wb - 1) * a));
    }
  }
  return pts;
}

export function locusDist2(cr: number, cg: number, locus: ChromaLocus): number {
  let bestD = Infinity;
  for (const [lr, lg] of locus) {
    const dr = cr - lr;
    const dg = cg - lg;
    const d = dr * dr + dg * dg;
    if (d < bestD) bestD = d;
  }
  return bestD;
}

// ACCUMULATED SHADES: chroma points of each ink actually OBSERVED during
// valid locks (the card sweeping through poses/lighting during the 9-dot
// calibration and training). Appended to the synthetic illuminant locus,
// so classification accepts every shade the ink has genuinely shown.
export type ShadePools = [[number, number][], [number, number][], [number, number][]]; // red, green, blue slots

// Per-frame classification LUT over quantized chroma space: 64×64 cells,
// each holding 0 (none) or 1..3 (ink index). Baked once per call (~270k
// flops), then the pixel loop is a single array lookup — the locus test
// costs nothing per pixel. Neutral competition is baked in.
const LUT_N = 64;
export function buildInkLUT(
  colors: { red: readonly [number, number, number]; green: readonly [number, number, number]; blue: readonly [number, number, number] },
  thr: number,
  shades?: ShadePools,
): Uint8Array {
  const loci = [
    buildChromaLocus(colors.red).concat(shades?.[0] ?? []),
    buildChromaLocus(colors.green).concat(shades?.[1] ?? []),
    buildChromaLocus(colors.blue).concat(shades?.[2] ?? []),
  ];
  const lut = new Uint8Array(LUT_N * LUT_N);
  const thr2 = thr * thr;
  for (let iy = 0; iy < LUT_N; iy += 1) {
    const cg = (iy + 0.5) / LUT_N;
    for (let ix = 0; ix < LUT_N; ix += 1) {
      const cr = (ix + 0.5) / LUT_N;
      const drN = cr - 1 / 3;
      const dgN = cg - 1 / 3;
      const dNeutral = drN * drN + dgN * dgN;
      let best = 0;
      let bestD = thr2;
      for (let k = 0; k < 3; k += 1) {
        const d = locusDist2(cr, cg, loci[k]);
        if (d < bestD && d < dNeutral) {
          bestD = d;
          best = k + 1;
        }
      }
      lut[iy * LUT_N + ix] = best;
    }
  }
  return lut;
}

// Chroma → LUT index (clamped).
export function lutIndex(cr: number, cg: number): number {
  const ix = Math.min(LUT_N - 1, (cr * LUT_N) | 0);
  const iy = Math.min(LUT_N - 1, (cg * LUT_N) | 0);
  return iy * LUT_N + ix;
}

export function trackColorFlag(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  normPx: number,
  offsetX: number,
  offsetY: number,
  prior: FlagTrackPrior,
): { observation: FlagObservation | null; debug: FlagDebug } {
  const n = width * height;
  const relax = Math.min(1, Math.max(0, prior.relax ?? 0));
  // Re-seek widening: each missed frame lets the prediction drift, so the
  // acceptance gates grow with the miss count (capped at 3×) to grab the
  // card back instead of demanding it reappear exactly on-prediction.
  const seek = 1 + Math.min(2, 0.2 * (prior.misses ?? 0));
  // Generous from the first locked frame — the seed was verified by full
  // acquisition, so the job here is to HOLD ON; a matured lock opens even
  // further so lighting shifts (a shadow crossing, exposure hunting) still
  // read as "the same filament".
  // Base color window from the user's slider (tight by default); maturity
  // opens it substantially — a held lock earns trust, and the continuous
  // color learning keeps the center honest while the window loosens.
  const chromaThr = (prior.chromaWindow ?? 0.09) * (1 + 0.8 * relax);
  const sumLo = 40 - 20 * relax; // dimmer pixels admitted once trusted
  const sumHi = 730 + 25 * relax; // and nearly-blown highlights too
  // ILLUMINANT-LOCUS classification (2700–6500 K synthetic sweep) with
  // neutral competition, baked into a per-call LUT — per pixel it's one
  // table lookup.
  // Illuminant gains from the white quadrant: expected ink colors follow
  // the LIGHT while the identity colors stay frozen.
  const ig = prior.illumGain ?? [1, 1, 1];
  const lit = (c: readonly [number, number, number]): [number, number, number] => [
    c[0] * ig[0],
    c[1] * ig[1],
    c[2] * ig[2],
  ];
  const litColors = { red: lit(prior.colors.red), green: lit(prior.colors.green), blue: lit(prior.colors.blue) };
  const lut = buildInkLUT(litColors, chromaThr, prior.shades);
  // Exact per-ink loci for the winner-take-all color ranking of blobs —
  // accumulated shades included.
  const loci = [
    buildChromaLocus(litColors.red).concat(prior.shades?.[0] ?? []),
    buildChromaLocus(litColors.green).concat(prior.shades?.[1] ?? []),
    buildChromaLocus(litColors.blue).concat(prior.shades?.[2] ?? []),
  ];
  const masks = [new Uint8Array(n), new Uint8Array(n), new Uint8Array(n)];
  for (let i = 0; i < n; i += 1) {
    const o = i * 4;
    const r = rgba[o];
    const g = rgba[o + 1];
    const b = rgba[o + 2];
    const sum = r + g + b;
    if (sum < sumLo || sum > sumHi) continue; // near-black noise / blown white
    const cls = lut[lutIndex(r / sum, g / sum)];
    if (cls > 0) masks[cls - 1][i] = 1;
  }

  // GRABBY once matured: a held lock relaxes GEOMETRY hard (smaller
  // fragments accepted, patches allowed further off-prediction) while the
  // COLOR requirement stays untouched — all three learned inks, always.
  const minArea = Math.max(6, prior.cellPx * prior.cellPx * (0.05 - 0.025 * relax));
  const shift = (p: Point): Point => ({ x: p.x + offsetX, y: p.y + offsetY });
  const localPredCenter = { x: prior.predictedCenter.x - offsetX, y: prior.predictedCenter.y - offsetY };
  const [a11, a12, a21, a22] = prior.affine;
  const predictQuad = (u: number, v: number): Point => ({
    x: localPredCenter.x + a11 * u + a12 * v,
    y: localPredCenter.y + a21 * u + a22 * v,
  });
  const tpl = templateFor(prior.palette ?? "cmy");
  const templates = [tpl.p1, tpl.p2, tpl.p3] as const;

  const picks: (Blob | null)[] = [null, null, null];
  const counts = [0, 0, 0];
  const pickRadius = prior.cellPx * (2.6 + 3.4 * relax) * seek; // near where it should be, wider while re-seeking
  for (let k = 0; k < 3; k += 1) {
    const blobs = maskBlobs(masks[k], width, height, minArea);
    counts[k] = blobs.length;
    const predicted = predictQuad(templates[k][0], templates[k][1]);
    let best: Blob | null = null;
    let bestScore = Infinity;
    for (const blob of blobs) {
      const dist = Math.hypot(blob.x - predicted.x, blob.y - predicted.y);
      if (dist > pickRadius) continue;
      // WINNER-TAKE-ALL by color: among in-radius candidates, the blob
      // CLOSEST in color to the locked ink IS the patch — the best blob
      // sets the dynamic threshold and the rest are discarded. Position
      // only breaks near-ties.
      const [sr, sg, sb] = sampleMeanRGB(rgba, width, height, blob.x, blob.y, 2);
      const ssum = Math.max(1, sr + sg + sb);
      const score = locusDist2(sr / ssum, sg / ssum, loci[k]) + (dist / Math.max(1, pickRadius)) * 1e-4;
      if (score < bestScore) {
        bestScore = score;
        best = blob;
      }
    }
    picks[k] = best;
  }

  const debug: FlagDebug = {
    failStage: "missing-color",
    threshold: 0,
    blobCount: counts[0] + counts[1] + counts[2],
    blobs: [],
    clusterCount: 0,
    tileCandidates: picks.filter((p): p is Blob => p !== null).map((b) => shift({ x: b.x, y: b.y })),
    quad: null,
    quadScore: null,
    dotCandidates: [],
    dotErrCells: null,
    cellPx: prior.cellPx,
    residual: null,
    colorCounts: { red: counts[0], green: counts[1], blue: counts[2] },
    colorGates: null,
  };
  // THREE-COLOR RULE with a RIGID-COHERENCE exception for matured locks:
  // two patches that demonstrably moved TOGETHER — same translation (i.e.
  // same velocity), same rotation, same scale as the established lock, and
  // the right colors (the picks are winner-take-all color matches) — are
  // the one solid card with a patch momentarily blanked. The third patch is
  // synthesized from the rigid motion. The caller caps consecutive soft
  // frames so pure 2-patch coasting can't dig in.
  const synthetic = [false, false, false];
  const foundCount = (picks[0] ? 1 : 0) + (picks[1] ? 1 : 0) + (picks[2] ? 1 : 0);
  let softPatches = 0;
  if (foundCount < 3) {
    let coherent = false;
    if (foundCount === 2 && relax >= 1) {
      const foundIdx: number[] = [];
      for (let k = 0; k < 3; k += 1) if (picks[k]) foundIdx.push(k);
      const [i, j] = foundIdx;
      const A = picks[i] as Blob;
      const B = picks[j] as Blob;
      const pi = predictQuad(templates[i][0], templates[i][1]);
      const pj = predictQuad(templates[j][0], templates[j][1]);
      // Same-velocity test: the two patches' displacements from prediction
      // must agree — rigid translation, not two unrelated blobs.
      const sameShift = Math.hypot(A.x - pi.x - (B.x - pj.x), A.y - pi.y - (B.y - pj.y)) <= prior.cellPx * 0.6;
      if (sameShift) {
        // Similarity implied by the two points → angle/scale coherence with
        // the established lock.
        const dtx = templates[j][0] - templates[i][0];
        const dty = templates[j][1] - templates[i][1];
        const den = dtx * dtx + dty * dty;
        const dpx = B.x - A.x;
        const dpy = B.y - A.y;
        const sa = (dpx * dtx + dpy * dty) / den;
        const sb = (dpy * dtx - dpx * dty) / den;
        const newScale = Math.hypot(sa, sb);
        const priorAngle = Math.atan2(a21, a11);
        const newAngle = Math.atan2(sb, sa);
        let dAng = Math.abs(newAngle - priorAngle) % (Math.PI * 2);
        if (dAng > Math.PI) dAng = Math.PI * 2 - dAng;
        const coScale = newScale / Math.max(1e-3, prior.cellPx);
        if (dAng < 0.21 && coScale > 0.75 && coScale < 1.3) {
          const k = 3 - i - j;
          const bx2 = A.x - (sa * templates[i][0] - sb * templates[i][1]);
          const by2 = A.y - (sb * templates[i][0] + sa * templates[i][1]);
          picks[k] = {
            x: bx2 + sa * templates[k][0] - sb * templates[k][1],
            y: by2 + sb * templates[k][0] + sa * templates[k][1],
            area: 1,
            w: 1,
            h: 1,
            x0: 0,
            y0: 0,
          };
          synthetic[k] = true;
          softPatches = 1;
          coherent = true;
        }
      }
    }
    if (!coherent) return { observation: null, debug };
  }
  if (!picks[0] || !picks[1] || !picks[2]) return { observation: null, debug };

  const affine = affineFrom3Pts(tpl.p1, tpl.p2, tpl.p3, picks[0], picks[1], picks[2]);
  const quadPx = (Math.hypot(affine.a11, affine.a21) + Math.hypot(affine.a12, affine.a22)) / 2;
  // Sanity vs the prediction: scale can't jump, center can't teleport. A
  // matured lock forgives more on the SHRINK side only — a brief cant of the
  // gun collapses the card's apparent scale (cosine), but nothing physical
  // grows it per-frame, and a symmetric relax lets scale runaway compound
  // (1.6×/frame snowballs to frame-sized "quadrants" in under a second).
  debug.failStage = "bad-color-layout";
  const scaleRatio = quadPx / Math.max(1e-3, prior.cellPx);
  if (scaleRatio < 0.5 - 0.35 * relax || scaleRatio > 1.5 + 0.3 * relax) return { observation: null, debug };
  const centerErr = Math.hypot(affine.bx - localPredCenter.x, affine.by - localPredCenter.y);
  if (centerErr > prior.cellPx * (2.6 + 3.4 * relax) * seek) return { observation: null, debug };
  // NO chirality gate (by request) — a mirrored-looking fit is accepted;
  // the scale/center gates above carry the sanity checking.

  debug.failStage = "ok";
  debug.cellPx = quadPx;
  debug.residual = centerErr / Math.max(1e-3, quadPx);
  const dotLocal = predictQuad(tpl.dot[0], tpl.dot[1]);
  // Perspective: hunt the real dot (a small blob near its predicted spot —
  // blue on the RGB card, BLACK on the CMY card) for a 4th correspondence →
  // homography features. Not required — affine features when not visible.
  let features = featuresFromAffine(affine, offsetX, offsetY, normPx);
  {
    const smallBlues = maskBlobs(masks[2], width, height, 4);
    let dotBlob: Blob | null = null;
    let bestD = quadPx * 0.5;
    for (const cand of smallBlues) {
      if (cand.area > quadPx * quadPx * 0.25) continue;
      const d = Math.hypot(cand.x - dotLocal.x, cand.y - dotLocal.y);
      if (d < bestD) {
        bestD = d;
        dotBlob = cand;
      }
    }
    if (!dotBlob) {
      // Darkness fallback — the CMY card's dot is BLACK (and an RGB blue
      // dot is dark enough to qualify too): centroid of dark pixels in a
      // small window at the predicted spot.
      const half = Math.max(3, Math.round(quadPx * 0.3));
      const x0 = Math.max(0, Math.round(dotLocal.x - half));
      const x1 = Math.min(width - 1, Math.round(dotLocal.x + half));
      const y0 = Math.max(0, Math.round(dotLocal.y - half));
      const y1 = Math.min(height - 1, Math.round(dotLocal.y + half));
      let cnt = 0;
      let sx = 0;
      let sy = 0;
      for (let y = y0; y <= y1; y += 1) {
        for (let x = x0; x <= x1; x += 1) {
          const o = (y * width + x) * 4;
          const gray = (rgba[o] * 77 + rgba[o + 1] * 150 + rgba[o + 2] * 29) >> 8;
          if (gray >= 85) continue;
          cnt += 1;
          sx += x;
          sy += y;
        }
      }
      if (cnt >= 4 && cnt <= quadPx * quadPx * 0.25) {
        dotBlob = { x: sx / cnt + 0.5, y: sy / cnt + 0.5, area: cnt, w: 1, h: 1, x0: 0, y0: 0 };
      }
    }
    if (dotBlob) {
      const h = solveHomographyPairs(
        [
          [tpl.p1[0], tpl.p1[1]],
          [tpl.p2[0], tpl.p2[1]],
          [tpl.p3[0], tpl.p3[1]],
          [tpl.dot[0], tpl.dot[1]],
        ],
        [
          { x: picks[0].x, y: picks[0].y },
          { x: picks[1].x, y: picks[1].y },
          { x: picks[2].x, y: picks[2].y },
          { x: dotBlob.x, y: dotBlob.y },
        ],
      );
      if (h) features = featuresFromHomography(h, offsetX, offsetY, normPx);
    }
  }
  // ILLUMINANT SAMPLE: the white quadrant is a built-in gray card. Sample
  // off-center (the black dot lives at the quadrant center — go 35% further
  // out along the quadrant diagonal), and reject implausible readings:
  // too dark (occluded), blown (specular glare), or so strongly colored it
  // was probably an ink mis-sample.
  let whiteRGB: [number, number, number] | null = null;
  {
    const wpt = predictQuad(tpl.dot[0] * 1.35, tpl.dot[1] * 1.35);
    const [wr, wg, wb] = sampleMeanRGB(rgba, width, height, wpt.x, wpt.y, 2);
    const wsum = wr + wg + wb;
    if (wsum > 150 && wsum < 740) {
      const wcr = wr / wsum;
      const wcg = wg / wsum;
      if (Math.hypot(wcr - 1 / 3, wcg - 1 / 3) < 0.15) whiteRGB = [wr, wg, wb];
    }
  }
  const observation: FlagObservation = {
    tiles: picks.map((b) => shift({ x: (b as Blob).x, y: (b as Blob).y })),
    dot: shift(dotLocal),
    center: { x: affine.bx + offsetX, y: affine.by + offsetY },
    cellPx: quadPx,
    // Soft (coherence-held) frames report a raised residual so the feature
    // filter trusts them less than fully-seen frames.
    residual: centerErr / Math.max(1e-3, quadPx) + softPatches * 0.12,
    features,
    whiteRGB,
    softPatches,
    // Prediction-filled slots report the PRIOR learned color instead of a
    // sample — the predicted spot is exactly where the glare/shadow that hid
    // the patch sits, and blending that in would tint the learned palette.
    // (Blending prior with prior is a no-op, so occluded slots simply don't
    // adapt this frame.)
    quadColors: {
      red: synthetic[0] ? prior.colors.red : sampleMeanRGB(rgba, width, height, picks[0].x, picks[0].y),
      green: synthetic[1] ? prior.colors.green : sampleMeanRGB(rgba, width, height, picks[1].x, picks[1].y),
      blue: synthetic[2] ? prior.colors.blue : sampleMeanRGB(rgba, width, height, picks[2].x, picks[2].y),
    },
  };
  return { observation, debug };
}

// ============================================================================
// SHAPE CARD — black patches on white, identity carried by TOPOLOGY.
//
// Patches at the color card's quadrant positions: a solid DISK (0 holes)
// where red was, a RING (1 hole) where green was, a TWO-HOLE patch where
// blue was, and the same small dot in the white quadrant. Hole count is
// invariant under any perspective/rotation/blur-skew, so — like color —
// identity is read directly off each blob and no combinatorial layout
// search is needed. Two filaments, and immune to white balance and
// low-light desaturation.
// ============================================================================

type ShapeBlob = Blob & { holes: number };

// Dark connected components + the number of enclosed light regions (holes)
// per component. A light region is a hole iff it is NOT reachable from the
// patch border by flooding light pixels; its enclosing dark component is
// the pixel to the left of its leftmost pixel.
function darkBlobsWithHoles(
  gray: Uint8Array,
  width: number,
  height: number,
  threshold: number,
  minArea: number,
  weights?: Uint8Array, // per-pixel confidence → sub-pixel weighted centroids
): ShapeBlob[] {
  const size = width * height;
  const labels = new Int32Array(size); // 0 = light/unvisited, >0 = dark blob id
  const stack = new Int32Array(size);
  type Stat = {
    area: number;
    wsum: number;
    sx: number;
    sy: number;
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
    holeAreas: number[];
  };
  const stats: Stat[] = [];

  // Pass 1: label dark components.
  for (let start = 0; start < size; start += 1) {
    if (labels[start] !== 0 || gray[start] >= threshold) continue;
    const id = stats.length + 1;
    let top = 0;
    stack[top++] = start;
    labels[start] = id;
    const s: Stat = { area: 0, wsum: 0, sx: 0, sy: 0, minX: width, maxX: 0, minY: height, maxY: 0, holeAreas: [] };
    while (top > 0) {
      const idx = stack[--top];
      const x = idx % width;
      const y = (idx / width) | 0;
      s.area += 1;
      const wt = weights ? weights[idx] : 1;
      s.wsum += wt;
      s.sx += x * wt;
      s.sy += y * wt;
      if (x < s.minX) s.minX = x;
      if (x > s.maxX) s.maxX = x;
      if (y < s.minY) s.minY = y;
      if (y > s.maxY) s.maxY = y;
      if (x > 0 && labels[idx - 1] === 0 && gray[idx - 1] < threshold) {
        labels[idx - 1] = id;
        stack[top++] = idx - 1;
      }
      if (x < width - 1 && labels[idx + 1] === 0 && gray[idx + 1] < threshold) {
        labels[idx + 1] = id;
        stack[top++] = idx + 1;
      }
      if (y > 0 && labels[idx - width] === 0 && gray[idx - width] < threshold) {
        labels[idx - width] = id;
        stack[top++] = idx - width;
      }
      if (y < height - 1 && labels[idx + width] === 0 && gray[idx + width] < threshold) {
        labels[idx + width] = id;
        stack[top++] = idx + width;
      }
    }
    stats.push(s);
  }

  // Pass 2: flood light pixels from the border (the "outside").
  const lightSeen = new Uint8Array(size);
  let top = 0;
  const pushLight = (idx: number) => {
    if (!lightSeen[idx] && labels[idx] === 0 && gray[idx] >= threshold) {
      lightSeen[idx] = 1;
      stack[top++] = idx;
    }
  };
  for (let x = 0; x < width; x += 1) {
    pushLight(x);
    pushLight(size - width + x);
  }
  for (let y = 0; y < height; y += 1) {
    pushLight(y * width);
    pushLight(y * width + width - 1);
  }
  while (top > 0) {
    const idx = stack[--top];
    const x = idx % width;
    if (x > 0) pushLight(idx - 1);
    if (x < width - 1) pushLight(idx + 1);
    if (idx >= width) pushLight(idx - width);
    if (idx < size - width) pushLight(idx + width);
  }

  // Pass 3: every remaining light component is a hole in some dark blob.
  for (let start = 0; start < size; start += 1) {
    if (lightSeen[start] || labels[start] !== 0 || gray[start] < threshold) continue;
    let holeTop = 0;
    stack[holeTop++] = start;
    lightSeen[start] = 1;
    let area = 0;
    let leftIdx = start;
    let leftX = start % width;
    while (holeTop > 0) {
      const idx = stack[--holeTop];
      const x = idx % width;
      area += 1;
      if (x < leftX) {
        leftX = x;
        leftIdx = idx;
      }
      const grow = (nIdx: number) => {
        if (!lightSeen[nIdx] && labels[nIdx] === 0 && gray[nIdx] >= threshold) {
          lightSeen[nIdx] = 1;
          stack[holeTop++] = nIdx;
        }
      };
      if (x > 0) grow(idx - 1);
      if (x < width - 1) grow(idx + 1);
      if (idx >= width) grow(idx - width);
      if (idx < size - width) grow(idx + width);
    }
    if (leftX === 0) continue;
    const owner = labels[leftIdx - 1];
    if (owner > 0) stats[owner - 1].holeAreas.push(area);
  }

  const blobs: ShapeBlob[] = [];
  for (const s of stats) {
    if (s.area < minArea) continue;
    // Count only SIGNIFICANT holes: a real ring hole is ~13% of its blob
    // and a two-hole's holes ~11% each, while glare speckle and layer-line
    // washout inside a printed patch are well under 1% — without this
    // filter a speck turns the ring into a fake "two-hole".
    const holeMin = Math.max(4, s.area * 0.02);
    const holes = s.holeAreas.reduce((count, a) => (a >= holeMin ? count + 1 : count), 0);
    blobs.push({
      x: s.sx / Math.max(1, s.wsum) + 0.5,
      y: s.sy / Math.max(1, s.wsum) + 0.5,
      area: s.area,
      w: s.maxX - s.minX + 1,
      h: s.maxY - s.minY + 1,
      x0: s.minX,
      y0: s.minY,
      holes,
    });
  }
  return blobs;
}

// Expected patch fill vs the quadrant cell (disks, not full squares).
const SHAPE_FILL = 0.62;

// Which patches must match their topology exactly for a lock. A slot with
// its requirement OFF accepts ANY red blob — the layout gates, chirality,
// and the other slots then carry the identity. `dot: true` additionally
// forbids the geometry-only (dotless) fallback.
export type ShapeRequirements = { disk: boolean; ring: boolean; two: boolean; dot: boolean };
export const DEFAULT_SHAPE_REQUIREMENTS: ShapeRequirements = { disk: true, ring: true, two: true, dot: false };

// Is this pixel a shape-card MARK? RED ONLY: red must be genuinely strong
// AND beat the better of green/blue by an absolute margin AND dominate
// proportionally. No darkness fallback — "dark" sweeps in half of any
// indoor scene (Otsu always splits the histogram), while a saturated red
// print has no competition in an ordinary room. Sensitivity levels trade
// selectivity against lighting: pick the strictest level whose tint
// overlay still fills the card's patches solidly.
export type RedSensitivity = "strict" | "normal" | "forgiving" | "wide";
const RED_GATES: Record<RedSensitivity, { floor: number; margin: number; ratio: number }> = {
  strict: { floor: 110, margin: 55, ratio: 1.45 }, // studio-bright, dead clean
  normal: { floor: 90, margin: 40, ratio: 1.3 }, // default
  forgiving: { floor: 75, margin: 28, ratio: 1.2 }, // dim rooms, mild washout
  wide: { floor: 60, margin: 16, ratio: 1.1 }, // last resort — expect strays
};
export function isShapeMark(r: number, g: number, b: number, sensitivity: RedSensitivity = "normal"): boolean {
  const gate = RED_GATES[sensitivity];
  const mx = Math.max(g, b);
  return r >= gate.floor && r - mx >= gate.margin && r >= mx * gate.ratio;
}

// The shape card's ANCHOR dots: small marks at the pattern's edge midpoints,
// in card units. They nearly double the correspondence baseline (~65 mm vs
// the 36 mm patch triangle), which is what pose accuracy scales with — found
// opportunistically after the lock and folded into a least-squares refit.
const SHAPE_ANCHORS: [number, number][] = [
  [0, 0.9],
  [0, -0.9],
  [0.9, 0],
  [-0.9, 0],
];

// Least-squares affine from arbitrary (card-point, image-point) pairs.
function fitAffinePairs(cardPts: number[][], imagePts: Point[]): Affine | null {
  let suu = 0;
  let suv = 0;
  let svv = 0;
  let su = 0;
  let sv = 0;
  const n = imagePts.length;
  let sxu = 0;
  let sxv = 0;
  let sx = 0;
  let syu = 0;
  let syv = 0;
  let sy = 0;
  for (let i = 0; i < n; i += 1) {
    const u = cardPts[i][0];
    const v = cardPts[i][1];
    const { x, y } = imagePts[i];
    suu += u * u;
    suv += u * v;
    svv += v * v;
    su += u;
    sv += v;
    sxu += x * u;
    sxv += x * v;
    sx += x;
    syu += y * u;
    syv += y * v;
    sy += y;
  }
  const m = [
    [suu, suv, su],
    [suv, svv, sv],
    [su, sv, n],
  ];
  const solve3 = (r0: number, r1: number, r2: number): [number, number, number] | null => {
    const a = m.map((row) => row.slice());
    const b = [r0, r1, r2];
    for (let col = 0; col < 3; col += 1) {
      let pivot = col;
      for (let row = col + 1; row < 3; row += 1) if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
      if (Math.abs(a[pivot][col]) < 1e-9) return null;
      if (pivot !== col) {
        const tmpRow = a[col];
        a[col] = a[pivot];
        a[pivot] = tmpRow;
        const tmpB = b[col];
        b[col] = b[pivot];
        b[pivot] = tmpB;
      }
      for (let row = col + 1; row < 3; row += 1) {
        const factor = a[row][col] / a[col][col];
        for (let k = col; k < 3; k += 1) a[row][k] -= factor * a[col][k];
        b[row] -= factor * b[col];
      }
    }
    const out: [number, number, number] = [0, 0, 0];
    for (let row = 2; row >= 0; row -= 1) {
      let acc = b[row];
      for (let k = row + 1; k < 3; k += 1) acc -= a[row][k] * out[k];
      out[row] = acc / a[row][row];
    }
    return out;
  };
  const rx = solve3(sxu, sxv, sx);
  const ry = solve3(syu, syv, sy);
  if (!rx || !ry) return null;
  return { a11: rx[0], a12: rx[1], bx: rx[2], a21: ry[0], a22: ry[1], by: ry[2] };
}

// Detect the black/white shape card in an RGBA patch. Identity by hole
// count: disk = 0, ring = 1, two-hole = 2 — then the exact same layout
// gates as the color card (size similarity, affine area consistency,
// chirality, dot confirmation with bounding-box check, dotless fallback).
export function analyzeShapeFlag(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  normPx: number,
  offsetX = 0,
  offsetY = 0,
  sensitivity: RedSensitivity = "normal",
  needs: ShapeRequirements = DEFAULT_SHAPE_REQUIREMENTS,
): { observation: FlagObservation | null; debug: FlagDebug } {
  const n = width * height;
  // Mark mask (0 = mark, 255 = background): strictly red pixels only.
  // Confidence = red's margin over the runner-up channel, for sub-pixel
  // weighted centroids.
  const mark = new Uint8Array(n);
  const conf = new Uint8Array(n);
  for (let i = 0; i < n; i += 1) {
    const o = i * 4;
    if (isShapeMark(rgba[o], rgba[o + 1], rgba[o + 2], sensitivity)) {
      mark[i] = 0;
      conf[i] = Math.min(255, Math.max(1, rgba[o] - Math.max(rgba[o + 1], rgba[o + 2])));
    } else {
      mark[i] = 255;
    }
  }
  const all = darkBlobsWithHoles(mark, width, height, 128, 5, conf).sort((a, b) => b.area - a.area);

  // Rings and two-holes are rare in a scene — those identify the card.
  // Solid dark blobs are everywhere, so disks lean on the layout gates.
  // A slot whose requirement is OFF accepts any red blob at all — layout,
  // size, and chirality then do the identifying.
  const disks = all.filter((b) => (!needs.disk || b.holes === 0) && b.area >= 12).slice(0, needs.disk ? 10 : 12);
  const rings = all.filter((b) => (!needs.ring || b.holes === 1) && b.area >= 12).slice(0, needs.ring ? 6 : 12);
  const twoholes = all.filter((b) => (!needs.two || b.holes >= 2) && b.area >= 12).slice(0, needs.two ? 6 : 12);
  const shift = (p: Point): Point => ({ x: p.x + offsetX, y: p.y + offsetY });

  const debug: FlagDebug = {
    failStage: "missing-shape",
    threshold: 0,
    blobCount: all.length,
    blobs: all.slice(0, 24).map((b) => ({ x: b.x + offsetX, y: b.y + offsetY, area: b.area })),
    clusterCount: 0,
    tileCandidates: [],
    quad: null,
    quadScore: null,
    dotCandidates: [],
    dotErrCells: null,
    cellPx: null,
    residual: null,
    colorCounts: { red: disks.length, green: rings.length, blue: twoholes.length },
    colorGates: null,
  };
  if (disks.length === 0 || rings.length === 0 || twoholes.length === 0) return { observation: null, debug };

  type Triple = {
    disk: ShapeBlob;
    ring: ShapeBlob;
    two: ShapeBlob;
    affine: Affine;
    quadPx: number;
    det: number;
    score: number;
    chiralityOk: boolean;
    predictedDot: Point;
  };
  const gates = { triples: 0, sizeOk: 0, areaOk: 0, nearestDotErr: Infinity };
  let bestGeom: Triple | null = null;
  let bestDot: { triple: Triple; dot: Point; dotErr: number; dotBlob: Blob; boxHit: boolean } | null = null;

  debug.failStage = "bad-shape-layout";
  for (const disk of disks)
    for (const ring of rings)
      for (const two of twoholes) {
        // Relaxed pools can overlap — a triple must be three distinct blobs.
        if (disk === ring || disk === two || ring === two) continue;
        gates.triples += 1;
        const areas = [disk.area, ring.area, two.area];
        const maxA = Math.max(...areas);
        const minA = Math.min(...areas);
        if (maxA > minA * 3.2) continue;
        gates.sizeOk += 1;
        // Disk sits in red's template slot, ring in green's, two-hole in
        // blue's — same mutual layout, so the same math applies verbatim.
        const affine = affineFrom3(disk, ring, two);
        const quadPx = (Math.hypot(affine.a11, affine.a21) + Math.hypot(affine.a12, affine.a22)) / 2;
        if (quadPx < 4) continue;
        const detSigned = affine.a11 * affine.a22 - affine.a12 * affine.a21;
        const det = Math.abs(detSigned);
        if (det < 1) continue;
        const chiralityOk = detSigned > 0;
        let areaScore = 0;
        let plausible = true;
        for (const area of areas) {
          const ratio = area / det;
          if (ratio < 0.25 || ratio > 1.1) plausible = false;
          areaScore += Math.abs(Math.log(Math.max(1e-3, ratio / SHAPE_FILL)));
        }
        if (!plausible) continue;
        gates.areaOk += 1;
        const score = areaScore + (chiralityOk ? 0 : 1.5);
        const predictedDot = {
          x: affine.a11 * COLOR_TEMPLATE.dot[0] + affine.a12 * COLOR_TEMPLATE.dot[1] + affine.bx,
          y: affine.a21 * COLOR_TEMPLATE.dot[0] + affine.a22 * COLOR_TEMPLATE.dot[1] + affine.by,
        };
        const triple: Triple = { disk, ring, two, affine, quadPx, det, score, chiralityOk, predictedDot };
        if (!bestGeom || score < bestGeom.score) bestGeom = triple;
        // The dot: a SMALL solid dark blob near the white quadrant center.
        for (const cand of all) {
          if (cand === disk || cand.holes !== 0) continue;
          if (cand.area > det * 0.25) continue;
          const err = Math.hypot(cand.x - predictedDot.x, cand.y - predictedDot.y);
          if (err / quadPx < gates.nearestDotErr) gates.nearestDotErr = err / quadPx;
          const dotSized = Math.max(cand.w, cand.h) <= quadPx * 0.85;
          const boxHit =
            dotSized &&
            predictedDot.x >= cand.x0 - 2 &&
            predictedDot.x <= cand.x0 + cand.w + 2 &&
            predictedDot.y >= cand.y0 - 2 &&
            predictedDot.y <= cand.y0 + cand.h + 2;
          if (err > quadPx * 0.6 && !boxHit) continue;
          const effErr = boxHit ? Math.min(err, quadPx * 0.25) : err;
          if (!bestDot || effErr + score * quadPx * 0.1 < bestDot.dotErr + bestDot.triple.score * bestDot.triple.quadPx * 0.1) {
            bestDot = { triple, dot: { x: cand.x, y: cand.y }, dotErr: effErr, dotBlob: cand, boxHit };
          }
        }
      }

  // Geometry-only fallback under tighter gates; the white quadrant must be
  // free of large dark blobs (the small dot is exempt via the area gate).
  // With needs.dot the fallback is disabled — no dot, no lock.
  let chosen: {
    triple: Triple;
    dot: Point;
    dotErr: number;
    dotless: boolean;
    dotBlob?: Blob;
    boxHit?: boolean;
  } | null = bestDot ? { ...bestDot, dotless: false } : null;
  if (!chosen && bestGeom && !needs.dot) {
    const t = bestGeom;
    const areas = [t.disk.area, t.ring.area, t.two.area];
    const tightSize = Math.max(...areas) <= Math.min(...areas) * 2.2;
    const tightArea = areas.every((a) => a / t.det >= 0.35 && a / t.det <= 0.95);
    const fourthClear = !all.some(
      (b) =>
        b !== t.disk &&
        b !== t.ring &&
        b !== t.two &&
        b.area > t.det * 0.3 &&
        Math.hypot(b.x - t.predictedDot.x, b.y - t.predictedDot.y) < t.quadPx * 0.45,
    );
    if (tightSize && tightArea && fourthClear && t.chiralityOk) {
      chosen = { triple: t, dot: t.predictedDot, dotErr: t.quadPx * 0.6, dotless: true };
    }
  }

  const reference = chosen?.triple ?? bestGeom;
  if (reference) {
    debug.tileCandidates = [reference.disk, reference.ring, reference.two].map((b) => shift({ x: b.x, y: b.y }));
    debug.cellPx = reference.quadPx;
  }
  debug.colorGates = {
    triples: gates.triples,
    sizeOk: gates.sizeOk,
    areaOk: gates.areaOk,
    nearestDotErrCells: Number.isFinite(gates.nearestDotErr) ? gates.nearestDotErr : null,
    predictedDot: reference ? shift(reference.predictedDot) : null,
    dotless: chosen?.dotless ?? false,
    strengths: null,
    chiralityOk: reference ? reference.chiralityOk : null,
    dotBoxHit: chosen?.boxHit ?? false,
    dotBox: chosen?.dotBlob
      ? { x: chosen.dotBlob.x0 + offsetX, y: chosen.dotBlob.y0 + offsetY, w: chosen.dotBlob.w, h: chosen.dotBlob.h }
      : null,
    anchors: null,
  };
  if (chosen) {
    debug.dotCandidates = [shift(chosen.dot)];
    debug.dotErrCells = chosen.dotErr / chosen.triple.quadPx;
    debug.residual = chosen.dotErr / chosen.triple.quadPx;
  }
  if (!chosen) return { observation: null, debug };

  debug.failStage = "ok";
  // Wide-baseline refinement: hunt for the 4 edge anchor dots at their
  // predicted positions and least-squares refit the pose over EVERY found
  // correspondence (patches + dot + anchors, up to 8 points spanning the
  // whole card instead of the 36 mm patch triangle).
  const A0 = chosen.triple.affine;
  const cardPts: number[][] = [
    [COLOR_TEMPLATE.red[0], COLOR_TEMPLATE.red[1]],
    [COLOR_TEMPLATE.green[0], COLOR_TEMPLATE.green[1]],
    [COLOR_TEMPLATE.blue[0], COLOR_TEMPLATE.blue[1]],
  ];
  const imgPts: Point[] = [chosen.triple.disk, chosen.triple.ring, chosen.triple.two].map((b) => ({ x: b.x, y: b.y }));
  if (!chosen.dotless) {
    cardPts.push([COLOR_TEMPLATE.dot[0], COLOR_TEMPLATE.dot[1]]);
    imgPts.push(chosen.dot);
  }
  let anchorsFound = 0;
  for (const [u, v] of SHAPE_ANCHORS) {
    const px = A0.a11 * u + A0.a12 * v + A0.bx;
    const py = A0.a21 * u + A0.a22 * v + A0.by;
    let best: ShapeBlob | null = null;
    let bestD = chosen.triple.quadPx * 0.22;
    for (const cand of all) {
      if (cand.holes !== 0 || cand.area > chosen.triple.det * 0.06) continue;
      const d = Math.hypot(cand.x - px, cand.y - py);
      if (d < bestD) {
        bestD = d;
        best = cand;
      }
    }
    if (best) {
      anchorsFound += 1;
      cardPts.push([u, v]);
      imgPts.push({ x: best.x, y: best.y });
    }
  }
  // Robust refit against the known geometry: outlier correspondences are
  // rejected by reprojection error, and the homography must beat the affine
  // to be trusted.
  const pose = robustPose(cardPts, imgPts);
  const A = pose.affine ?? A0;
  const quadPxR = (Math.hypot(A.a11, A.a21) + Math.hypot(A.a12, A.a22)) / 2;
  if (debug.colorGates) debug.colorGates.anchors = anchorsFound;
  debug.cellPx = quadPxR;
  const fitResidual = pose.rmsPx !== null ? pose.rmsPx / Math.max(1e-3, quadPxR) : null;
  if (fitResidual !== null) debug.residual = fitResidual;
  let features = featuresFromAffine(A, offsetX, offsetY, normPx);
  if (pose.h) features = featuresFromHomography(pose.h, offsetX, offsetY, normPx);
  const observation: FlagObservation = {
    tiles: [chosen.triple.disk, chosen.triple.ring, chosen.triple.two].map((b) => shift({ x: b.x, y: b.y })),
    dot: shift(chosen.dot),
    center: { x: A.bx + offsetX, y: A.by + offsetY },
    cellPx: quadPxR,
    residual: fitResidual ?? chosen.dotErr / chosen.triple.quadPx,
    features,
    // The actual RED of this print under this lighting, sampled at each
    // patch — seeds the shape tracker's learned-chromaticity mask.
    quadColors: {
      red: sampleMeanRGB(rgba, width, height, chosen.triple.disk.x, chosen.triple.disk.y),
      green: sampleMeanRGB(rgba, width, height, chosen.triple.ring.x, chosen.triple.ring.y),
      blue: sampleMeanRGB(rgba, width, height, chosen.triple.two.x, chosen.triple.two.y),
    },
  };
  return { observation, debug };
}

// ============================================================================
// SHAPE TRACK MODE — after acquisition locks the red shape card, identity is
// carried by POSITION CONTINUITY, not per-frame topology: hole counts are
// only needed at acquisition/re-seek, so a ring hole blurring shut for a few
// frames no longer breaks the lock. Pixels are matched against the LEARNED
// red chromaticity of the actual print under the actual lighting (far more
// forgiving than the acquisition red gate), each patch slot takes the blob
// nearest its predicted position, and — like the color tracker — the gates
// open with the lock's maturity (prior.relax).
// ============================================================================
export function trackShapeFlag(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  normPx: number,
  offsetX: number,
  offsetY: number,
  prior: FlagTrackPrior,
): { observation: FlagObservation | null; debug: FlagDebug } {
  const n = width * height;
  const relax = Math.min(1, Math.max(0, prior.relax ?? 0));
  const seek = 1 + Math.min(2, 0.2 * (prior.misses ?? 0)); // re-seek widening (see color tracker)
  const learned = [chroma(prior.colors.red), chroma(prior.colors.green), chroma(prior.colors.blue)];
  const chromaThr = (prior.chromaWindow ?? 0.09) * (1 + 0.8 * relax);
  const sumLo = 40 - 20 * relax;
  const sumHi = 730 + 25 * relax;
  // ONE mask: near ANY of the three learned patch colors — they're all the
  // same red print, just sampled under slightly different shading, so
  // splitting them into per-color masks would fragment each patch.
  // Illuminant-locus LUT (see the color tracker) — any ink class → mark.
  const igS = prior.illumGain ?? [1, 1, 1];
  const litS = (c: readonly [number, number, number]): [number, number, number] => [
    c[0] * igS[0],
    c[1] * igS[1],
    c[2] * igS[2],
  ];
  const lutS = buildInkLUT(
    { red: litS(prior.colors.red), green: litS(prior.colors.green), blue: litS(prior.colors.blue) },
    chromaThr,
  );
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i += 1) {
    const o = i * 4;
    const r = rgba[o];
    const g = rgba[o + 1];
    const b = rgba[o + 2];
    const sum = r + g + b;
    if (sum < sumLo || sum > sumHi) continue;
    if (lutS[lutIndex(r / sum, g / sum)] > 0) mask[i] = 1;
  }

  const minArea = Math.max(6, prior.cellPx * prior.cellPx * (0.05 - 0.025 * relax));
  const blobs = maskBlobs(mask, width, height, minArea);
  const shift = (p: Point): Point => ({ x: p.x + offsetX, y: p.y + offsetY });
  const localPredCenter = { x: prior.predictedCenter.x - offsetX, y: prior.predictedCenter.y - offsetY };
  const [a11, a12, a21, a22] = prior.affine;
  const predictQuad = (u: number, v: number): Point => ({
    x: localPredCenter.x + a11 * u + a12 * v,
    y: localPredCenter.y + a21 * u + a22 * v,
  });
  const templates = [COLOR_TEMPLATE.red, COLOR_TEMPLATE.green, COLOR_TEMPLATE.blue] as const;

  // Greedy slot assignment: nearest blob to each predicted patch position,
  // scored by distance + learned-color mismatch; a blob can serve one slot.
  const pickRadius = prior.cellPx * (2.6 + 3.4 * relax) * seek;
  const picks: (Blob | null)[] = [null, null, null];
  const counts = [0, 0, 0];
  const used = new Set<Blob>();
  for (let k = 0; k < 3; k += 1) {
    const predicted = predictQuad(templates[k][0], templates[k][1]);
    let best: Blob | null = null;
    let bestScore = Infinity;
    for (const blob of blobs) {
      if (used.has(blob)) continue;
      const dist = Math.hypot(blob.x - predicted.x, blob.y - predicted.y);
      if (dist > pickRadius) continue;
      counts[k] += 1;
      const [sr, sg, sb] = sampleMeanRGB(rgba, width, height, blob.x, blob.y, 2);
      const ssum = Math.max(1, sr + sg + sb);
      const dcr = sr / ssum - learned[k][0];
      const dcg = sg / ssum - learned[k][1];
      const score = dist + Math.sqrt(dcr * dcr + dcg * dcg) * prior.cellPx * 12;
      if (score < bestScore) {
        bestScore = score;
        best = blob;
      }
    }
    picks[k] = best;
    if (best) used.add(best);
  }

  const debug: FlagDebug = {
    failStage: "missing-shape",
    threshold: 0,
    blobCount: blobs.length,
    blobs: [],
    clusterCount: 0,
    tileCandidates: picks.filter((p): p is Blob => p !== null).map((b) => shift({ x: b.x, y: b.y })),
    quad: null,
    quadScore: null,
    dotCandidates: [],
    dotErrCells: null,
    cellPx: prior.cellPx,
    residual: null,
    colorCounts: { red: counts[0], green: counts[1], blue: counts[2] },
    colorGates: null,
  };
  // THREE-COLOR RULE: all three patches must be genuinely found — no
  // synthetic fill-ins (see the color tracker).
  const synthetic = [false, false, false];
  const foundCount = (picks[0] ? 1 : 0) + (picks[1] ? 1 : 0) + (picks[2] ? 1 : 0);
  if (foundCount < 3) return { observation: null, debug };
  if (!picks[0] || !picks[1] || !picks[2]) return { observation: null, debug };

  const affine = affineFrom3(picks[0], picks[1], picks[2]);
  const quadPx = (Math.hypot(affine.a11, affine.a21) + Math.hypot(affine.a12, affine.a22)) / 2;
  debug.failStage = "bad-shape-layout";
  const scaleRatio = quadPx / Math.max(1e-3, prior.cellPx);
  if (scaleRatio < 0.5 - 0.35 * relax || scaleRatio > 1.5 + 0.3 * relax) return { observation: null, debug };
  const centerErr = Math.hypot(affine.bx - localPredCenter.x, affine.by - localPredCenter.y);
  if (centerErr > prior.cellPx * (2.6 + 3.4 * relax) * seek) return { observation: null, debug };
  // NO chirality gate (by request) — see the color tracker.

  debug.failStage = "ok";
  debug.cellPx = quadPx;
  debug.residual = centerErr / Math.max(1e-3, quadPx);
  const dotLocal = predictQuad(COLOR_TEMPLATE.dot[0], COLOR_TEMPLATE.dot[1]);
  // Perspective: gather every extra correspondence visible in the learned
  // mask — the dot and the four edge anchors at their predicted spots — and
  // fit a homography when we have 4+ points.
  let features = featuresFromAffine(affine, offsetX, offsetY, normPx);
  {
    const smalls = maskBlobs(mask, width, height, 4);
    const cardPts: number[][] = [
      [COLOR_TEMPLATE.red[0], COLOR_TEMPLATE.red[1]],
      [COLOR_TEMPLATE.green[0], COLOR_TEMPLATE.green[1]],
      [COLOR_TEMPLATE.blue[0], COLOR_TEMPLATE.blue[1]],
    ];
    const imgPts: Point[] = picks.map((b) => ({ x: (b as Blob).x, y: (b as Blob).y }));
    const grab = (u: number, v: number, radius: number, maxArea: number) => {
      const at = predictQuad(u, v);
      let best: Blob | null = null;
      let bestD = radius;
      for (const cand of smalls) {
        if (cand.area > maxArea) continue;
        const d = Math.hypot(cand.x - at.x, cand.y - at.y);
        if (d < bestD) {
          bestD = d;
          best = cand;
        }
      }
      if (best) {
        cardPts.push([u, v]);
        imgPts.push({ x: best.x, y: best.y });
      }
    };
    grab(COLOR_TEMPLATE.dot[0], COLOR_TEMPLATE.dot[1], quadPx * 0.5, quadPx * quadPx * 0.25);
    for (const [u, v] of SHAPE_ANCHORS) grab(u, v, quadPx * 0.25, quadPx * quadPx * 0.06);
    if (imgPts.length >= 4) {
      // Robust: outlier anchors rejected by reprojection before the pose
      // is trusted (see robustPose).
      const pose = robustPose(cardPts, imgPts);
      if (pose.h) features = featuresFromHomography(pose.h, offsetX, offsetY, normPx);
      if (pose.rmsPx !== null) debug.residual = pose.rmsPx / Math.max(1e-3, quadPx);
    }
  }
  const observation: FlagObservation = {
    tiles: picks.map((b) => shift({ x: (b as Blob).x, y: (b as Blob).y })),
    dot: shift(dotLocal),
    center: { x: affine.bx + offsetX, y: affine.by + offsetY },
    cellPx: quadPx,
    residual: centerErr / Math.max(1e-3, quadPx),
    features,
    // Prediction-filled slots report the PRIOR learned color instead of a
    // sample — the predicted spot is exactly where the glare/shadow that hid
    // the patch sits, and blending that in would tint the learned palette.
    // (Blending prior with prior is a no-op, so occluded slots simply don't
    // adapt this frame.)
    quadColors: {
      red: synthetic[0] ? prior.colors.red : sampleMeanRGB(rgba, width, height, picks[0].x, picks[0].y),
      green: synthetic[1] ? prior.colors.green : sampleMeanRGB(rgba, width, height, picks[1].x, picks[1].y),
      blue: synthetic[2] ? prior.colors.blue : sampleMeanRGB(rgba, width, height, picks[2].x, picks[2].y),
    },
  };
  return { observation, debug };
}
