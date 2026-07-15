// Precision-target statistics, in the conventions precision shooters use.
//
// Position units are TARGET RADII from the target center (0,0 = center,
// 1.0 = the outermost scoring ring), screen-y down. The standard measures:
//
// - MPI (mean point of impact): the group's centroid. Its offset from the
//   point of aim is the zero error, quoted as distance + clock direction,
//   and split into windage (x) / elevation (y) components.
// - MEAN RADIUS: average distance of shots from the MPI. Statistically far
//   more stable than extreme spread (every shot contributes) — the measure
//   ballisticians prefer.
// - EXTREME SPREAD: max center-to-center distance between any two shots —
//   the classic "group size" people quote at the range.
// - RADIAL σ and σx/σy: dispersion measures. A tall, narrow group (σy ≫ σx)
//   points at breathing/vertical stringing; wide (σx ≫ σy) at trigger jerk.
// - CEP50: median radius from MPI — the circle containing half the shots.
// - SPLITS / HIT FACTOR: time between consecutive shots, string time, and
//   score-per-second (the USPSA scoring economy of accuracy vs speed).
// - K-MEANS (k=2): detects whether the "group" is really two groups — the
//   classic flinch signature is a main cluster plus a satellite low-left.

export type TargetShot = { x: number; y: number; atMs: number };

export type ClusterResult = {
  k: 1 | 2;
  centers: { x: number; y: number; count: number }[];
  separationRatio: number; // center distance vs within-cluster spread
};

export type TargetStats = {
  count: number;
  totalScore: number;
  avgScore: number;
  mpiX: number;
  mpiY: number;
  offset: number; // |MPI − center|, in target radii
  offsetClock: string; // "4 o'clock" (— for a centered group)
  meanRadius: number; // avg distance from MPI
  radialSd: number; // σ of distance from MPI
  sdX: number;
  sdY: number;
  extremeSpread: number;
  cep50: number;
  avgFromCenter: number; // accuracy incl. zero error
  avgSplitMs: number | null;
  bestSplitMs: number | null;
  stringSec: number | null; // first shot → last shot
  hitFactor: number | null; // score / string seconds
  clusters: ClusterResult;
};

// Ten scoring rings, a tenth of the radius each: r ≤ 0.1 → 10 … ≤ 1.0 → 1,
// outside the rings → 0.
export function scoreRing(r: number): number {
  if (r > 1) return 0;
  return Math.max(1, 10 - Math.floor(r * 10));
}

// Clock direction of (x, y) as seen on the target (y down): 12 o'clock = up.
export function clockDirection(x: number, y: number): string {
  const deg = (Math.atan2(x, -y) * 180) / Math.PI;
  const hour = ((Math.round(deg / 30) + 12) % 12) || 12;
  return `${hour} o'clock`;
}

// Deterministic 2-means (seeded from the farthest pair — no randomness, so
// results are reproducible shot to shot).
function twoMeans(shots: TargetShot[]): ClusterResult {
  const one = (): ClusterResult => {
    const cx = shots.reduce((s, p) => s + p.x, 0) / Math.max(1, shots.length);
    const cy = shots.reduce((s, p) => s + p.y, 0) / Math.max(1, shots.length);
    return { k: 1, centers: [{ x: cx, y: cy, count: shots.length }], separationRatio: 0 };
  };
  if (shots.length < 4) return one();

  // Seeds: the two shots farthest apart.
  let ai = 0;
  let bi = 1;
  let best = -1;
  for (let i = 0; i < shots.length; i += 1)
    for (let j = i + 1; j < shots.length; j += 1) {
      const d = Math.hypot(shots[i].x - shots[j].x, shots[i].y - shots[j].y);
      if (d > best) {
        best = d;
        ai = i;
        bi = j;
      }
    }
  let a = { x: shots[ai].x, y: shots[ai].y };
  let b = { x: shots[bi].x, y: shots[bi].y };
  let assign: number[] = [];
  for (let iter = 0; iter < 12; iter += 1) {
    assign = shots.map((p) => (Math.hypot(p.x - a.x, p.y - a.y) <= Math.hypot(p.x - b.x, p.y - b.y) ? 0 : 1));
    const sums = [
      { x: 0, y: 0, n: 0 },
      { x: 0, y: 0, n: 0 },
    ];
    shots.forEach((p, i) => {
      sums[assign[i]].x += p.x;
      sums[assign[i]].y += p.y;
      sums[assign[i]].n += 1;
    });
    if (sums[0].n === 0 || sums[1].n === 0) return one();
    a = { x: sums[0].x / sums[0].n, y: sums[0].y / sums[0].n };
    b = { x: sums[1].x / sums[1].n, y: sums[1].y / sums[1].n };
  }
  const nA = assign.filter((v) => v === 0).length;
  const nB = shots.length - nA;
  if (nA < 2 || nB < 2) return one();
  const within =
    shots.reduce((s, p, i) => {
      const c = assign[i] === 0 ? a : b;
      return s + Math.hypot(p.x - c.x, p.y - c.y);
    }, 0) / shots.length;
  const separation = Math.hypot(a.x - b.x, a.y - b.y);
  const ratio = separation / Math.max(1e-6, 2 * within);
  if (ratio <= 1) return one();
  return {
    k: 2,
    centers: [
      { x: a.x, y: a.y, count: nA },
      { x: b.x, y: b.y, count: nB },
    ],
    separationRatio: ratio,
  };
}

export function computeTargetStats(shots: TargetShot[]): TargetStats | null {
  const n = shots.length;
  if (n === 0) return null;
  const mpiX = shots.reduce((s, p) => s + p.x, 0) / n;
  const mpiY = shots.reduce((s, p) => s + p.y, 0) / n;
  const radii = shots.map((p) => Math.hypot(p.x - mpiX, p.y - mpiY));
  const meanRadius = radii.reduce((s, r) => s + r, 0) / n;
  const radialSd = Math.sqrt(radii.reduce((s, r) => s + (r - meanRadius) ** 2, 0) / n);
  const sdX = Math.sqrt(shots.reduce((s, p) => s + (p.x - mpiX) ** 2, 0) / n);
  const sdY = Math.sqrt(shots.reduce((s, p) => s + (p.y - mpiY) ** 2, 0) / n);
  let extremeSpread = 0;
  for (let i = 0; i < n; i += 1)
    for (let j = i + 1; j < n; j += 1)
      extremeSpread = Math.max(extremeSpread, Math.hypot(shots[i].x - shots[j].x, shots[i].y - shots[j].y));
  const sortedRadii = [...radii].sort((x, y) => x - y);
  const cep50 = sortedRadii[Math.floor((n - 1) / 2)];
  const fromCenter = shots.map((p) => Math.hypot(p.x, p.y));
  const avgFromCenter = fromCenter.reduce((s, r) => s + r, 0) / n;
  const totalScore = fromCenter.reduce((s, r) => s + scoreRing(r), 0);
  const offset = Math.hypot(mpiX, mpiY);

  const ordered = [...shots].sort((p, q) => p.atMs - q.atMs);
  const splits: number[] = [];
  for (let i = 1; i < ordered.length; i += 1) splits.push(ordered[i].atMs - ordered[i - 1].atMs);
  // Splits over 20 s mean the shooter paused — not cadence.
  const realSplits = splits.filter((ms) => ms < 20000);
  const avgSplitMs = realSplits.length > 0 ? realSplits.reduce((s, v) => s + v, 0) / realSplits.length : null;
  const bestSplitMs = realSplits.length > 0 ? Math.min(...realSplits) : null;
  const stringSec = n >= 2 ? (ordered[n - 1].atMs - ordered[0].atMs) / 1000 : null;
  const hitFactor = stringSec !== null && stringSec > 0 ? totalScore / stringSec : null;

  return {
    count: n,
    totalScore,
    avgScore: totalScore / n,
    mpiX,
    mpiY,
    offset,
    offsetClock: offset < 0.02 ? "centered" : clockDirection(mpiX, mpiY),
    meanRadius,
    radialSd,
    sdX,
    sdY,
    extremeSpread,
    cep50,
    avgFromCenter,
    avgSplitMs,
    bestSplitMs,
    stringSec,
    hitFactor,
    clusters: twoMeans(shots),
  };
}
