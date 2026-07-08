// Shared types + pure statistics for the saved-shot library (browse / combine /
// stats). All geometry is computed in INCHES internally; the UI converts to the
// chosen display unit at render time.

export type SavedShot = {
  n: number;
  x: number; // pixel position in the session's reference frame
  y: number;
  dpx: number;
  din: number | null; // estimated hole diameter, inches
  t: number;
  method: string;
  group: number | null;
  conf: number;
};

export type SavedTarget = {
  widthInches?: number | null;
  heightInches?: number | null;
  pixelsPerInch?: number | null;
  unit?: string | null;
  roi?: unknown;
  // Per-group aim points (where each group was aiming), keyed by the group id that
  // matches each shot's `group`. Saved so stats can normalize on intent (aim),
  // measuring bias/offset-from-aim, not just dispersion around the impact centroid.
  aimPoints?: Record<string, { x: number; y: number }> | null;
} | null;

export type SessionMeta = {
  id: string;
  name: string | null;
  shot_count: number;
  created_at: string;
};

export type SessionFull = SessionMeta & {
  target: SavedTarget;
  shots: SavedShot[];
};

export type Pt = { x: number; y: number; sessionId: string; din: number | null; n: number; aimed: boolean };

export type DisplayUnit = "in" | "cm" | "mm";

// How to normalize a session's shots before combining/measuring:
//  - "none": raw frame coordinates.
//  - "poi":  centered on the session's impact centroid (dispersion comparison).
//  - "aim":  centered on each group's AIM point, so the centroid offset is the
//            real bias from where you intended to hit (accuracy normalization).
export type AlignMode = "none" | "poi" | "aim";

// inches → display-unit multiplier (1 in = 2.54 cm = 25.4 mm).
export function unitFactor(unit: DisplayUnit): number {
  return unit === "cm" ? 2.54 : unit === "mm" ? 25.4 : 1;
}

export function unitLabel(unit: DisplayUnit): string {
  return unit;
}

// inches-per-pixel for a session; 0 when uncalibrated (positions stay in pixels).
export function sessionInchesPerPx(target: SavedTarget): number {
  const ppi = target?.pixelsPerInch;
  return ppi && Number.isFinite(ppi) && ppi > 0 ? 1 / ppi : 0;
}

function centroidOf(shots: SavedShot[]): { x: number; y: number } {
  let sx = 0;
  let sy = 0;
  for (const s of shots) {
    sx += s.x;
    sy += s.y;
  }
  const n = Math.max(1, shots.length);
  return { x: sx / n, y: sy / n };
}

// Per-group impact centroids (fallback reference when a group has no aim point).
function groupCentroids(shots: SavedShot[]): Map<number | null, { x: number; y: number }> {
  const sums = new Map<number | null, { x: number; y: number; n: number }>();
  for (const s of shots) {
    const e = sums.get(s.group) ?? { x: 0, y: 0, n: 0 };
    e.x += s.x;
    e.y += s.y;
    e.n += 1;
    sums.set(s.group, e);
  }
  const out = new Map<number | null, { x: number; y: number }>();
  for (const [k, e] of sums) out.set(k, { x: e.x / e.n, y: e.y / e.n });
  return out;
}

export function sessionHasAimPoints(session: SessionFull): boolean {
  const ap = session.target?.aimPoints;
  return !!ap && Object.keys(ap).length > 0;
}

// Convert one session's shots to inch-space points under the chosen alignment.
// Uncalibrated sessions (no pixels-per-inch) fall back to pixel units.
export function shotsToPoints(session: SessionFull, mode: AlignMode): Pt[] {
  const inPerPx = sessionInchesPerPx(session.target);
  const scale = inPerPx > 0 ? inPerPx : 1;
  const aim = session.target?.aimPoints ?? {};
  const cents = mode === "aim" ? groupCentroids(session.shots) : null;
  const sessionCentroid = mode === "poi" ? centroidOf(session.shots) : { x: 0, y: 0 };

  return session.shots.map((s) => {
    let rx = 0;
    let ry = 0;
    let aimed = false;
    if (mode === "poi") {
      rx = sessionCentroid.x;
      ry = sessionCentroid.y;
    } else if (mode === "aim") {
      const a = s.group !== null && s.group !== undefined ? aim[String(s.group)] : undefined;
      if (a) {
        rx = a.x;
        ry = a.y;
        aimed = true;
      } else {
        // No aim point for this group → fall back to its impact centroid so the
        // shot is still positioned sensibly (but it carries no bias signal).
        const c = cents?.get(s.group) ?? centroidOf(session.shots);
        rx = c.x;
        ry = c.y;
      }
    }
    return {
      x: (s.x - rx) * scale,
      y: (s.y - ry) * scale,
      sessionId: session.id,
      din: s.din,
      n: s.n,
      aimed,
    };
  });
}

export type GroupStats = {
  count: number;
  centroidX: number;
  centroidY: number;
  extremeSpread: number; // max center-to-center distance (inches)
  meanRadius: number; // average distance from centroid (inches)
  rmsRadius: number;
  cep: number; // median radius — 50% circular error probable (inches)
  stdX: number;
  stdY: number;
  meanDiameter: number | null; // average hole diameter (inches)
};

export function computeStats(points: Pt[]): GroupStats | null {
  const n = points.length;
  if (n === 0) return null;
  let sx = 0;
  let sy = 0;
  for (const p of points) {
    sx += p.x;
    sy += p.y;
  }
  const cx = sx / n;
  const cy = sy / n;

  const radii: number[] = [];
  let sumR = 0;
  let sumR2 = 0;
  let varX = 0;
  let varY = 0;
  for (const p of points) {
    const dx = p.x - cx;
    const dy = p.y - cy;
    const r = Math.hypot(dx, dy);
    radii.push(r);
    sumR += r;
    sumR2 += r * r;
    varX += dx * dx;
    varY += dy * dy;
  }

  let extreme = 0;
  // O(n^2) pairwise is fine for libraries of up to a few hundred shots.
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const d = Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y);
      if (d > extreme) extreme = d;
    }
  }

  radii.sort((a, b) => a - b);
  const mid = Math.floor(radii.length / 2);
  const cep = radii.length % 2 === 0 ? (radii[mid - 1] + radii[mid]) / 2 : radii[mid];

  const diameters = points.map((p) => p.din).filter((d): d is number => d !== null && Number.isFinite(d));
  const meanDiameter = diameters.length > 0 ? diameters.reduce((a, b) => a + b, 0) / diameters.length : null;

  return {
    count: n,
    centroidX: cx,
    centroidY: cy,
    extremeSpread: extreme,
    meanRadius: sumR / n,
    rmsRadius: Math.sqrt(sumR2 / n),
    cep,
    stdX: Math.sqrt(varX / n),
    stdY: Math.sqrt(varY / n),
    meanDiameter,
  };
}

// "Group size" the way shooters quote it: extreme center-to-center spread plus one
// bullet diameter (edge-to-edge). Falls back to center-to-center when no diameter.
export function groupSize(stats: GroupStats): number {
  return stats.meanDiameter ? stats.extremeSpread + stats.meanDiameter : stats.extremeSpread;
}
