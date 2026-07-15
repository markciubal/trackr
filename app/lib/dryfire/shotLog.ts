// Persistent point-of-aim vs point-of-impact log.
//
// Every training surface that knows BOTH where the bullet SHOULD have gone
// (the intended centerpoint — bullseye center, called zone center) and where
// the shot actually landed appends a record here. Deviations are stored in
// REFERENCE RADII — the target's scoring radius or the called zone's radius —
// so shots from different surfaces, screen sizes, and distances live on one
// comparable scale, and (0,0) is always "exactly where it should be".
//
// The log lives in localStorage (per browser, no account needed) and the
// /stats page reads it. Kept deliberately small and flat so any future
// surface (live-fire scan, new drill types) can call appendShot too.

import { computeTargetStats, type TargetStats } from "./targetStats";

export type ShotSource = "target" | "drill";

export type LoggedShot = {
  t: number; // epoch ms
  src: ShotSource;
  label: string; // "bullseye", or the called zone's color/attribute
  dx: number; // impact − intended centerpoint, in reference radii (x right)
  dy: number; // (y down, same convention as targetStats)
  hit: boolean; // landed inside the reference ring / correct zone
};

const KEY = "trackr-shot-log-v1";
const CAP = 4000; // ~a season of dry fire; oldest entries roll off

function isLoggedShot(v: unknown): v is LoggedShot {
  if (typeof v !== "object" || v === null) return false;
  const s = v as Record<string, unknown>;
  return (
    typeof s.t === "number" &&
    (s.src === "target" || s.src === "drill") &&
    typeof s.label === "string" &&
    typeof s.dx === "number" &&
    Number.isFinite(s.dx) &&
    typeof s.dy === "number" &&
    Number.isFinite(s.dy) &&
    typeof s.hit === "boolean"
  );
}

export function loadShotLog(): LoggedShot[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isLoggedShot);
  } catch {
    return [];
  }
}

export function appendShot(shot: LoggedShot): void {
  if (typeof window === "undefined") return;
  try {
    const log = loadShotLog();
    log.push(shot);
    window.localStorage.setItem(KEY, JSON.stringify(log.slice(-CAP)));
  } catch {
    // Quota/serialization problems must never break a training session.
  }
}

export function clearShotLog(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(KEY);
  } catch {
    // ignore
  }
}

// ---- Aggregation for the stats page ----------------------------------------

export type LogAggregate = {
  count: number;
  hitRate: number; // 0..1
  firstT: number;
  lastT: number;
  // Full precision-group statistics over the DEVIATIONS: since every shot is
  // re-centered on its own intended point, MPI = systematic aim bias, mean
  // radius/CEP = wobble, exactly as on a single target.
  stats: TargetStats | null;
  // Robust (median) bias — outlier shots can't drag these.
  medianDx: number;
  medianDy: number;
};

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function aggregateShots(shots: LoggedShot[]): LogAggregate | null {
  if (shots.length === 0) return null;
  const stats = computeTargetStats(shots.map((s) => ({ x: s.dx, y: s.dy, atMs: s.t })));
  return {
    count: shots.length,
    hitRate: shots.filter((s) => s.hit).length / shots.length,
    firstT: Math.min(...shots.map((s) => s.t)),
    lastT: Math.max(...shots.map((s) => s.t)),
    stats,
    medianDx: median(shots.map((s) => s.dx)),
    medianDy: median(shots.map((s) => s.dy)),
  };
}

export type DayBucket = {
  day: string; // local YYYY-MM-DD
  count: number;
  hitRate: number;
  offset: number; // |mean deviation| that day — zero/bias error
  meanRadius: number; // spread about that day's own MPI — wobble
};

// Per-local-day trend buckets, oldest → newest, days with no shots omitted.
export function dayBuckets(shots: LoggedShot[]): DayBucket[] {
  const byDay = new Map<string, LoggedShot[]>();
  for (const s of shots) {
    const d = new Date(s.t);
    const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    const list = byDay.get(key);
    if (list) list.push(s);
    else byDay.set(key, [s]);
  }
  return [...byDay.entries()]
    .sort((a, b) => (a[0] < b[0] ? -1 : 1))
    .map(([day, list]) => {
      const mx = list.reduce((s, p) => s + p.dx, 0) / list.length;
      const my = list.reduce((s, p) => s + p.dy, 0) / list.length;
      const meanRadius = list.reduce((s, p) => s + Math.hypot(p.dx - mx, p.dy - my), 0) / list.length;
      return {
        day,
        count: list.length,
        hitRate: list.filter((s) => s.hit).length / list.length,
        offset: Math.hypot(mx, my),
        meanRadius,
      };
    });
}
