"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { clockDirection } from "@/app/lib/dryfire/targetStats";
import {
  aggregateShots,
  clearShotLog,
  dayBuckets,
  loadShotLog,
  type LoggedShot,
  type ShotSource,
} from "@/app/lib/dryfire/shotLog";

// Every distance on this page is in REFERENCE RADII (R): each shot is stored
// as its deviation from the point the bullet SHOULD have hit — the bullseye
// center or the called zone's center — normalized by that target's radius.
// 1.0 R = the edge of whatever was being shot at, on any screen, any drill.

const SOURCE_META: Record<ShotSource, { name: string; dot: string }> = {
  target: { name: "Bullseye", dot: "#38bdf8" },
  drill: { name: "Called drills", dot: "#fbbf24" },
};

function fmtR(v: number): string {
  return `${v.toFixed(2)} R`;
}

function StatCard({ label, value, sub, title }: { label: string; value: string; sub?: string; title?: string }) {
  return (
    <div className="rounded-lg border border-gray-800 bg-neutral-950 px-3 py-2.5" title={title}>
      <p className="text-[10px] uppercase tracking-wide text-gray-500">{label}</p>
      <p className="mt-0.5 text-lg font-semibold text-white">{value}</p>
      {sub ? <p className="text-[11px] text-gray-500">{sub}</p> : null}
    </div>
  );
}

// The composite impact plot: every shot re-centered on its own intended
// point, so the picture reads like one giant target — center = perfect.
function DeviationPlot({ shots }: { shots: LoggedShot[] }) {
  const R = 150; // px per reference radius at scale 1
  const MAXR = 2.2; // plot window, in reference radii
  const size = 2 * MAXR * R * 0.5; // rendered at half scale via viewBox
  const agg = aggregateShots(shots);
  const clipped = shots.filter((s) => Math.hypot(s.dx, s.dy) > MAXR).length;
  return (
    <div className="rounded-lg border border-gray-800 bg-neutral-950 p-3">
      <p className="text-[11px] uppercase tracking-wide text-gray-500">
        Where they should be vs where they went
      </p>
      <p className="mb-2 text-[11px] text-gray-600">
        Every shot re-centered on its intended point. The bold ring = the target/zone edge (1 R).
      </p>
      <svg viewBox={`${-MAXR * R} ${-MAXR * R} ${2 * MAXR * R} ${2 * MAXR * R}`} className="mx-auto block w-full max-w-[420px]" style={{ height: size }}>
        {/* rings */}
        {[0.5, 1, 1.5, 2].map((r) => (
          <circle
            key={r}
            cx={0}
            cy={0}
            r={r * R}
            fill="none"
            stroke={r === 1 ? "#e5e7eb" : "#374151"}
            strokeWidth={r === 1 ? 2 : 1}
          />
        ))}
        <line x1={-MAXR * R} y1={0} x2={MAXR * R} y2={0} stroke="#374151" strokeWidth={1} />
        <line x1={0} y1={-MAXR * R} x2={0} y2={MAXR * R} stroke="#374151" strokeWidth={1} />
        {/* shots */}
        {shots.map((s, i) => {
          const r = Math.hypot(s.dx, s.dy);
          if (r > MAXR) return null;
          return (
            <circle
              key={i}
              cx={s.dx * R}
              cy={s.dy * R}
              r={4}
              fill={SOURCE_META[s.src].dot}
              opacity={s.hit ? 0.8 : 0.45}
            />
          );
        })}
        {/* CEP50 about the MPI, then the MPI cross and robust median cross */}
        {agg?.stats ? (
          <>
            <circle
              cx={agg.stats.mpiX * R}
              cy={agg.stats.mpiY * R}
              r={agg.stats.cep50 * R}
              fill="none"
              stroke="#f87171"
              strokeWidth={1.5}
              strokeDasharray="6 5"
            />
            <g stroke="#f87171" strokeWidth={2.5}>
              <line x1={agg.stats.mpiX * R - 9} y1={agg.stats.mpiY * R} x2={agg.stats.mpiX * R + 9} y2={agg.stats.mpiY * R} />
              <line x1={agg.stats.mpiX * R} y1={agg.stats.mpiY * R - 9} x2={agg.stats.mpiX * R} y2={agg.stats.mpiY * R + 9} />
            </g>
            <g stroke="#4ade80" strokeWidth={2}>
              <line x1={agg.medianDx * R - 7} y1={agg.medianDy * R - 7} x2={agg.medianDx * R + 7} y2={agg.medianDy * R + 7} />
              <line x1={agg.medianDx * R - 7} y1={agg.medianDy * R + 7} x2={agg.medianDx * R + 7} y2={agg.medianDy * R - 7} />
            </g>
          </>
        ) : null}
      </svg>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-gray-500">
        <span><span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: SOURCE_META.target.dot }} />bullseye</span>
        <span><span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: SOURCE_META.drill.dot }} />called drill</span>
        <span className="text-red-400">＋ MPI · ⤬ CEP50</span>
        <span className="text-green-400">⨯ median bias (robust)</span>
        {clipped > 0 ? <span>({clipped} far miss{clipped === 1 ? "" : "es"} outside the plot)</span> : null}
      </div>
    </div>
  );
}

// Daily trend: wobble (mean radius) and zero error (offset), in R.
function TrendChart({ shots }: { shots: LoggedShot[] }) {
  const buckets = dayBuckets(shots).slice(-30);
  if (buckets.length < 2) {
    return (
      <div className="rounded-lg border border-gray-800 bg-neutral-950 p-3 text-[12px] text-gray-500">
        Trends appear once you have shots on two or more days.
      </div>
    );
  }
  const W = 560;
  const H = 150;
  const PAD = 26;
  const maxY = Math.max(1, ...buckets.map((b) => Math.max(b.meanRadius, b.offset))) * 1.15;
  const maxN = Math.max(...buckets.map((b) => b.count));
  const x = (i: number) => PAD + (i * (W - 2 * PAD)) / Math.max(1, buckets.length - 1);
  const y = (v: number) => H - PAD - (v / maxY) * (H - 2 * PAD);
  const line = (get: (b: (typeof buckets)[number]) => number) =>
    buckets.map((b, i) => `${x(i).toFixed(1)},${y(get(b)).toFixed(1)}`).join(" ");
  return (
    <div className="rounded-lg border border-gray-800 bg-neutral-950 p-3">
      <p className="text-[11px] uppercase tracking-wide text-gray-500">Trend · last {buckets.length} training days</p>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-1 w-full">
        {/* per-day volume, faint bars */}
        {buckets.map((b, i) => (
          <rect
            key={b.day}
            x={x(i) - 4}
            y={H - PAD - (b.count / maxN) * 28}
            width={8}
            height={(b.count / maxN) * 28}
            fill="#1f2937"
          />
        ))}
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#374151" />
        {/* 1R guide: at/above this your wobble is the whole target */}
        {maxY > 1 ? (
          <line x1={PAD} y1={y(1)} x2={W - PAD} y2={y(1)} stroke="#4b5563" strokeDasharray="3 4" />
        ) : null}
        <polyline points={line((b) => b.meanRadius)} fill="none" stroke="#38bdf8" strokeWidth={2} />
        <polyline points={line((b) => b.offset)} fill="none" stroke="#f87171" strokeWidth={2} />
        {buckets.map((b, i) => (
          <g key={b.day}>
            <circle cx={x(i)} cy={y(b.meanRadius)} r={2.6} fill="#38bdf8" />
            <circle cx={x(i)} cy={y(b.offset)} r={2.6} fill="#f87171" />
          </g>
        ))}
        <text x={PAD} y={12} fontSize={10} fill="#6b7280">
          {buckets[0].day}
        </text>
        <text x={W - PAD} y={12} fontSize={10} fill="#6b7280" textAnchor="end">
          {buckets[buckets.length - 1].day}
        </text>
      </svg>
      <div className="mt-1 flex gap-4 text-[11px] text-gray-500">
        <span className="text-sky-400">— wobble (mean radius)</span>
        <span className="text-red-400">— zero error (MPI offset)</span>
        <span className="text-gray-600">▮ shots/day</span>
      </div>
    </div>
  );
}

function BreakdownTable({ shots }: { shots: LoggedShot[] }) {
  const groups = useMemo(() => {
    const rows: { name: string; shots: LoggedShot[] }[] = [];
    (Object.keys(SOURCE_META) as ShotSource[]).forEach((src) => {
      const inSrc = shots.filter((s) => s.src === src);
      if (inSrc.length > 0) rows.push({ name: SOURCE_META[src].name, shots: inSrc });
    });
    // Per called color, most-shot first.
    const byLabel = new Map<string, LoggedShot[]>();
    for (const s of shots) {
      if (s.src !== "drill") continue;
      const list = byLabel.get(s.label);
      if (list) list.push(s);
      else byLabel.set(s.label, [s]);
    }
    [...byLabel.entries()]
      .sort((a, b) => b[1].length - a[1].length)
      .slice(0, 8)
      .forEach(([label, list]) => rows.push({ name: `· called “${label}”`, shots: list }));
    return rows;
  }, [shots]);
  if (groups.length === 0) return null;
  return (
    <div className="overflow-x-auto rounded-lg border border-gray-800 bg-neutral-950 p-3">
      <p className="mb-2 text-[11px] uppercase tracking-wide text-gray-500">Breakdown</p>
      <table className="w-full min-w-[520px] text-left text-[12px]">
        <thead>
          <tr className="text-[10px] uppercase tracking-wide text-gray-600">
            <th className="pb-1 font-medium">context</th>
            <th className="pb-1 font-medium">shots</th>
            <th className="pb-1 font-medium">hit %</th>
            <th className="pb-1 font-medium" title="Systematic bias: how far the group center sits from where the bullets should be">zero error</th>
            <th className="pb-1 font-medium" title="Wobble: average distance from the group's own center">mean radius</th>
            <th className="pb-1 font-medium" title="Median radius — circle containing half the shots">CEP50</th>
          </tr>
        </thead>
        <tbody className="text-gray-300">
          {groups.map(({ name, shots: list }) => {
            const agg = aggregateShots(list);
            if (!agg?.stats) return null;
            return (
              <tr key={name} className="border-t border-gray-900">
                <td className="py-1.5 pr-2">{name}</td>
                <td className="py-1.5 pr-2 tabular-nums">{agg.count}</td>
                <td className="py-1.5 pr-2 tabular-nums">{Math.round(agg.hitRate * 100)}%</td>
                <td className="py-1.5 pr-2 tabular-nums">
                  {fmtR(agg.stats.offset)}
                  <span className="text-gray-600"> {agg.stats.offsetClock !== "centered" ? `@ ${agg.stats.offsetClock}` : ""}</span>
                </td>
                <td className="py-1.5 pr-2 tabular-nums">{fmtR(agg.stats.meanRadius)}</td>
                <td className="py-1.5 tabular-nums">{fmtR(agg.stats.cep50)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function StatsView() {
  const [log, setLog] = useState<LoggedShot[]>([]);
  const [filter, setFilter] = useState<ShotSource | "all">("all");
  useEffect(() => {
    setLog(loadShotLog());
  }, []);

  const shots = useMemo(() => (filter === "all" ? log : log.filter((s) => s.src === filter)), [log, filter]);
  const agg = useMemo(() => aggregateShots(shots), [shots]);

  return (
    <main className="min-h-screen bg-neutral-900 px-4 py-6 text-gray-100">
      <div className="mx-auto max-w-4xl space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Shooting statistics</h1>
            <p className="text-[12px] text-gray-500">
              Every dry-fire shot, measured against where the bullet <em>should</em> have gone. Units are
              reference radii (R): 1.0 = the edge of whatever was called.
            </p>
          </div>
          <Link href="/dryfire" className="shrink-0 text-sm text-sky-300 hover:underline">
            Dry-fire →
          </Link>
        </div>

        {log.length === 0 ? (
          <div className="rounded-lg border border-gray-800 bg-neutral-950 p-6 text-sm text-gray-400">
            No shots logged yet. Train on the <Link href="/dryfire" className="text-sky-300 hover:underline">dry-fire page</Link> —
            every trigger break on the bullseye or in a called drill is recorded here automatically
            (locally, in this browser).
          </div>
        ) : (
          <>
            <div className="flex items-center gap-1.5">
              {(["all", "target", "drill"] as const).map((f) => (
                <button
                  key={f}
                  type="button"
                  onClick={() => setFilter(f)}
                  aria-pressed={filter === f}
                  className={`rounded px-2.5 py-1 text-[12px] transition ${
                    filter === f ? "bg-sky-500/20 text-sky-100" : "text-gray-400 hover:bg-neutral-950"
                  }`}
                >
                  {f === "all" ? `All (${log.length})` : `${SOURCE_META[f].name} (${log.filter((s) => s.src === f).length})`}
                </button>
              ))}
            </div>

            {agg?.stats ? (
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <StatCard label="shots" value={String(agg.count)} sub={`${Math.round(agg.hitRate * 100)}% on target`} />
                <StatCard
                  label="zero error"
                  value={fmtR(agg.stats.offset)}
                  sub={agg.stats.offsetClock === "centered" ? "centered" : `toward ${agg.stats.offsetClock}`}
                  title="Where the whole group sits vs where it should: sights/calibration/zero, not wobble"
                />
                <StatCard
                  label="median bias"
                  value={fmtR(Math.hypot(agg.medianDx, agg.medianDy))}
                  sub={
                    Math.hypot(agg.medianDx, agg.medianDy) < 0.02
                      ? "centered"
                      : `toward ${clockDirection(agg.medianDx, agg.medianDy)}`
                  }
                  title="Robust (median) version of zero error — wild flyers can't move it"
                />
                <StatCard
                  label="wobble"
                  value={fmtR(agg.stats.meanRadius)}
                  sub={`CEP50 ${fmtR(agg.stats.cep50)}`}
                  title="Mean radius about the group's own center — pure precision, zero error removed"
                />
                <StatCard
                  label="σx / σy"
                  value={`${agg.stats.sdX.toFixed(2)} / ${agg.stats.sdY.toFixed(2)}`}
                  sub={
                    agg.stats.sdY > agg.stats.sdX * 1.5
                      ? "vertical stringing"
                      : agg.stats.sdX > agg.stats.sdY * 1.5
                        ? "horizontal stringing"
                        : "even dispersion"
                  }
                  title="Per-axis spread: tall groups → breathing/vertical; wide → trigger jerk/grip"
                />
                <StatCard label="extreme spread" value={fmtR(agg.stats.extremeSpread)} sub="worst two shots" />
                <StatCard
                  label="two groups?"
                  value={agg.stats.clusters.k === 2 ? "yes" : "no"}
                  sub={
                    agg.stats.clusters.k === 2
                      ? `satellite of ${Math.min(...agg.stats.clusters.centers.map((c) => c.count))} shots — flinch signature`
                      : "one clean cluster"
                  }
                  title="Deterministic 2-means: a main cluster plus a low-left satellite is the classic flinch"
                />
                <StatCard
                  label="logged since"
                  value={new Date(agg.firstT).toLocaleDateString()}
                  sub={`latest ${new Date(agg.lastT).toLocaleDateString()}`}
                />
              </div>
            ) : null}

            <DeviationPlot shots={shots} />
            <TrendChart shots={shots} />
            <BreakdownTable shots={shots} />

            <div className="flex items-center justify-between text-[11px] text-gray-600">
              <span>Stored locally in this browser · capped at 4000 shots (oldest roll off)</span>
              <button
                type="button"
                onClick={() => {
                  if (window.confirm("Delete the entire shot log? This cannot be undone.")) {
                    clearShotLog();
                    setLog([]);
                  }
                }}
                className="rounded px-2 py-1 text-red-400/80 transition hover:bg-red-500/10 hover:text-red-300"
              >
                Clear all data
              </button>
            </div>
          </>
        )}
      </div>
    </main>
  );
}
