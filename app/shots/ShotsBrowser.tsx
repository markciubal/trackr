"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type AlignMode,
  type DisplayUnit,
  type GroupStats,
  type Pt,
  type SessionFull,
  type SessionMeta,
  computeStats,
  groupSize,
  sessionHasAimPoints,
  sessionInchesPerPx,
  shotsToPoints,
  unitFactor,
} from "@/app/lib/shots/stats";

const ALIGN_LABELS: Record<AlignMode, string> = {
  none: "None",
  poi: "Point of impact",
  aim: "Aim point",
};

const SESSION_COLORS = [
  "#38bdf8",
  "#f472b6",
  "#a3e635",
  "#fbbf24",
  "#c084fc",
  "#fb7185",
  "#34d399",
  "#60a5fa",
  "#f59e0b",
  "#2dd4bf",
];

function colorForIndex(index: number): string {
  return SESSION_COLORS[index % SESSION_COLORS.length];
}

function fmtDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  } catch {
    return iso.slice(0, 10);
  }
}

export function ShotsBrowser({ initialSessions }: { initialSessions: SessionMeta[] }) {
  const [sessions, setSessions] = useState<SessionMeta[]>(initialSessions);
  const [selectedIds, setSelectedIds] = useState<string[]>(initialSessions.length > 0 ? [initialSessions[0].id] : []);
  const [fullById, setFullById] = useState<Record<string, SessionFull>>({});
  const [loadingIds, setLoadingIds] = useState<string[]>([]);
  const [alignMode, setAlignMode] = useState<AlignMode>("aim");
  const [alignTouched, setAlignTouched] = useState(false);
  const [unit, setUnit] = useState<DisplayUnit>("in");
  const [error, setError] = useState<string | null>(null);

  // Stable color per session id (by its position in the list).
  const colorById = useMemo<Record<string, string>>(() => {
    const map: Record<string, string> = {};
    sessions.forEach((s, i) => {
      map[s.id] = colorForIndex(i);
    });
    return map;
  }, [sessions]);

  // Fetch full data (shots) for any selected session we don't have yet.
  useEffect(() => {
    const missing = selectedIds.filter((id) => !fullById[id] && !loadingIds.includes(id));
    if (missing.length === 0) return;
    setLoadingIds((prev) => [...prev, ...missing]);
    let cancelled = false;
    Promise.all(
      missing.map(async (id) => {
        try {
          const res = await fetch(`/api/annotations/${id}`, { cache: "no-store" });
          const json = (await res.json().catch(() => ({}))) as { session?: SessionFull; error?: string };
          if (!res.ok || !json.session) throw new Error(json.error ?? "Failed to load session.");
          return json.session;
        } catch (e) {
          if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load a session.");
          return null;
        }
      }),
    ).then((loaded) => {
      if (cancelled) return;
      setFullById((prev) => {
        const next = { ...prev };
        for (const s of loaded) if (s) next[s.id] = s;
        return next;
      });
      setLoadingIds((prev) => prev.filter((id) => !missing.includes(id)));
    });
    return () => {
      cancelled = true;
    };
  }, [selectedIds, fullById, loadingIds]);

  const toggleSession = useCallback((id: string) => {
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }, []);

  const deleteSession = useCallback(async (id: string) => {
    if (!window.confirm("Delete this saved session? This can't be undone.")) return;
    try {
      const res = await fetch(`/api/annotations/${id}`, { method: "DELETE" });
      if (!res.ok) {
        const json = (await res.json().catch(() => ({}))) as { error?: string };
        throw new Error(json.error ?? "Delete failed.");
      }
      setSessions((prev) => prev.filter((s) => s.id !== id));
      setSelectedIds((prev) => prev.filter((x) => x !== id));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed.");
    }
  }, []);

  // Selected sessions that have loaded, and whether any selected session is uncalibrated.
  const loadedSelected = useMemo(
    () => selectedIds.map((id) => fullById[id]).filter((s): s is SessionFull => Boolean(s)),
    [selectedIds, fullById],
  );
  const anyUncalibrated = loadedSelected.some((s) => sessionInchesPerPx(s.target) <= 0);
  const unitsArePixels = loadedSelected.length > 0 && loadedSelected.every((s) => sessionInchesPerPx(s.target) <= 0);
  const anyAimPoints = loadedSelected.some(sessionHasAimPoints);

  // Default the alignment to "aim" when aim points exist, otherwise "poi" — until
  // the user picks one themselves.
  useEffect(() => {
    if (alignTouched || loadedSelected.length === 0) return;
    setAlignMode(anyAimPoints ? "aim" : "poi");
  }, [alignTouched, anyAimPoints, loadedSelected.length]);

  // Combined inch-space points across the selected sessions.
  const points = useMemo<Pt[]>(
    () => loadedSelected.flatMap((s) => shotsToPoints(s, alignMode)),
    [loadedSelected, alignMode],
  );
  const aimedCount = useMemo(() => points.filter((p) => p.aimed).length, [points]);

  const combinedStats = useMemo(() => computeStats(points), [points]);
  const perSession = useMemo(
    () =>
      loadedSelected.map((s) => ({
        session: s,
        stats: computeStats(shotsToPoints(s, alignMode)),
      })),
    [loadedSelected, alignMode],
  );

  return (
    <div className="grid gap-4 lg:grid-cols-[20rem_1fr]">
      {/* Session list — pick one or many to combine. */}
      <aside className="rounded-xl border border-gray-700 bg-neutral-950 p-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-300">Sessions</h2>
          <div className="flex gap-2 text-[11px]">
            <button
              type="button"
              onClick={() => setSelectedIds(sessions.map((s) => s.id))}
              className="rounded border border-gray-700 px-2 py-0.5 text-gray-300 transition hover:bg-neutral-800"
            >
              All
            </button>
            <button
              type="button"
              onClick={() => setSelectedIds([])}
              className="rounded border border-gray-700 px-2 py-0.5 text-gray-300 transition hover:bg-neutral-800"
            >
              None
            </button>
          </div>
        </div>
        {sessions.length === 0 ? (
          <p className="mt-3 text-xs text-gray-500">
            No saved sessions yet. Run a scan and use the Save step to add shots to your library.
          </p>
        ) : (
          <ul className="mt-3 space-y-1.5">
            {sessions.map((s) => {
              const selected = selectedIds.includes(s.id);
              const loading = loadingIds.includes(s.id);
              return (
                <li key={s.id}>
                  <div
                    className={`flex items-center gap-2 rounded-md border px-2 py-1.5 transition ${
                      selected ? "border-sky-500/50 bg-sky-500/10" : "border-gray-800 hover:bg-neutral-900"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selected}
                      onChange={() => toggleSession(s.id)}
                      aria-label={`Select ${s.name ?? "session"}`}
                    />
                    <span
                      className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                      style={{ backgroundColor: colorById[s.id] }}
                    />
                    <button type="button" onClick={() => toggleSession(s.id)} className="min-w-0 flex-1 text-left">
                      <span className="block truncate text-xs text-gray-100">{s.name || "Untitled session"}</span>
                      <span className="block text-[10px] text-gray-500">
                        {s.shot_count} shot{s.shot_count === 1 ? "" : "s"} · {fmtDate(s.created_at)}
                        {loading ? " · loading…" : ""}
                      </span>
                    </button>
                    <button
                      type="button"
                      onClick={() => deleteSession(s.id)}
                      aria-label="Delete session"
                      className="shrink-0 rounded p-1 text-gray-500 transition hover:bg-rose-500/15 hover:text-rose-300"
                      title="Delete"
                    >
                      ✕
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
        {error ? <p className="mt-3 text-[11px] text-rose-300">{error}</p> : null}
      </aside>

      {/* Plot + statistics for the combined selection. */}
      <section className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-gray-700 bg-neutral-950 p-3">
          <p className="text-sm text-gray-300">
            {selectedIds.length === 0
              ? "Select one or more sessions to combine."
              : `${selectedIds.length} session${selectedIds.length === 1 ? "" : "s"} · ${points.length} shots`}
          </p>
          <div className="flex flex-wrap items-center gap-3 text-[11px]">
            <div className="flex items-center gap-1.5">
              <span className="text-gray-400">Align</span>
              <div className="inline-flex overflow-hidden rounded-md border border-gray-700">
                {(["none", "poi", "aim"] as AlignMode[]).map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => {
                      setAlignTouched(true);
                      setAlignMode(m);
                    }}
                    disabled={m === "aim" && !anyAimPoints}
                    className={`px-2 py-0.5 transition disabled:cursor-not-allowed disabled:opacity-40 ${
                      alignMode === m ? "bg-sky-500/20 text-sky-100" : "text-gray-300 hover:bg-neutral-800"
                    }`}
                    aria-pressed={alignMode === m}
                    title={m === "aim" && !anyAimPoints ? "No aim points saved in these sessions" : ALIGN_LABELS[m]}
                  >
                    {ALIGN_LABELS[m]}
                  </button>
                ))}
              </div>
            </div>
            <div className="inline-flex overflow-hidden rounded-md border border-gray-700">
              {(["in", "cm", "mm"] as DisplayUnit[]).map((u) => (
                <button
                  key={u}
                  type="button"
                  onClick={() => setUnit(u)}
                  className={`px-2 py-0.5 transition ${unit === u ? "bg-sky-500/20 text-sky-100" : "text-gray-300 hover:bg-neutral-800"}`}
                  aria-pressed={unit === u}
                >
                  {u}
                </button>
              ))}
            </div>
          </div>
        </div>

        {alignMode === "aim" ? (
          <p className="rounded-md border border-sky-500/25 bg-sky-500/10 px-3 py-2 text-[11px] text-sky-100">
            Aligned on each group&apos;s aim point — the centroid offset below is your{" "}
            <span className="font-semibold">bias from aim</span> (accuracy); the spread is your precision.
            {aimedCount < points.length
              ? ` ${points.length - aimedCount} of ${points.length} shots had no aim point and fall back to their group's center.`
              : ""}
          </p>
        ) : null}

        {unitsArePixels ? (
          <p className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-[11px] text-amber-100">
            These sessions weren&apos;t calibrated, so distances are shown in pixels (the unit toggle has no effect).
          </p>
        ) : anyUncalibrated ? (
          <p className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-[11px] text-amber-100">
            Some selected sessions aren&apos;t calibrated; their shots are mixed in using pixel distances.
          </p>
        ) : null}

        <ScatterPlot
          points={points}
          colorById={colorById}
          unit={unit}
          pixels={unitsArePixels}
          stats={combinedStats}
          alignMode={alignMode}
        />

        {combinedStats ? (
          <StatsPanel
            title="Combined"
            stats={combinedStats}
            unit={unit}
            pixels={unitsArePixels}
            alignMode={alignMode}
            emphasis
          />
        ) : selectedIds.length > 0 ? (
          <p className="text-sm text-gray-500">Loading shots…</p>
        ) : null}

        {perSession.length > 1 ? (
          <div className="space-y-2">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-400">Per session</h3>
            <div className="grid gap-2 sm:grid-cols-2">
              {perSession.map(({ session, stats }) =>
                stats ? (
                  <StatsPanel
                    key={session.id}
                    title={session.name || "Untitled"}
                    color={colorById[session.id]}
                    stats={stats}
                    unit={unit}
                    pixels={sessionInchesPerPx(session.target) <= 0}
                    alignMode={alignMode}
                  />
                ) : null,
              )}
            </div>
          </div>
        ) : null}
      </section>
    </div>
  );
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-gray-400">{label}</dt>
      <dd className="tabular-nums text-gray-100">{value}</dd>
    </div>
  );
}

function StatsPanel({
  title,
  stats,
  unit,
  pixels,
  color,
  emphasis,
  alignMode,
}: {
  title: string;
  stats: GroupStats;
  unit: DisplayUnit;
  pixels: boolean;
  color?: string;
  emphasis?: boolean;
  alignMode: AlignMode;
}) {
  const f = pixels ? 1 : unitFactor(unit);
  const u = pixels ? "px" : unit;
  const v = (inches: number, digits = 2) => `${(inches * f).toFixed(digits)} ${u}`;
  const bias = Math.hypot(stats.centroidX, stats.centroidY);
  return (
    <div
      className={`rounded-xl border bg-neutral-950 p-3 ${emphasis ? "border-sky-500/40" : "border-gray-800"}`}
    >
      <div className="flex items-center gap-2">
        {color ? <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} /> : null}
        <h3 className="truncate text-sm font-semibold text-white">{title}</h3>
        <span className="ml-auto text-xs text-gray-500">{stats.count} shots</span>
      </div>
      <dl className="mt-2 grid grid-cols-1 gap-x-4 gap-y-1 text-[11px] sm:grid-cols-2">
        {alignMode === "aim" ? (
          <StatRow label="Bias from aim" value={v(bias)} />
        ) : null}
        <StatRow label="Group size" value={v(groupSize(stats))} />
        <StatRow label="Extreme spread" value={v(stats.extremeSpread)} />
        <StatRow label="Mean radius" value={v(stats.meanRadius)} />
        <StatRow label="CEP (50%)" value={v(stats.cep)} />
        <StatRow label="RMS radius" value={v(stats.rmsRadius)} />
        <StatRow label="Std dev X / Y" value={`${(stats.stdX * f).toFixed(2)} / ${(stats.stdY * f).toFixed(2)} ${u}`} />
        {stats.meanDiameter ? <StatRow label="Mean hole ⌀" value={v(stats.meanDiameter)} /> : null}
      </dl>
    </div>
  );
}

function ScatterPlot({
  points,
  colorById,
  unit,
  pixels,
  stats,
  alignMode,
}: {
  points: Pt[];
  colorById: Record<string, string>;
  unit: DisplayUnit;
  pixels: boolean;
  stats: GroupStats | null;
  alignMode: AlignMode;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
    const dpr = typeof window !== "undefined" ? Math.min(2, window.devicePixelRatio || 1) : 1;
    const cssSize = canvas.clientWidth || 480;
    const size = Math.round(cssSize * dpr);
    if (canvas.width !== size || canvas.height !== size) {
      canvas.width = size;
      canvas.height = size;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssSize, cssSize);
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, cssSize, cssSize);

    const f = pixels ? 1 : unitFactor(unit);
    const pad = 18;
    const plot = cssSize - pad * 2;
    const cx0 = pad + plot / 2;
    const cy0 = pad + plot / 2;

    // Extent in display units (symmetric around origin / centroid).
    let maxAbs = 0.5;
    for (const p of points) {
      maxAbs = Math.max(maxAbs, Math.abs(p.x) * f, Math.abs(p.y) * f);
    }
    maxAbs *= 1.15; // margin
    const scale = plot / 2 / maxAbs; // px per display-unit

    const toPx = (vx: number, vy: number) => ({
      px: cx0 + vx * f * scale,
      py: cy0 + vy * f * scale, // image orientation (y down), matching the in-app map
    });

    // Grid: nice step (1, 2, 5 × 10^k) covering the extent.
    const rawStep = maxAbs / 4;
    const pow = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const norm = rawStep / pow;
    const niceStep = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * pow;
    ctx.strokeStyle = "rgba(148,163,184,0.12)";
    ctx.fillStyle = "rgba(148,163,184,0.55)";
    ctx.lineWidth = 1;
    ctx.font = "10px sans-serif";
    for (let g = niceStep; g <= maxAbs; g += niceStep) {
      for (const sign of [1, -1]) {
        const gx = cx0 + sign * g * scale;
        const gy = cy0 + sign * g * scale;
        ctx.beginPath();
        ctx.moveTo(gx, pad);
        ctx.lineTo(gx, pad + plot);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(pad, gy);
        ctx.lineTo(pad + plot, gy);
        ctx.stroke();
      }
    }
    // Axes.
    ctx.strokeStyle = "rgba(148,163,184,0.35)";
    ctx.beginPath();
    ctx.moveTo(pad, cy0);
    ctx.lineTo(pad + plot, cy0);
    ctx.moveTo(cx0, pad);
    ctx.lineTo(cx0, pad + plot);
    ctx.stroke();

    // Aim origin (amber) + bias vector to the impact centroid, in aim mode.
    if (alignMode === "aim") {
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx0, cy0, 6, 0, Math.PI * 2);
      ctx.stroke();
      if (stats) {
        const c = toPx(stats.centroidX, stats.centroidY);
        ctx.strokeStyle = "rgba(251,191,36,0.6)";
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(cx0, cy0);
        ctx.lineTo(c.px, c.py);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    if (stats) {
      const c = toPx(stats.centroidX, stats.centroidY);
      // Mean-radius circle.
      ctx.strokeStyle = "rgba(56,189,248,0.5)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(c.px, c.py, stats.meanRadius * f * scale, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      // Centroid cross.
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(c.px - 6, c.py);
      ctx.lineTo(c.px + 6, c.py);
      ctx.moveTo(c.px, c.py - 6);
      ctx.lineTo(c.px, c.py + 6);
      ctx.stroke();
    }

    // Shots.
    for (const p of points) {
      const { px, py } = toPx(p.x, p.y);
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fillStyle = colorById[p.sessionId] ?? "#e5e7eb";
      ctx.globalAlpha = 0.85;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Scale label.
    ctx.fillStyle = "rgba(148,163,184,0.7)";
    ctx.font = "10px sans-serif";
    ctx.fillText(`grid = ${niceStep.toFixed(niceStep < 1 ? 2 : 0)} ${pixels ? "px" : unit}`, pad + 2, pad + plot - 4);
    };

    draw();
    const observer = new ResizeObserver(() => draw());
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [points, colorById, unit, pixels, stats, alignMode]);

  return (
    <div className="rounded-xl border border-gray-700 bg-neutral-950 p-3">
      <canvas ref={canvasRef} className="mx-auto block aspect-square w-full max-w-md" />
      {points.length === 0 ? (
        <p className="mt-2 text-center text-xs text-gray-500">No shots to plot.</p>
      ) : null}
    </div>
  );
}
